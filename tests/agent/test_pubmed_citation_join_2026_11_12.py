"""Sprint 2b: a `/pubmed` answer cited three records and the page drew four notes.

The live turn of 2026-11-12 answered "cervical spondylosis" with three PubMed records and
the browser drew, beside them, four reference rows titled "Not Available" and three pills
with the same label. The numbers in the prose were `362`, `393` and `405`; the three real
cards were numbered from `40591977`. Nothing was random and nothing was hallucinated: the
model wrote a record identifier without its label — `^[36208893]` — and the next layer
read the eight-digit number as a POSITION in the conversation's evidence.

`_conclude` is where that number was accepted. It treated `1 <= n <= limit or
str(n) in allowed` as one question with one answer, so an identifier that resolved only
because it was retrieved came back as "the marker is already correct" and was left
byte-for-byte. Everything downstream reads a bare number as a position:
`presentation.resolve_sources` looked up passage 36208893, found nothing, and published a
"not available" row where the record should have been. The page then numbered the real
records from the largest cite it had seen, which was the identifier.

So the number is now classified rather than tested once: in range is a position and is
written back unchanged; out of range but retrieved is an identifier and is written back
WITH its label; anything else is dropped and flagged as it always was. The label matters
because `^[PMID: n]` is the one shape every reader of an answer agrees is an identifier.

A second defect shipped in the same report and has its own fix: PMID 39320076 carries
`[Not Available].` as its `<ArticleTitle>` and the real title in `<VernacularTitle>`, so
one of the four placeholder rows was the record's own placeholder rather than a phantom.
`literature._title` reads the fallback field and strips matched brackets only.

Each test names the defect, so a regression points at its own cause.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import pytest

from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations
from services.agent.presentation import a2ui
from services.mcp.tools import literature

# The three records the live search returned, with the titles the endpoint states.
PMID_ONE = "36208893"
PMID_PLACEHOLDER = "39320076"
PMID_THREE = "40591976"


def _record(pmid: str, title: str, *, abstract: str) -> dict:
    return {"pmid": pmid, "title": title, "abstract": abstract,
            "journal": "The Orthopedic clinics of North America",
            "journal_abbrev": "Orthop Clin North Am", "pub_date": "2022 Oct",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}


#: The `/pubmed cervical spondylosis` search, as the tool returns it.
PUBMED_CALL = {
    "name": "search_literature",
    "args": {"query": "cervical spondylosis", "max_results": 3},
    "response": {
        "query": "cervical spondylosis",
        "returned": 3,
        "articles": [
            # 39320076 is stored with the title the fixed extractor produces, because
            # that is what the tool now returns for the record: the placeholder lives in
            # ArticleTitle and the real title in VernacularTitle.
            _record(PMID_ONE, "Degenerative Cervical Myelopathy: Evaluation and Management.",
                    abstract="Cervical spondylotic myelopathy is the commonest cause of "
                             "spinal cord dysfunction in older adults."),
            _record(PMID_PLACEHOLDER, "Cervical spondylotic myelopathy.",
                    abstract="Conservative treatment is first-line."),
            _record(PMID_THREE, "Diagnosis and Management of Degenerative Cervical Myelopathy.",
                    abstract="Surgery is offered for progressive deficits."),
        ],
    },
}

#: Five note sections, so the turn also has positions 1-5 to confuse an identifier with.
SECTIONS = {
    "name": "rag_search_sections",
    "args": {"hadm_id": 90000081},
    "response": {"returned": 5, "passages": [
        {"cite": number, "section": section, "text": f"The {section} section."}
        for number, section in enumerate(
            ["brief_hospital_course", "discharge_instructions", "admission_diagnosis",
             "past_medical_history", "physical_exam"], start=1)
    ]},
}


# --- The classification: a number is a position or an identifier, never both ---------

class TestABareIdentifierIsNotAPosition:
    def test_a_lone_bare_identifier_is_written_back_with_its_label(self):
        """The defect in one line. `^[36208893]` resolved — the number names a record this
        turn retrieved — so it was left alone, and "left alone" meant served in the one
        shape that no layer can read as a record."""
        answer = f"Myelopathy is the commonest cause.^[{PMID_ONE}]"

        kept, flags = citations.enforce_citations(answer, 0, {PMID_ONE})

        assert kept == f"Myelopathy is the commonest cause.^[PMID: {PMID_ONE}]"
        assert flags == []

    def test_every_record_of_a_cluster_gets_its_own_labelled_marker(self):
        """Three identifiers in one marker: one per record, each labelled, because the
        browser draws one control per marker."""
        answer = (f"Three papers agree.^[{PMID_ONE}, {PMID_PLACEHOLDER}, {PMID_THREE}]")

        kept, flags = citations.enforce_citations(
            answer, 0, {PMID_ONE, PMID_PLACEHOLDER, PMID_THREE})

        assert kept == (
            f"Three papers agree.^[PMID: {PMID_ONE}]"
            f" ^[PMID: {PMID_PLACEHOLDER}] ^[PMID: {PMID_THREE}]"
        )
        assert flags == []

    def test_a_position_is_still_left_byte_for_byte(self):
        """The half of the old rule that was right, and must stay right: a number in range
        is a position, and rewriting a correct citation is its own defect."""
        answer = "Her course is in the note.^[2]"

        kept, flags = citations.enforce_citations(answer, 5, set())

        assert kept == answer
        assert flags == []

    def test_a_cluster_of_positions_is_still_expanded_unlabelled(self):
        answer = "Both sections agree.^[1, 3]"

        kept, flags = citations.enforce_citations(answer, 5, set())

        assert kept == "Both sections agree.^[1] ^[3]"
        assert flags == []

    def test_a_cluster_may_hold_a_position_and_an_identifier(self):
        """The two schemes coexist on a page — five note sections and three records — so
        the classification is per number rather than per marker."""
        answer = f"The note says this^[1] and the trial agrees.^[3, {PMID_ONE}]"

        kept, flags = citations.enforce_citations(answer, 5, {PMID_ONE})

        assert kept == f"The note says this^[1] and the trial agrees.^[3] ^[PMID: {PMID_ONE}]"
        assert flags == []

    def test_an_identifier_the_turn_did_not_retrieve_is_still_removed_and_flagged(self):
        """The guardrail the fix must not weaken. `enforce_citations` exists to delete a
        fabricated identifier; only a retrieved one is relabelled."""
        answer = "A trial proved it.^[PMID: 99999999]"

        kept, flags = citations.enforce_citations(answer, 5, {PMID_ONE})

        assert kept == "A trial proved it."
        assert flags == ["citation_out_of_range:^99999999"]

    def test_a_fabricated_bare_number_that_is_neither_position_nor_record_is_removed(self):
        """Nothing about the classification makes an unknown number survivable."""
        answer = "A trial proved it.^[99999999]"

        kept, flags = citations.enforce_citations(answer, 5, {PMID_ONE})

        assert kept == "A trial proved it."
        assert flags == ["citation_out_of_range:^99999999"]


# --- What the reader receives: the served page, end to end --------------------------

THREE_SENTENCES = (
    f"Myelopathy is commonest in older adults.^[{{one}}] "
    f"Conservative care comes first.^[{{two}}] "
    f"Surgery is for progressive deficits.^[{{three}}]"
)

#: The thread's second turn: one note citation, then three records cited bare.
THREAD_PROSE = (
    f"Her course was uncomplicated.^[1] Myelopathy is commonest in older adults.^[{{one}}] "
    f"Conservative care comes first.^[{{two}}] "
    f"Surgery is for progressive deficits.^[{{three}}]"
)


def _serve(bare: bool) -> dict:
    """The reported thread, turn 2, settled and composed the way a live one is.

    The thread matters and is not decoration. On a `/pubmed` turn with no earlier note
    retrieval the conversation's array holds no passages, `resolve_sources` returns early,
    and the defect is invisible. It is the note passages of the previous turn that made
    36208893 look like a passage number the answer was allowed to address — which is what
    the reader's thread was: `/ehr`, then `/pubmed`.

    The presentation contract is what the browser consumes, so it is what the test reads:
    its `references` are the drawer rows and its `sources` are what a marker opens. A test
    that stopped at `resolve_sources` would have passed on the defective build, because the
    numbering that overflowed is decided after it.
    """
    marker = (lambda pmid: f"^{{{pmid}}}") if bare else (lambda pmid: f"^[PMID: {pmid}]")
    prose = THREAD_PROSE.format(
        one=marker(PMID_ONE), two=marker(PMID_PLACEHOLDER), three=marker(PMID_THREE))
    settled = answer_text.settle_answer(
        prose=prose, deterministic=None, tool_calls=[PUBMED_CALL],
        replayed_tool_calls=[SECTIONS], command="literature")
    return a2ui.compose_presentation(
        "pubmed cervical spondylosis", settled.prose, [PUBMED_CALL],
        evidence=[SECTIONS, PUBMED_CALL], command="literature")


def _published(presentation: dict) -> list[dict]:
    return [row for row in presentation["references"] if row.get("pmid")]


@pytest.mark.parametrize("bare", [True, False], ids=["bare", "labelled"])
def test_the_three_records_are_numbered_after_the_note_they_follow(bare):
    """The symptom that was reported: the cards were numbered `40591977`, `40591978`,
    `40591979`, because the highest citation index the page had seen was a PMID.

    Measured against the defective build, this test's subject produced exactly those three
    indices, with three rows at `36208893`/`39320076`/`40591976` labelled "not available"
    ahead of them.
    """
    presentation = _serve(bare)

    assert [row["index"] for row in _published(presentation)] == [2, 3, 4]
    assert [row["index"] for row in presentation["references"]] == [1, 2, 3, 4]


@pytest.mark.parametrize("bare", [True, False], ids=["bare", "labelled"])
def test_no_row_is_a_placeholder(bare):
    """The rows titled "Not Available". They were not records at all: each was
    `resolve_source` answering for a passage number that did not exist."""
    presentation = _serve(bare)

    for row in presentation["references"]:
        assert "not available" not in str(row.get("title") or "").lower(), row
        assert "not available" not in str(row.get("section") or "").lower(), row
        assert "information is available for this patient" not in str(
            row.get("text") or ""), row
    for row in presentation["sources"]:
        assert row.get("section") != "not available", row


@pytest.mark.parametrize("bare", [True, False], ids=["bare", "labelled"])
def test_no_index_is_a_record_identifier(bare):
    """The overflow guard: an identifier must not survive as a citation index. Eight-digit
    indices are the fingerprint of the defect, whatever route produced them."""
    presentation = _serve(bare)

    indexes = [row["index"] for row in presentation["references"]]
    assert all(index < 1000 for index in indexes), (
        f"a record identifier reached the page as a citation index: {indexes}")


@pytest.mark.parametrize("bare", [True, False], ids=["bare", "labelled"])
def test_the_rows_are_the_records_and_carry_their_metadata(bare):
    """Numbered after the note AND pointing at their own records — the join the report said
    had broken between the prose and the reference drawer."""
    presentation = _serve(bare)

    published = _published(presentation)
    assert [row["pmid"] for row in published] == [PMID_ONE, PMID_PLACEHOLDER, PMID_THREE]
    assert published[1]["title"] == "Cervical spondylotic myelopathy."


@pytest.mark.parametrize("bare", [True, False], ids=["bare", "labelled"])
def test_the_note_citation_still_opens_its_own_passage(bare):
    """The other half of the same page: fixing the identifier must not move the note."""
    presentation = _serve(bare)

    note = presentation["sources"][0]
    assert note["cite"] == 1
    assert note["section"] == "brief_hospital_course"


def test_a_note_turn_beside_a_literature_turn_keeps_its_own_numbering():
    """The first turn of the reported thread: five note sections and one record on one page,
    where the record resolved correctly. It must still."""
    settled = answer_text.settle_answer(
        prose="Her course is in the note.^[1] The trial reports 4.4%.^[PMID: 28142261]",
        deterministic=None, tool_calls=[SECTIONS], replayed_tool_calls=[],
        command="summarize")
    presentation = a2ui.compose_presentation(
        "summarise admission 90000081", settled.prose, [SECTIONS], command="summarize")

    assert presentation["sources"][0]["section"] == "brief_hospital_course"
    assert presentation["sources"][0]["cite"] == 1


# --- The record's own placeholder: title extraction ---------------------------------

def _article_xml(inner: str) -> ElementTree.Element:
    """One `PubmedArticle`, with the article body supplied by the caller."""
    return ElementTree.fromstring(
        "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>39320076</PMID>"
        f"<Article>{inner}</Article>"
        "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
    ).find(".//PubmedArticle")


class TestTheTitleComesFromTheRecord:
    def test_the_placeholder_falls_back_to_the_vernacular_title(self):
        """PMID 39320076 verbatim, read from efetch on 2026-11-12: the journal published no
        English title, so PubMed wrote its own placeholder into `ArticleTitle` and kept the
        Danish title in `VernacularTitle`.

        Serving "Not Available" above an abstract that is present tells the reader the
        record is empty.
        """
        article = _article_xml(
            "<ArticleTitle>[Not Available].</ArticleTitle>"
            "<VernacularTitle>Cervical spondylotic myelopathy.</VernacularTitle>"
        )

        assert literature._title(article) == "Cervical spondylotic myelopathy."

    def test_the_placeholder_with_no_fallback_becomes_a_clean_label(self):
        """Two fields, both empty of information: the record is real and linkable by its
        PMID, so it is labelled rather than shown as the XML's own words."""
        article = _article_xml("<ArticleTitle>[Not Available].</ArticleTitle>")

        assert literature._title(article) == literature.UNTITLED_ARTICLE

    def test_a_translated_title_loses_its_brackets_and_keeps_its_full_stop(self):
        """`[Cervical spondylotic myelopathy].` is a real title in the article's own
        language. The brackets are markup; the full stop is the title's."""
        article = _article_xml(
            "<ArticleTitle>[Cervical spondylotic myelopathy].</ArticleTitle>")

        assert literature._title(article) == "Cervical spondylotic myelopathy."

    def test_a_translated_title_outranks_its_english_rendering(self):
        """ArticleTitle is what the journal published, so a bracketed title there is used
        rather than replaced by the VernacularTitle: the fallback is for placeholders, not
        for translations."""
        article = _article_xml(
            "<ArticleTitle>[Zervikale Myelopathie].</ArticleTitle>"
            "<VernacularTitle>Cervical myelopathy.</VernacularTitle>"
        )

        assert literature._title(article) == "Zervikale Myelopathie."

    def test_an_unmatched_bracket_is_left_exactly_as_the_record_wrote_it(self):
        """The reason the strip is a matched pair. Stripping the opening bracket
        unconditionally turned `[Not Available].` into `Not Available].` — a string that is
        neither the record's title nor a clean label, and which the reader cannot read as
        either."""
        article = _article_xml("<ArticleTitle>[Cervical myelopathy.</ArticleTitle>")

        assert literature._title(article) == "[Cervical myelopathy."

    def test_a_plain_title_is_untouched(self):
        article = _article_xml(
            "<ArticleTitle>Diagnosis and Management of Degenerative Cervical "
            "Myelopathy.</ArticleTitle>")

        assert literature._title(article) == (
            "Diagnosis and Management of Degenerative Cervical Myelopathy.")

    def test_a_title_wrapped_in_markup_is_still_read_whole(self):
        """Unchanged from `_text`: PubMed wraps spans in `<i>`, and reading `.text` alone
        truncates the title at the first tag."""
        article = _article_xml(
            "<ArticleTitle>Ceftriaxone for <i>lower extremity</i> edema</ArticleTitle>")

        assert literature._title(article) == "Ceftriaxone for lower extremity edema"

    def test_the_placeholder_never_reaches_the_articles_list(self):
        """The aggregate check: whatever field the placeholder hides in, no record leaves
        the tool titled with it, because a title is what the reference row prints."""
        root = ElementTree.fromstring("""
        <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>39320076</PMID>
          <Article><ArticleTitle>[Not Available].</ArticleTitle>
            <VernacularTitle>Cervical spondylotic myelopathy.</VernacularTitle>
          </Article>
        </MedlineCitation></PubmedArticle>
        <PubmedArticle><MedlineCitation><PMID>1</PMID>
          <Article><ArticleTitle>[Not Available].</ArticleTitle></Article>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """)

        assert [a["title"] for a in literature._articles(root)] == [
            "Cervical spondylotic myelopathy.", literature.UNTITLED_ARTICLE]
