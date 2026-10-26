"""The records an answer's citations resolve to.

Everything between the model's `^[1]` markers and the rows a reader can click: the
passage a marker points at, the section label on that row, the drawer record for a
note citation and the drawer record for an article, and the narrative surface that
carries them.

Split out of `a2ui.py`, which is the composer. Nothing here builds an envelope or
decides what a turn is drawn as — it answers "what is this citation, and what does a
reader see for it", which is a question about the evidence rather than about the
canvas. `a2ui` re-exports every name here, because the tests and `guardrail` reach
them through it and the module boundary is not worth a second call path.

`_usable_articles` lives here rather than with the canvas that also reads it: it is
the rule for what may be cited at all (a record with no pmid cannot be linked), and
the canvas is a consumer of that rule. The notes that describe an EMPTY or
UNAVAILABLE search stay with the canvas, because they are wording for a surface
that did not get any records.
"""

from services.agent.citations import (
    citable_passages,
    cited_numbers,
    extract_section,
)

# The fields a literature record is put on the wire with, in the order the card shows
# them.
#
# The abstract was removed from this list on 2026-10-13 with the ruling that "the
# reader's next step is the record at PubMed, not a second copy of its text here" — at
# the time the only surface was a card that showed a title and a journal. Phase F
# (2026-10-26) puts the record behind a citation marker like every other source in the
# product, and a reference that opens onto a title and nothing else is not a reference:
# the reader cannot weigh a claim whose evidence is a heading. The abstract is the
# passage, so it travels.
_LITERATURE_FIELDS = ("pmid", "title", "journal", "pub_date", "authors", "abstract",
                      "url")

def _usable_articles(literature) -> list[dict]:
    """The articles from a literature payload, each shaped for the canvas.

    A record with no pmid is dropped: it cannot be cited or linked, which is the
    whole use of it. The remaining fields are coerced to strings rather than
    passed through, because the console's component schema is strict — one
    missing key drops the entire card, and a card that fails to draw is worse
    than a card with a blank journal.

    An empty list is the answer for three different situations — no search was
    run, the search failed, or it matched nothing — which the canvas treats the
    same and the answer text and provenance caption distinguish between.
    """
    if not isinstance(literature, dict):
        return []
    articles = literature.get("articles")
    if not isinstance(articles, list):
        return []
    usable = []
    for article in articles:
        if not isinstance(article, dict) or not article.get("pmid"):
            continue
        usable.append({
            field: article.get(field) if isinstance(article.get(field), str) else ""
            for field in _LITERATURE_FIELDS
        })
    return usable


def _unavailable_text(section: str) -> str:
    """The deterministic message for a question whose target section the note
    does not have. Never mine the content from unrelated narrative."""
    if section == "discharge_medications":
        return "No discharge medication information is available for this patient."
    return f"No {section.replace('_', ' ')} information is available for this patient."


def resolve_source(passages: list[dict], passage_n: int,
                   sections: tuple[str, ...] = ()) -> dict:
    """Resolve one citation to the passage and section body it supports.

    `passage_n` is the model's ORIGINAL 1-based passage number (before
    renumbering). `sections` is the question's section intent; when given, the
    section is located by label or extracted from any whole-note chunk, and
    the model's number is ignored — the model mis-numbers citations (a meds
    answer cites ^[1] while its supporting passage sits elsewhere).

    Returns {"section", "text"}. Callers own the badge number and the query.
    """
    cited = None
    intent_body = None
    matched_section = None
    for sec in sections:
        cited = next((p for p in passages if p.get("section") == sec), None)
        if cited is not None:
            matched_section = sec
            break
        # The index stores whole-note chunks: a passage labeled with a
        # different section still CONTAINS the target section, and the
        # labeled chunk can miss the top-k (near-tied whole-note
        # embeddings). Pull the section body out of the passage text.
        for p in passages:
            body = extract_section(p.get("text", ""), sec)
            if body:
                cited = p
                intent_body = body
                matched_section = sec
                break
        if cited is not None:
            break

    if cited is None and sections:
        # The note has NONE of the targeted sections. The deterministic answer
        # is "not available", not a passage mined from unrelated narrative.
        return {"section": "not available",
                "text": _unavailable_text(sections[0])}

    if cited is None:
        # Clamp to a real passage so an out-of-range number never raises.
        idx = min(max(passage_n, 1), len(passages)) - 1
        cited = passages[idx]
        shown_section = cited.get("section", "discharge note")
        section_text = extract_section(cited.get("text", ""), shown_section)
    elif intent_body is not None:
        shown_section = matched_section
        section_text = intent_body
    else:
        shown_section = matched_section or cited.get("section", "discharge note")
        section_text = extract_section(cited.get("text", ""), shown_section)

    return {"section": shown_section,
            "text": section_text or cited.get("text", ""),
            # What a citation for this passage calls itself, when the passage is not a
            # note section. A laboratory trajectory is evidence of a different kind — it
            # has an analyte rather than a heading — so it names itself here and
            # `note_citations` uses the name instead of a section label it would have to
            # guess at.
            "cite_source": str(cited.get("cite_source") or ""),
            "cite_title": str(cited.get("cite_title") or "")}


#: The badge a published record carries. One word, because a pill is a tag beside a
#: title rather than a sentence about it.
_LITERATURE_BADGE = "PubMed"


def _surface_components(canvas: dict | None):
    """Every component a composed surface carries, in the order it draws them.

    The components are not on the envelope: an A2UI surface is a list of protocol
    messages, and the component tree is the `updateComponents` payload of one of them.
    Reading `canvas["components"]` found nothing and would have gone on finding nothing,
    because the envelope is valid without that key.
    """
    for message in (canvas or {}).get("messages") or []:
        if not isinstance(message, dict):
            continue
        update = message.get("updateComponents")
        if not isinstance(update, dict):
            continue
        for component in update.get("components") or []:
            if isinstance(component, dict):
                yield component


def register_benchmark_sources(canvas: dict | None, start: int,
                              existing_pmids=()) -> tuple[list[dict], list[dict]]:
    """Give the comparison table's published cohorts a citation identity.

    `ComparativeRiskMatrix` draws curated PubMed cohorts beside this patient's estimate,
    and it is built OUTSIDE the prose citation pipeline: the rows come from
    `literature_benchmarks`, selected by the admission's phenotype, not from a passage
    the answer cites. So the table arrived with a journal, a year and a PMID on screen,
    and the turn arrived with no record of any of them — the references drawer listed
    `1..4` (the note sections) and the benchmark, which the whole comparison rests on,
    could not be opened from the turn that showed it.

    This registers each benchmark row as a source at `start` and onwards, and stamps the
    row itself with `source_index` / `source_label` so the surface can draw the pill
    beside it. Two lists are returned because the turn carries two: `sources` is the
    citation channel (numbered, `cite`), and `references` is the drawer's own rows. They
    are built together here so a number written in one cannot disagree with the other.

    Called AFTER every flag derived from `sources` has been computed. The cache of note
    passages is what `grounded_in_record` and `scope_suppressed` ask about — whether the
    answer rests on THIS patient's record — and a published cohort about somebody else's
    patients must not be able to answer yes to that question by being appended to the
    list it is asked of.
    """
    sources: list[dict] = []
    references: list[dict] = []
    seen = {str(pmid).strip() for pmid in existing_pmids if str(pmid or "").strip()}
    index = start
    for component in _surface_components(canvas):
        if component.get("component") != "ComparativeRiskMatrix":
            continue
        for row in component.get("rows") or []:
            if not isinstance(row, dict):
                continue
            pmid = str(row.get("pmid") or "").strip()
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            # What the table calls the row: the journal and the year. It stays the pill's
            # label, because a pill is set inside a sentence or a cell and a full article
            # title is not.
            citation = str(row.get("source") or "").strip() or f"PMID {pmid}"
            label = f"{index} · {citation}"
            url = str(row.get("url") or "").strip() or (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            # The article, which is what the record IS. A row that named only its journal
            # left the drawer showing `5. The Journal of Arthroplasty · 2022` and nothing
            # else — no paper, no authors, nothing to weigh it by.
            title = str(row.get("title") or "").strip() or citation
            authors = str(row.get("authors") or "").strip()
            journal = str(row.get("journal") or "").strip()
            year = str(row.get("year") or "").strip()
            # A citation line, in the order a reader reads one: where it was published,
            # when, by whom, and which record it is.
            journal_citation = " ".join(
                part for part in (
                    f"{journal}." if journal else "",
                    f"{year}." if year else "",
                    authors,
                ) if part
            ).strip()
            subtitle = " · ".join(
                part for part in (journal_citation, f"PMID {pmid}") if part
            )
            # The cohort and the sentence the rate was read from: what the popover and the
            # drawer show for the reader to weigh the comparison rather than take it.
            passage = " — ".join(
                part for part in (str(row.get("cohort_summary") or "").strip(),
                                  str(row.get("excerpt") or "").strip())
                if part
            )
            # Stamped on the row so the surface draws the pill from the same number,
            # rather than resolving a PMID back to an index a second time.
            row["source_index"] = index
            row["source_label"] = label
            sources.append({
                "cite": index,
                "kind": "Literature",
                "badge": _LITERATURE_BADGE,
                # The article, not its journal: this is the field the card's heading and
                # the drawer's row are titled from.
                "title": title,
                "authors": authors,
                "label": label,
                "subtitle": subtitle,
                "journal_citation": journal_citation,
                "section": citation,
                "text": passage,
                "tier": str(row.get("evidence_tier") or "").strip(),
                "pmid": pmid,
                "url": url,
            })
            references.append({
                "index": index,
                "source": _LITERATURE_BADGE,
                # The drawer's own row, shaped like the note rows beside it: a title, a
                # subtitle and the passage. `title` is the article — a row headed with its
                # journal was the naked `5. The Journal of Arthroplasty · 2022` a reader
                # reported, with nothing to weigh it by.
                "title": title,
                "label": label,
                "subtitle": subtitle,
                "authors": authors,
                "excerpt": passage,
                "pmid": pmid,
                "url": url,
            })
            index += 1
    return sources, references


def resolve_sources(answer: str, citation_map: dict[str, int],
                    rag: dict | None, sections: tuple[str, ...],
                    evidence: list[dict] | None = None) -> list[dict]:
    """One resolved source per citation number in the (renumbered) answer.

    This is the list the browser renders on a footnote click: it looks up the
    clicked number and displays the entry — no vocabulary, no heuristics.
    Section intent is applied only to a single-citation answer; a
    multi-citation answer cites its passages in array order and maps by
    number.

    `evidence` is the whole conversation's tool calls, in the order the model was shown
    them, and it supersedes `rag` when given: a marker addresses ONE array, and that array
    holds every citable unit the conversation produced — four note sections from `/ehr`,
    then ten analytes from a lab drill-down — not only the passages of a single retrieval
    payload. Resolving against the retrieval alone left a lab citation pointing at a note
    passage or at nothing (Phase E, 2026-10-26).
    """
    if evidence is not None:
        passages = [
            passage for call in evidence
            for passage in citable_passages((call or {}).get("response"))
        ]
    else:
        passages = (rag or {}).get("passages") or []
    if not passages:
        return []
    query = (rag or {}).get("query") or "discharge note"
    cites = cited_numbers(answer)
    if not cites:
        return []
    intent = sections if len(cites) == 1 else ()
    return [
        {"cite": n,
         **resolve_source(passages, citation_map.get(str(n), n), intent),
         "query": query}
        for n in cites
    ]


# The section names as a clinician reads them. A separate vocabulary from
# `feature_labels`, which names MODEL features: these name parts of a note, and the
# two happen to share the word "discharge" without meaning the same thing by it.
_SECTION_LABELS = {
    "history_of_present_illness": "History of present illness",
    "brief_hospital_course": "Brief hospital course",
    "discharge_diagnosis": "Discharge diagnosis",
    "discharge_medications": "Discharge medications",
    "medications_on_transfer": "Medications on transfer",
    "discharge_instructions": "Discharge instructions",
    "discharge_condition": "Discharge condition",
    "discharge_disposition": "Discharge disposition",
    "followup_instructions": "Follow-up instructions",
    # Not a note heading: a laboratory trajectory, the other channel a chart answer can
    # rest on. It is here because this map is where a section key becomes words a reader
    # sees, and a lab citation that fell through to the generic branch would read
    # "laboratory trends" in lower case beside "Discharge diagnosis".
    "laboratory_trends": "Laboratory trends",
}


def _section_label(section: object) -> str:
    name = str(section or "")
    return _SECTION_LABELS.get(name, name.replace("_", " ") or "Note passage")


def narrative_surface(answer: str, citations: list[dict]) -> dict:
    """The prose, and the evidence its markers point at, as one component.

    The markers and the sentences they annotate travel together because a marker is a
    control the reader clicks while reading the sentence it sits in. The answer used to
    be drawn as plain text in a separate element with nothing to click, because the
    basic Text component renders Markdown and Markdown cannot host a button.

    `citations` are built by the caller from the evidence this turn actually rests on;
    see `note_citations` and `literature_citations`. Each entry's `index` is the
    number the marker shows, so the two cannot drift.

    There is no `streaming` key. There was one — a boolean the component read to draw
    its markers inert while the answer was still arriving — and no caller could ever set
    it: this envelope is composed once the turn has finished, so the value was `False`
    on every payload that ever reached a browser. The draft that fills the canvas while
    the answer is being written is drawn by the page (Phase 5B), not by this component,
    so the key was removed from the contract rather than left as a promise nothing kept.
    """
    return {"text": answer, "citations": citations}


def note_citations(sources: list[dict]) -> list[dict]:
    """One citation per resolved source, keyed by its renumbered marker.

    The citation layer renumbers `^[n]` to first-appearance order; reading the
    renumbered `cite` here keeps the marker and the popover in step without a second
    mapping to drift.

    A source that names itself wins over the section vocabulary. A note passage does not
    need to — every note passage is a discharge note and differs only by section — but a
    laboratory trajectory is not a note passage at all, and labelling it "Discharge note"
    would tell the reader their creatinine came out of the prose.
    """
    citations = []
    for source in sources:
        title = str(source.get("cite_title") or "") or _section_label(
            source.get("section"))
        citations.append({
            "index": int(source["cite"]),
            # The channel the evidence came from. Note passages all share one channel
            # and differ by section, so the section becomes the title.
            "source": str(source.get("cite_source") or "Discharge note"),
            "title": title,
            # What the marker SAYS, resolved here rather than in the browser. The marker
            # reads `1 · Hospital Course` and not a bare `1`, because a number tells the
            # reader nothing about what they are about to open — and resolving it here
            # keeps the section vocabulary in the one module that already owns it
            # (`_SECTION_LABELS`), rather than a second copy in the client that would
            # drift the first time a section is renamed.
            "label": f"{int(source['cite'])} · {title}",
            "excerpt": str(source.get("text") or ""),
        })
    return citations


#: How long a literature marker's source token may get before it is cut back.
#:
#: A journal name is unbounded — "Journal of the American College of Cardiology" is
#: forty-seven characters — and an inline marker is a pill set inside a sentence, so an
#: unbounded label is a pill whose own box wraps to a second line and reads as a broken
#: one. Asked for by a reader looking at exactly that, 2026-10-16.
#:
#: Twenty-four characters keeps "Home healthcare now" and "Patient education and…"
#: recognisable. The full title is a hover away on the marker's popover and a line below in
#: the record list, which is where a reader goes to weigh a source anyway.
_CITE_SOURCE_CHARS = 24


def _short_source(name: str) -> str:
    """A source name cut back to the marker's budget, at a word boundary.

    At a boundary rather than mid-word, and with an ellipsis rather than silently: a cut
    that lands inside a word reads as a typo, and one with nothing to mark it reads as the
    whole name.
    """
    text = " ".join(str(name or "").split())
    if len(text) <= _CITE_SOURCE_CHARS:
        return text
    cut = text[:_CITE_SOURCE_CHARS]
    space = cut.rfind(" ")
    if space > 8:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


def _year_of(pub_date: object) -> str:
    """The four-digit year a PubMed date states, or an empty string.

    PubMed dates are not one format — "2023", "2023 Aug 01", "2022 Sep-Oct" — so this
    reads the year off rather than parsing a date. Nothing here needs a day.
    """
    for token in str(pub_date or "").replace("-", " ").split():
        if len(token) == 4 and token.isdigit():
            return token
    return ""


#: How much of an abstract a hover card carries.
#:
#: The card is 340px wide and floats over the answer being read, so it is a place to judge
#: a source, not a place to read one. 400 characters is the opening of a structured
#: abstract — enough to see what the study did and found — and the record itself is one
#: click away on the title above it. Cut at a sentence end where there is one inside the
#: budget, because a cut mid-clause reads as a truncation error rather than as a summary.
_CITE_EXCERPT_CHARS = 400


def _excerpt(text: object) -> str:
    """The opening of an abstract, bounded and marked as an opening."""
    body = " ".join(str(text or "").split())
    if len(body) <= _CITE_EXCERPT_CHARS:
        return body
    cut = body[:_CITE_EXCERPT_CHARS]
    stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if stop > _CITE_EXCERPT_CHARS // 2:
        return cut[: stop + 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).rstrip(" ,.;:") + "…"


def _article_record(article: dict, index: int) -> dict:
    """One published record, in the shape every other citable source already has.

    Built once and read by both channels — the narrative's citations and the turn's
    `sources` — so a marker, a hover card and a drawer row cannot disagree about which
    paper they name. Phase F (2026-10-26): a literature turn used to be the one surface
    with its own presentation, a boxed card under the prose and a marker that opened
    PubMed directly; what it never had was the hover card and the reference row that
    every other citation in the product gets.
    """
    journal = str(article.get("journal") or "").strip()
    year = _year_of(article.get("pub_date"))
    authors = str(article.get("authors") or "").strip()
    pmid = str(article.get("pmid") or "").strip()
    title = str(article.get("title") or "Untitled record").strip()
    # The citation line, in the order a reader reads one: where it was published, when,
    # by whom, and which record it is. Every part is optional except the identifier,
    # because a record that states no author must not render as "et al." against nobody.
    journal_citation = " ".join(part for part in (
        f"{journal}." if journal else "",
        f"{year}." if year else "",
        authors,
    ) if part).strip()
    subtitle = " · ".join(part for part in (journal_citation, f"PMID {pmid}")
                           if part)
    # What the pill SAYS. The journal and the year, because a pill is set inside a
    # sentence and a title is not; cut to a marker's length by `_short_source`, which is
    # the same budget the comparative table's pill uses.
    label = f"{index} · " + " · ".join(part for part in (
        _short_source(journal) or "PubMed", year) if part)
    return {
        "index": index,
        "kind": "Literature",
        "badge": _LITERATURE_BADGE,
        "title": title,
        "journal": journal,
        "year": year,
        "authors": authors,
        "label": label,
        "subtitle": subtitle,
        "journal_citation": journal_citation,
        # The passage: what the record reports. A reference that opens onto a title and
        # nothing else cannot be weighed.
        "text": _excerpt(article.get("abstract")),
        "pmid": pmid,
        "url": str(article.get("url") or "").strip() or (
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""),
    }


def literature_citations(articles: list[dict]) -> list[dict]:
    """One citation per retrieved article, shaped like every other citation.

    `index` is the 1-based position in the record order the model was given, which is
    exactly what the formatting instruction tells it to cite — the marker in the prose and
    the record beneath it are the same record by construction.

    The marker's DISPLAY is the number, the journal and the year; the identifier stays the
    link target so a reader who wants the paper is one click from the card's title. That
    trade was made on 2026-10-15 (journal over identifier), revisited on 2026-10-16 (the
    pills were reading long, so the journal is cut to `_CITE_SOURCE_CHARS`), and completed
    here: the record now also carries the article, its authors and its abstract, because
    the card behind the pill is the same card a note citation opens.

    `source` is the EYEBROW the card shows under the title. For a note passage that is
    "Discharge note" and the section is the title; for a published record the title is the
    article, so the eyebrow is the citation line itself.
    """
    citations = []
    for position, article in enumerate(articles, start=1):
        record = _article_record(article, position)
        citation = {
            "index": record["index"],
            "source": record["subtitle"],
            "title": record["title"],
            "label": record["label"],
            "subtitle": record["subtitle"],
            "authors": record["authors"],
            "journal_citation": record["journal_citation"],
            "excerpt": record["text"],
        }
        if record["pmid"]:
            citation["pmid"] = record["pmid"]
        if record["url"]:
            citation["url"] = record["url"]
        citations.append(citation)
    return citations


def literature_sources(articles: list[dict], start: int = 1) -> list[dict]:
    """The retrieved records as `sources` entries, numbered from `start`.

    A literature turn had no `sources` at all: the records lived in the `literature`
    payload, which the canvas drew as cards, and the citation channel held only note
    passages. The consequence was a turn whose markers could not address anything — the
    browser resolved `^[PMID: n]` by identifier as a special case, and a `^[1]` written on
    a literature turn pointed at nothing.

    They are `kind="Literature"` for the same reason the comparison table's cohorts are:
    a published record was not retrieved from THIS patient's note, so it carries the
    identifier and the link a reader can open instead of a retrieval query.
    """
    sources = []
    for offset, article in enumerate(articles):
        record = _article_record(article, start + offset)
        sources.append({
            "cite": record["index"],
            "kind": record["kind"],
            "badge": record["badge"],
            "title": record["title"],
            "authors": record["authors"],
            "label": record["label"],
            "subtitle": record["subtitle"],
            "journal_citation": record["journal_citation"],
            # The section a note passage would name. For a published record it is the
            # citation tram, which is what the table's benchmark rows use too.
            "section": " · ".join(part for part in (record["journal"], record["year"])
                                  if part) or record["badge"],
            "text": record["text"],
            "tier": "",
            "pmid": record["pmid"],
            "url": record["url"],
        })
    return sources


def _narrative(answer: str, sources: list[dict], literature: dict | None) -> dict | None:
    """The narrative surface for this answer, or None when there is no prose.

    Citations come from whichever evidence the turn rests on. A literature turn cites its
    articles and nothing else: those records ARE the evidence, and resolving the markers
    against note passages would leave every marker in a summary unbacked — which the
    component draws as a disabled control, on a summary that was properly grounded.

    The turn is only a literature turn when it produced RECORDS. An empty search result
    is a `dict`, so a bare `is not None` test made every chart answer that happened to
    carry a fruitless PubMed call cite zero articles: the note passages had already been
    resolved, and this function threw them away in favour of an empty list. The reader
    then had a `^[1]` that resolved to nothing beside a card saying no literature was
    found (2026-10-22). Records or note sources — never an empty list over a full one.
    """
    if not answer.strip():
        return None
    articles = _usable_articles(literature or {})
    if articles and literature is not None:
        return narrative_surface(answer, literature_citations(articles))
    return narrative_surface(answer, note_citations(sources))
