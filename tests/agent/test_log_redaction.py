"""The NCBI credential must not reach a log line.

`httpx` logs every request URL at INFO, and E-utilities takes its API key as a query
parameter — so the key was being written to Cloud Logging on every literature search.
These pin the redaction, and equally pin that it did not cost the request line, which
is the part worth keeping.
"""

import logging


def _httpx_request_log():
    return logging.getLogger("httpx")


def test_an_ncbi_request_url_is_redacted(caplog):
    import services.mcp.server  # noqa: F401  — installing the filter is the point

    with caplog.at_level(logging.INFO, logger="httpx"):
        _httpx_request_log().info(
            'HTTP Request: %s %s "%s %d %s"',
            "GET",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            "?tool=ecc-clinical-copilot&email=someone@example.com&api_key=deadbeefcafe",
            "HTTP/1.1",
            200,
            "OK",
        )

    assert "deadbeefcafe" not in caplog.text, "the API key reached the log"
    assert "someone@example.com" not in caplog.text, "the contact address reached the log"
    assert "api_key=<redacted>" in caplog.text
    assert "email=<redacted>" in caplog.text


def test_the_redaction_keeps_the_diagnosis(caplog):
    """The line that showed both NCBI calls answering 200 has to survive.

    Suppressing httpx's logging would have removed the leak just as well and would
    have removed the evidence with it.
    """
    import services.mcp.server  # noqa: F401

    with caplog.at_level(logging.INFO, logger="httpx"):
        _httpx_request_log().info(
            'HTTP Request: %s %s "%s %d %s"',
            "GET",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?tool=ecc-clinical-copilot&email=someone@example.com&api_key=deadbeefcafe"
            "&db=pubmed&id=36318680&retmode=xml",
            "HTTP/1.1",
            200,
            "OK",
        )

    assert "efetch.fcgi" in caplog.text
    assert "36318680" in caplog.text
    assert "200" in caplog.text


def test_a_request_without_credentials_is_untouched(caplog):
    """The filter is narrow: only the named parameters are rewritten."""
    import services.mcp.server  # noqa: F401

    with caplog.at_level(logging.INFO, logger="httpx"):
        _httpx_request_log().info(
            'HTTP Request: %s %s "%s %d %s"',
            "POST",
            "https://us-east1-aiplatform.googleapis.com/v1beta1/projects/x:predict",
            "HTTP/1.1",
            200,
            "OK",
        )

    assert "us-east1-aiplatform.googleapis.com" in caplog.text
    assert "v1beta1/projects/x:predict" in caplog.text
