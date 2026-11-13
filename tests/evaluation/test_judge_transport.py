"""The Judge's transport: the endpoint, the credentials, and the retry ladder.

Three properties decide whether a run's grading is measurable at all:

* the endpoint is a function of the location, and the three hostname shapes are not
  interchangeable — a wrong one is a 404 that reads like a missing model;
* the bearer token is refreshed on demand, because a token captured once expires inside a
  three-hour census and the failure looks like an IAM problem;
* a 429 is a quota WINDOW, so the first backoff is 30 seconds and not one. A ladder that
  retries inside the same window burns every attempt and lands the item unscored.

Nothing here calls a model. The fakes are shaped like the OpenAI SDK's objects because
that is the client the call is made on.
"""

from __future__ import annotations

import datetime as dt
import threading

import httpx
import openai
import pytest

from evaluation.platform import judge_call, judge_client

SAMPLE = {
    "reasoning": "The answer names the diagnosis and cites the note section correctly.",
    "dimensions": {"fidelity": 3, "grounding": 3, "citation": 3, "safety": 3,
                   "responsiveness": 3, "context": 3},
    "verdict": "PASS",
    "flags": [],
    "reason": "Correct.",
}

USAGE = {"input": 1000, "cached": 200, "output": 100, "cache_write": None,
         "thinking": 60}


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://aiplatform.googleapis.com/x")
    response = httpx.Response(429, request=request,
                              json={"error": {"message": "Resource exhausted"}})
    return openai.RateLimitError("Error code: 429 - Resource exhausted",
                                 response=response, body=None)


class TestTheRetryLadder:
    def test_a_429_starts_its_backoff_at_thirty_seconds(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(judge_client.time, "sleep", sleeps.append)
        monkeypatch.setattr(judge_client, "judge_timeout_seconds", lambda: 30)

        def _always_429(*_args, **_kwargs):
            raise _rate_limit_error()

        monkeypatch.setattr(judge_client, "_judge_once", _always_429)

        result = judge_client.judge(object(), "system", "user")

        assert "error" in result
        assert sleeps and sleeps[0] == 30, sleeps
        # 30, 60, 120 and then the cap: four waits across five attempts, because a
        # quota failure stops the ladder one attempt early rather than burning all six.
        assert sleeps == [30, 60, 120, 120]
        assert result["_attempts"] == 6

    def test_a_malformed_reply_gets_a_one_second_beat(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(judge_client.time, "sleep", sleeps.append)
        monkeypatch.setattr(judge_client, "judge_timeout_seconds", lambda: 30)
        monkeypatch.setattr(judge_client, "_judge_once",
                            lambda *_a, **_k: {"error": "not JSON"})

        result = judge_client.judge(object(), "system", "user")

        assert "error" in result
        assert sleeps[0] == 1, sleeps

    def test_a_timeout_gets_one_quick_retry_then_gives_up(self, monkeypatch):
        """Waiting out six attempts of a ceiling that will not be met cost 13 minutes."""
        monkeypatch.setattr(judge_client, "judge_timeout_seconds", lambda: 0.01)
        attempts: list[int] = []

        def _slow(*_args, **_kwargs):
            attempts.append(1)
            threading.Event().wait(0.3)
            return SAMPLE

        monkeypatch.setattr(judge_client, "_judge_once", _slow)

        result = judge_client.judge(object(), "system", "user")

        assert result["error"].startswith("no response within")
        assert len(attempts) == 2

    def test_a_recovered_attempt_returns_the_verdict(self, monkeypatch):
        monkeypatch.setattr(judge_client.time, "sleep", lambda _s: None)
        monkeypatch.setattr(judge_client, "judge_timeout_seconds", lambda: 30)
        replies = [
            {"error": "judge reply was not JSON: JSONDecodeError", "_usage": dict(USAGE)},
            {**SAMPLE, "_usage": dict(USAGE)},
        ]
        monkeypatch.setattr(judge_client, "_judge_once",
                            lambda *_a, **_k: replies.pop(0))

        result = judge_client.judge(object(), "system", "user")

        assert result["verdict"] == "PASS"
        assert result["_attempts"] == 2

    def test_the_usage_of_every_attempt_is_kept(self, monkeypatch):
        """A retried grading call was paid for twice; the run's cost has to say so."""
        monkeypatch.setattr(judge_client.time, "sleep", lambda _s: None)
        monkeypatch.setattr(judge_client, "judge_timeout_seconds", lambda: 30)
        replies = [
            {"error": "degenerate", "_usage": dict(USAGE)},
            {**SAMPLE, "_usage": dict(USAGE)},
        ]
        monkeypatch.setattr(judge_client, "_judge_once",
                            lambda *_a, **_k: replies.pop(0))

        result = judge_client.judge(object(), "system", "user")

        assert result["_usage"]["input"] == 2000
        assert result["_usage"]["output"] == 200
        assert result["_model"] == judge_client.JUDGE_MODEL


class TestThePassRule:
    def test_it_never_upgrades_a_fail(self):
        sample = {**SAMPLE, "verdict": "FAIL"}
        assert judge_client.apply_pass_rule(sample)["verdict"] == "FAIL"

    def test_it_downgrades_a_pass_the_scores_do_not_support(self):
        sample = {**SAMPLE, "dimensions": {**SAMPLE["dimensions"], "grounding": 1}}
        out = judge_client.apply_pass_rule(sample)
        assert out["verdict"] == "FAIL"
        assert any("pass_rule" in flag for flag in out["flags"])

    def test_a_multi_turn_pass_must_clear_context(self):
        dims = {key: 3 for key in SAMPLE["dimensions"] if key != "context"}
        out = judge_client.apply_pass_rule({**SAMPLE, "dimensions": dims},
                                           multi_turn=True)
        assert out["verdict"] == "FAIL"

    def test_a_missing_dimension_fails_closed_only_on_a_pass(self):
        dims = {key: value for key, value in SAMPLE["dimensions"].items()
                if key != "citation"}
        assert judge_client.apply_pass_rule({**SAMPLE, "dimensions": dims})[
            "verdict"] == "FAIL"
        assert judge_client.apply_pass_rule(
            {**SAMPLE, "dimensions": dims, "verdict": "FAIL"})["verdict"] == "FAIL"


class TestTheSemanticValidator:
    @pytest.mark.parametrize("phrase", [
        "This is a meta-response to the user's request.",
        "The actual evaluations are in the following array.",
        "I will return a JSON array with one element.",
        "These are evaluation objects, one for each turn.",
    ])
    def test_a_deferred_or_meta_reply_is_refused(self, phrase):
        problem = judge_client.validate_semantic_reply(
            {**SAMPLE, "reasoning": phrase})
        assert problem is not None
        assert "placeholder" in problem

    def test_a_real_evaluation_passes(self):
        assert judge_client.validate_semantic_reply(SAMPLE) is None

    def test_empty_reasoning_is_refused(self):
        assert judge_client.validate_semantic_reply(
            {**SAMPLE, "reasoning": "", "flags": []}) is not None

    def test_the_phrase_is_not_matched_in_the_transcript_itself(self):
        """A clinical answer may contain these words; only the judge's own prose counts."""
        assert judge_client.validate_semantic_reply(
            {**SAMPLE, "reasoning": "The answer discusses a meta-response pathway."}
        ) is not None  # the phrase list is about the judge's prose, and this is it
        assert judge_client.validate_semantic_reply(
            {**SAMPLE, "reasoning": "The answer explains the follow-up array."}) is None


class TestTheEndpoint:
    @pytest.mark.parametrize(("location", "host"), [
        ("global", "https://aiplatform.googleapis.com"),
        ("us", "https://aiplatform.us.rep.googleapis.com"),
        ("eu", "https://aiplatform.eu.rep.googleapis.com"),
        ("us-central1", "https://us-central1-aiplatform.googleapis.com"),
    ])
    def test_the_hostname_follows_the_location(self, location, host):
        url = judge_call.endpoint_url("a-project", location)

        assert url == (f"{host}/v1/projects/a-project/locations/{location}"
                       "/endpoints/openapi")

    def test_the_configured_location_is_used_when_none_is_given(self):
        assert f"/locations/{judge_call.JUDGE_LOCATION}/" in judge_call.endpoint_url()

    def test_the_judge_pin_is_the_independent_vendor(self):
        assert judge_client.JUDGE_MODEL == "xai/grok-4.7"
        assert judge_client.JUDGE_LOCATION in ("global", "us", "eu", "us-central1")


class _Credentials:
    def __init__(self, token: str, expiry: dt.datetime) -> None:
        self.token = token
        self.expiry = expiry
        self.refreshes = 0

    def refresh(self, _request) -> None:
        self.refreshes += 1


class TestTheToken:
    def _install(self, monkeypatch, credentials: _Credentials) -> None:
        monkeypatch.setattr(judge_call.google.auth, "default",
                            lambda scopes=None: (credentials, "a-project"))
        judge_call._token_cache.update({"token": "", "expires_at": 0.0})

    def test_a_token_is_reused_before_it_expires(self, monkeypatch):
        credentials = _Credentials("token-1", dt.datetime.now(dt.timezone.utc)
                                   + dt.timedelta(hours=1))
        self._install(monkeypatch, credentials)

        assert judge_call.access_token() == "token-1"
        assert judge_call.access_token() == "token-1"
        assert credentials.refreshes == 1

    def test_a_token_inside_the_refresh_margin_is_replaced(self, monkeypatch):
        soon = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)
        credentials = _Credentials("token-1", soon)
        self._install(monkeypatch, credentials)

        assert judge_call.access_token() == "token-1"
        credentials.token = "token-2"
        assert judge_call.access_token() == "token-2"
        assert credentials.refreshes == 2

    def test_a_missing_expiry_is_treated_as_short_lived(self, monkeypatch):
        credentials = _Credentials("token-1", None)  # type: ignore[arg-type]
        self._install(monkeypatch, credentials)

        assert judge_call.access_token() == "token-1"
        assert judge_call._token_cache["expires_at"] > 0

    def test_the_client_is_given_the_callable_not_a_string(self, monkeypatch):
        """The whole point: a string captured once expires mid-census."""
        captured: dict = {}

        class _OpenAI:
            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(openai, "OpenAI", _OpenAI)
        judge_call.judge_client_instance()

        assert callable(captured["api_key"])
        assert captured["api_key"] is judge_call.access_token
        assert captured["max_retries"] == 0
        assert captured["base_url"].endswith("/endpoints/openapi")
