"""The price table and the arithmetic, checked against the numbers and the traps.

Two traps, both measured live in Sprint 01b rather than reasoned about:

* a provider's reported `output` count already includes its reasoning tokens, and its
  `input` count already includes the cached read. So `thinking` and `cached` are
  sub-counts, and the only place they appear in the bill is the cached rate applied to
  the cached part — never as an addition to a total that already holds them;
* a model id the table does not know must raise. Returning `0.0` would report an unpriced
  arm as the cheapest arm in the comparison it exists to win.

The rates themselves are transcribed from the console for the six models on the roster,
so one test reads every row and asserts the shape of what a report will print.
"""

from __future__ import annotations

import pytest

from evaluation.platform import pricing


def test_every_roster_model_has_a_rate():
    for model_id in pricing.ROSTER:
        row = pricing.describe(model_id)
        assert row["model_id"] == model_id
        assert row["role"]
        for field in ("input_per_million", "cached_read_per_million",
                      "cache_write_per_million", "output_per_million"):
            assert isinstance(row[field], float), (model_id, field)
            assert row[field] >= 0.0, (model_id, field)


@pytest.mark.parametrize(("alias", "canonical"), [
    ("grok-4.7", "xai/grok-4.7"),
    ("xai/grok-4.7", "xai/grok-4.7"),
    ("meta/muse-spark-1.3", "muse-spark-1.3"),
    ("muse-spark-1.3", "muse-spark-1.3"),
    ("gemini-3.1-pro-preview", "gemini-3.1-pro"),
    ("gemini-3.1-pro", "gemini-3.1-pro"),
    ("  GEMINI-3.1-FLASH-LITE  ", "gemini-3.1-flash-lite"),
])
def test_a_served_alias_prices_as_its_canonical_id(alias, canonical):
    assert pricing.rate_for(alias) == pricing.rate_for(canonical)
    assert pricing.known(alias) is True


def test_an_unknown_model_raises_rather_than_costing_zero():
    with pytest.raises(pricing.UnknownModelError):
        pricing.rate_for("gemini-9-imaginary")
    with pytest.raises(pricing.UnknownModelError):
        pricing.cost_usd("gemini-9-imaginary",
                         {"input": 1000, "output": 1000, "cached": None,
                          "cache_write": None, "thinking": None})
    assert pricing.known("gemini-9-imaginary") is False


def test_the_published_rate_table_is_what_the_code_says():
    """The six rows a report prints, transcribed once and asserted once."""
    expected = {
        "gemini-3.1-flash-lite": (0.25, 0.025, 0.0, 1.50),
        "gemini-3.8-flash": (0.75, 0.075, 0.0, 3.75),
        "gemini-3.1-pro": (2.00, 0.20, 0.0, 12.00),
        "muse-spark-1.3": (1.25, 0.15, 0.0, 4.25),
        "claude-sonnet-5-5": (2.00, 0.20, 2.50, 10.00),
        "xai/grok-4.7": (2.00, 0.50, 0.0, 6.00),
    }
    for model_id, (price_in, cached, write, price_out) in expected.items():
        rate = pricing.rate_for(model_id)
        assert (rate.input, rate.cached_read, rate.cache_write, rate.output) == (
            price_in, cached, write, price_out), model_id


class TestNoDoubleCounting:
    def test_thinking_inside_output_is_not_added_again(self):
        """The Gemini measurement: input 32, output 160, total 192, thinking 153.

        `total == input + output`, so the 153 reasoning tokens are already inside the
        160. Billing output + thinking would charge them a second time at the output
        rate and make every reasoning model look twice as expensive as it is.
        """
        usage = {"input": 32, "output": 160, "thinking": 153, "total": 192,
                 "cached": None, "cache_write": None}
        rate = pricing.rate_for("gemini-3.1-flash-lite")

        cost = pricing.cost_usd("gemini-3.1-flash-lite", usage)

        assert pricing.billable_output(usage) == 160
        assert cost == pytest.approx((32 * rate.input + 160 * rate.output) / 1e6)

    def test_the_meta_measurement_prices_the_same_way(self):
        """input 59, output 374, total 433, thinking 361 on the Responses API."""
        rate = pricing.rate_for("meta/muse-spark-1.3")
        usage = {"input": 59, "output": 374, "thinking": 361, "total": 433,
                 "cached": None, "cache_write": None}
        cost = pricing.cost_usd("meta/muse-spark-1.3", usage)

        assert pricing.billable_output(usage) == 374
        assert cost == pytest.approx((59 * rate.input + 374 * rate.output) / 1e6)

    def test_reasoning_reported_outside_output_is_billed_once_and_added(self):
        """The grok measurement, live on 2026-11-13: output 390, thinking 4347.

        More reasoning tokens than completion tokens is proof they are not inside
        `output`, so they are billed as output. Dropping them would price the most
        expensive grading call in the run as if it had written one paragraph.
        """
        rate = pricing.rate_for("xai/grok-4.7")
        usage = {"input": 8622, "output": 390, "thinking": 4347,
                 "cached": 0, "cache_write": None}

        cost = pricing.cost_usd("xai/grok-4.7", usage)

        assert pricing.billable_output(usage) == 390 + 4347
        assert cost == pytest.approx((8622 * rate.input + 4737 * rate.output) / 1e6)

    def test_a_total_that_covers_the_reasoning_settles_it(self):
        """When `total` accounts for the thinking, the thinking is inside output."""
        inside = {"input": 100, "output": 500, "thinking": 300, "total": 600}
        outside = {"input": 100, "output": 500, "thinking": 300, "total": 900}

        assert pricing.billable_output(inside) == 500
        assert pricing.billable_output(outside) == 800

    def test_reasoning_outside_output_is_added_even_when_shorter_than_the_reply(self):
        """`total > input + output` is the decisive signal, not the size of `thinking`.

        A verbose grading critique can write more completion than reasoning — 900 against
        600 — while still being billed for the reasoning separately, and the reported
        total is what proves it: 8000 + 900 + 600 = 9500. Gating on `thinking > output`
        alone would drop those 600 tokens and price a long Judge call short, which is the
        case the grok measurement does not exercise because there the reasoning was
        longer than the reply.
        """
        usage = {"input": 8000, "output": 900, "thinking": 600, "total": 9500,
                 "cached": None, "cache_write": None}

        assert pricing.billable_output(usage) == 1500

    def test_with_no_total_the_conservative_reading_is_used(self):
        """No total to check against: assume inside output unless it cannot be."""
        assert pricing.billable_output({"output": 500, "thinking": 300}) == 500
        assert pricing.billable_output({"output": 100, "thinking": 300}) == 400

    def test_a_cached_read_is_subtracted_from_input_and_priced_at_its_own_rate(self):
        rate = pricing.rate_for("xai/grok-4.7")
        usage = {"input": 1000, "cached": 400, "output": 100, "cache_write": None,
                 "thinking": None}

        cost = pricing.cost_usd("xai/grok-4.7", usage)

        assert cost == pytest.approx(
            (600 * rate.input + 400 * rate.cached_read + 100 * rate.output) / 1e6)
        # And the naive "add everything" reading is strictly more expensive, which is
        # what makes this a real assertion rather than an identity.
        assert cost < (1400 * rate.input + 100 * rate.output) / 1e6

    def test_billable_input_is_clamped_at_zero(self):
        """A provider reporting more cached than input cannot produce a negative bill."""
        assert pricing.billable_input({"input": 10, "cached": 25}) == 0
        assert pricing.billable_input({"input": None, "cached": 25}) == 0
        assert pricing.billable_input({}) == 0

    def test_a_cache_write_is_priced_only_where_the_provider_charges_for_it(self):
        usage = {"input": 1000, "cached": 0, "cache_write": 1000, "output": 0,
                 "thinking": None}
        assert pricing.cost_usd("claude-sonnet-5-5", usage) == pytest.approx(
            (1000 * 2.00 + 1000 * 2.50) / 1e6)
        assert pricing.cost_usd("gemini-3.1-flash-lite", usage) == pytest.approx(
            (1000 * 0.25) / 1e6)


class TestUsageNormalisation:
    def test_langfuse_usage_details_spellings(self):
        assert pricing.normalise_usage(
            {"input": 32, "output": 160, "total": 192}) == {
            "input": 32, "cached": None, "output": 160, "cache_write": None,
            "thinking": None, "total": 192}

    def test_openai_spellings(self):
        block = {"prompt_tokens": 900, "completion_tokens": 120,
                 "total_tokens": 1020,
                 "prompt_tokens_details": {"cached_tokens": 300},
                 "completion_tokens_details": {"reasoning_tokens": 40}}

        assert pricing.normalise_usage(block) == {
            "input": 900, "cached": 300, "output": 120, "cache_write": None,
            "thinking": 40, "total": 1020}

    def test_an_unreported_field_stays_none_rather_than_becoming_zero(self):
        assert pricing.normalise_usage({})["thinking"] is None
        assert pricing.normalise_usage(None)["input"] is None

    def test_merging_sums_what_was_reported_and_keeps_none_apart(self):
        merged = pricing.merge_usage([
            {"input": 10, "output": 5, "cached": None, "thinking": 2},
            {"input": 20, "output": None, "cached": 4, "thinking": None},
        ])

        assert merged["input"] == 30
        assert merged["output"] == 5
        assert merged["cached"] == 4
        assert merged["thinking"] == 2
