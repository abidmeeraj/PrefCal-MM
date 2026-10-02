"""Unit tests for the model-free parts of PrefCal-MM. Run with ``pytest tests/``."""
import math

import numpy as np
import pandas as pd
import pytest

from prefcal_mm.aggregation import bradley_terry as bt
from prefcal_mm.aggregation import composite, preferences
from prefcal_mm.normalisation import normalise
from prefcal_mm.parsing import extract_first_int, parse_batched_verdicts
from prefcal_mm.pillars.factual import _parse_facts, unified_score


# ── parsing ────────────────────────────────────────────────────────────────

def test_extract_first_int():
    assert extract_first_int("- Coherence: 4") == 4
    assert extract_first_int("7 then 3", allowed=range(1, 6)) == 3
    assert extract_first_int("no score") is None


def test_batched_verdicts_strict():
    raw = "1. supported\n2. not_supported\n3. not supported"
    assert parse_batched_verdicts(raw, 3) == ["supported", "not_supported", "not_supported"]


def test_batched_verdicts_missing_line_is_unk():
    assert parse_batched_verdicts("1. supported", 2) == ["supported", "unk"]


def test_batched_verdicts_positional_fallback():
    # Offset numbering (5., 6.) for a 2-claim batch is recovered by position.
    assert parse_batched_verdicts("5. not_supported\n6. supported", 2) == ["not_supported", "supported"]


def test_batched_verdicts_strips_chat_artifacts():
    raw = "1. not_supported [/INST] 1. supported"
    assert parse_batched_verdicts(raw, 1, strip_artifacts=True) == ["supported"]


def test_fact_parser():
    raw = "- John was arrested.\n2) It happened in Paris.\nshort\n- John was arrested."
    assert _parse_facts(raw) == ["John was arrested.", "It happened in Paris."]


# ── unified factual score ──────────────────────────────────────────────────

def test_unified_score_or_and_unk():
    facts = [
        {"fact": "a", "text_verdict": "supported", "visual_verdict": "skipped_text_supported"},
        {"fact": "b", "text_verdict": "not_supported", "visual_verdict": "supported"},
        {"fact": "c", "text_verdict": "not_supported", "visual_verdict": "not_supported"},
        {"fact": "d", "text_verdict": "unk", "visual_verdict": "not_supported"},
    ]
    score, _, stats = unified_score(facts)
    assert score == round(2 / 3, 6)          # 'd' is excluded from the denominator
    assert stats["n_supported_text_only"] == 1
    assert stats["n_supported_image_only"] == 1
    assert stats["n_supported_neither"] == 1
    assert stats["n_unk"] == 1


def test_unified_score_empty_is_one():
    assert unified_score([])[0] == 1.0


# ── normalisation ──────────────────────────────────────────────────────────

def test_normalise_bounds():
    assert normalise(1, "image_text_relevance") == 0.0
    assert normalise(5, "image_text_relevance") == 1.0
    with pytest.raises(ValueError):
        normalise(6, "image_text_relevance")


# ── preferences and Bradley-Terry ──────────────────────────────────────────

def _toy_corpus(n_articles=40, n_systems=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_articles):
        for s in range(n_systems):
            f = rng.uniform(0, 1, size=4)
            quality = 2.0 * f[0] + 1.0 * f[1] + 0.5 * f[2] + rng.normal(0, 0.1)
            rows.append({"article_id": f"a{a}", "system_name": f"s{s}",
                         "sfact_unified": f[0], "srel": f[1], "scoh": f[2], "sflu": f[3],
                         "overall_quality": quality})
    return pd.DataFrame(rows)


def test_derive_pairs_discards_ties():
    df = pd.DataFrame({"article_id": ["a"] * 3, "system_name": ["x", "y", "z"],
                       "overall_quality": [3.0, 3.0, 4.0]})
    pairs = preferences.derive_pairs(df)
    assert len(pairs) == 2
    assert all(p["preferred"] in ("i", "j") for p in pairs)


def test_bt_recovers_informative_features():
    df = _toy_corpus()
    pairs = preferences.derive_pairs(df)
    index = preferences.build_sample_index(df)
    w = bt.train_stage1(df, pairs, index)
    assert math.isclose(sum(w[k] for k in bt.STAGE1_KEYS), 1.0, rel_tol=1e-9)
    assert w["w_fact"] > w["w_rel"] > w["w_coh"]
    assert w["cv_accuracy"] > 0.7


def test_normalise_nonnegative_clips():
    out = bt.normalise_nonnegative(np.array([2.0, -1.0, 2.0]))
    assert out.tolist() == [0.5, 0.0, 0.5]


# ── composite ──────────────────────────────────────────────────────────────

def test_released_weights_and_composite():
    s1, s2 = composite.load_weights()
    assert math.isclose(sum(s1[k] for k in bt.STAGE1_KEYS), 1.0, rel_tol=1e-6)
    assert math.isclose(sum(s2[k] for k in bt.STAGE2_KEYS), 1.0, rel_tol=1e-6)
    stext = composite.stext_unified(1.0, 1.0, 1.0, 1.0, s1)
    assert math.isclose(stext, 1.0, abs_tol=1e-5)
    assert math.isnan(composite.prefcal_mm(stext, 1.0, float("nan"), s2))


# ── diversity formulas (need torch) ────────────────────────────────────────

def test_dpp_orthogonal_vs_duplicate():
    torch = pytest.importorskip("torch")
    from prefcal_mm.pillars.diversity import dpp_logdet

    orthogonal = torch.eye(3, 8)
    duplicate = torch.ones(3, 8)
    assert dpp_logdet(orthogonal)[0] > 0.99
    assert dpp_logdet(duplicate)[0] < 0.2
