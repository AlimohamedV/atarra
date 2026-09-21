"""Tests for the decision-support and forecasting layers."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from atarra.core.errors import AtarraError
from atarra.decision.harvest import (
    BiomassObservation,
    harvest_window,
    nutrient_content_fraction,
    senescence_risk_curve,
)
from atarra.decision.threat import ThreatInputs, blockage_threat_score, rank_by_threat
from atarra.forecast.growth import (
    AreaObservation,
    expansion_rate,
    fit_logistic,
    project_coverage,
    time_to_blockage,
)


class TestThreatScore:
    def test_score_is_bounded(self):
        for coverage in np.linspace(0, 1, 11):
            for width in (5, 20, 60, 200, 500):
                result = blockage_threat_score(ThreatInputs(coverage, width, 0.5))
                assert 0.0 <= result.score <= 100.0

    def test_full_coverage_narrow_canal_is_critical(self):
        result = blockage_threat_score(ThreatInputs(1.0, 5.0, 1.0))
        assert result.score > 90
        assert result.band == "critical"

    def test_clear_wide_canal_is_low(self):
        result = blockage_threat_score(ThreatInputs(0.0, 200.0, 0.0))
        assert result.score < 10
        assert result.band == "low"

    def test_more_coverage_scores_higher(self):
        low = blockage_threat_score(ThreatInputs(0.2, 20.0, 0.5))
        high = blockage_threat_score(ThreatInputs(0.8, 20.0, 0.5))
        assert high.score > low.score

    def test_narrower_canal_scores_higher_at_equal_coverage(self):
        narrow = blockage_threat_score(ThreatInputs(0.6, 10.0, 0.5))
        wide = blockage_threat_score(ThreatInputs(0.6, 100.0, 0.5))
        assert narrow.score > wide.score

    def test_higher_priority_scores_higher(self):
        low = blockage_threat_score(ThreatInputs(0.6, 20.0, 0.1))
        high = blockage_threat_score(ThreatInputs(0.6, 20.0, 0.9))
        assert high.score > low.score

    def test_components_are_returned_for_audit(self):
        result = blockage_threat_score(ThreatInputs(0.5, 40.0, 0.5))
        payload = result.as_dict()
        assert set(payload["components"]) == {"coverage", "width", "priority"}

    def test_out_of_range_coverage_is_clamped_and_noted(self):
        """A fraction above 1 is a geometry bug; surface it rather than score it."""
        result = blockage_threat_score(ThreatInputs(1.5, 20.0, 0.5))
        assert result.coverage_component == 1.0
        assert any("clamped" in note for note in result.notes)

    def test_modifiers_are_bounded(self):
        """Modifiers must not be able to reorder the scoring bands."""
        base = blockage_threat_score(ThreatInputs(0.5, 20.0, 0.5))
        extreme = blockage_threat_score(
            ThreatInputs(0.5, 20.0, 0.5, biomass_density=1.0, downstream_pressure=1.0)
        )
        assert abs(extreme.score - base.score) <= 10.0

    def test_band_thresholds(self):
        assert blockage_threat_score(ThreatInputs(0.0, 500.0, 0.0)).band == "low"
        assert blockage_threat_score(ThreatInputs(1.0, 1.0, 1.0)).band == "critical"

    def test_ranking_puts_most_urgent_first(self):
        items = [
            (ThreatInputs(0.2, 100.0, 0.1), "minor"),
            (ThreatInputs(0.95, 8.0, 0.9), "urgent"),
            (ThreatInputs(0.5, 40.0, 0.5), "moderate"),
        ]
        ranked = rank_by_threat(items)
        assert ranked[0][0] == "urgent"
        assert ranked[-1][0] == "minor"


class TestGrowth:
    def test_logistic_recovers_known_parameters(self):
        """A synthetic logistic must be recoverable, or the fit is not trustworthy."""
        rng = np.random.default_rng(0)
        truth = {"K": 0.85, "r": 0.05, "t0": 90.0}
        origin = date(2023, 4, 1)
        observations = []
        for day in range(0, 200, 10):
            value = truth["K"] / (1 + np.exp(-truth["r"] * (day - truth["t0"])))
            observations.append(
                AreaObservation(origin + timedelta(days=day), float(value + rng.normal(0, 0.01)))
            )

        fit = fit_logistic(observations)
        assert fit.converged
        assert fit.r_squared > 0.95
        assert fit.K == pytest.approx(truth["K"], rel=0.15)
        assert fit.r == pytest.approx(truth["r"], rel=0.25)
        assert fit.t0 == pytest.approx(truth["t0"], abs=20)

    def test_too_few_observations_raises(self):
        origin = date(2023, 6, 1)
        with pytest.raises(AtarraError, match="at least"):
            fit_logistic([AreaObservation(origin, 0.1), AreaObservation(origin, 0.2)])

    def test_doubling_time_positive(self):
        origin = date(2023, 4, 1)
        observations = [
            AreaObservation(origin + timedelta(days=day), 0.1 + 0.003 * day) for day in range(0, 100, 10)
        ]
        fit = fit_logistic(observations)
        assert fit.doubling_time_days() > 0

    def test_expansion_rate_doubling(self):
        origin = date(2023, 4, 1)
        observations = [
            AreaObservation(origin, 10.0),
            AreaObservation(origin + timedelta(days=10), 20.0),
        ]
        estimate = expansion_rate(observations)
        assert estimate.doubling_time_days == pytest.approx(10.0, rel=1e-3)
        assert estimate.span_days == 10

    def test_expansion_rate_needs_two_points(self):
        with pytest.raises(AtarraError, match="at least two"):
            expansion_rate([AreaObservation(date(2023, 1, 1), 1.0)])

    def test_expansion_rate_handles_non_positive_values(self):
        """Logarithms are undefined at zero; the fallback must not raise."""
        origin = date(2023, 4, 1)
        estimate = expansion_rate(
            [AreaObservation(origin, 0.0), AreaObservation(origin + timedelta(days=10), 5.0)]
        )
        assert estimate.daily_rate == pytest.approx(0.5)

    def test_time_to_blockage_hand_computed(self):
        # From 0.25 to 0.75 at 0.01/day: ln(3)/0.01 = 109.86 days
        result = time_to_blockage(0.25, 0.01, threshold=0.75)
        assert result["days"] == pytest.approx(np.log(3) / 0.01, rel=1e-3)

    def test_already_blocked(self):
        result = time_to_blockage(0.9, 0.01, threshold=0.75)
        assert result["days"] == 0
        assert result["already_blocked"] is True

    def test_not_growing_returns_no_date(self):
        result = time_to_blockage(0.3, 0.0)
        assert result["days"] is None
        assert "never" in result["reason"]

    def test_projection_saturates(self):
        assert project_coverage(0.5, 0.5, 100) == 1.0


class TestHarvestWindow:
    def test_nutrient_curve_peaks_in_august(self):
        assert nutrient_content_fraction(date(2023, 8, 15)) > nutrient_content_fraction(date(2023, 5, 15))
        assert nutrient_content_fraction(date(2023, 8, 15)) > nutrient_content_fraction(date(2023, 11, 15))

    def test_nutrient_curve_is_bounded(self):
        for month in range(1, 13):
            value = nutrient_content_fraction(date(2023, month, 15))
            assert 0.0 <= value <= 1.0

    def test_window_ends_on_the_peak_and_starts_earlier(self):
        observations = [
            BiomassObservation(date(2023, month, 15), value)
            for month, value in zip(range(4, 10), [0.15, 0.35, 0.60, 0.80, 0.85, 0.70])
        ]
        window = harvest_window("burullus", observations, lead_days=14)
        assert window.window_end == window.peak_date
        assert (window.window_start - window.window_end).days == -14
        assert window.days == 15

    def test_peak_stays_in_august_despite_a_bad_fit(self):
        """A midwinter harvest would be worse than no recommendation at all."""
        observations = [
            BiomassObservation(date(2023, 1, 5), 0.10),
            BiomassObservation(date(2023, 1, 20), 0.70),
            BiomassObservation(date(2023, 2, 5), 0.20),
            BiomassObservation(date(2023, 2, 20), 0.60),
        ]
        window = harvest_window("burullus", observations, season_year=2023)
        assert window.peak_date.month in {8}
        assert window.warnings

    def test_noise_falls_back_to_prior_with_a_warning(self):
        observations = [
            BiomassObservation(date(2023, 5, 1), 0.5),
            BiomassObservation(date(2023, 6, 1), 0.5),
            BiomassObservation(date(2023, 7, 1), 0.5),
        ]
        window = harvest_window("burullus", observations, season_year=2023)
        assert window.peak_date == date(2023, 8, 15)
        assert window.warnings

    def test_rationale_mentions_senescence(self):
        observations = [BiomassObservation(date(2023, 8, 1), 0.8)]
        window = harvest_window("burullus", observations, season_year=2023)
        assert "senescence" in window.rationale.lower()

    def test_empty_observations_raises(self):
        with pytest.raises(AtarraError, match="at least one"):
            harvest_window("burullus", [])

    def test_negative_lead_days_raises(self):
        with pytest.raises(AtarraError, match="lead_days"):
            harvest_window("burullus", [BiomassObservation(date(2023, 8, 1), 0.8)], lead_days=-1)

    def test_serialises_to_dict(self):
        window = harvest_window(
            "burullus", [BiomassObservation(date(2023, 8, 1), 0.8)], season_year=2023
        )
        payload = window.as_dict()
        assert payload["season_year"] == 2023
        assert "window_start" in payload and "rationale" in payload

    def test_risk_curve_covers_a_year(self):
        curve = senescence_risk_curve(date(2023, 1, 1), days=365)
        assert len(curve) == 365
        assert max(point["nutrient_fraction"] for point in curve) > 0.9

    def test_loss_avoided_is_a_fraction(self):
        window = harvest_window(
            "burullus", [BiomassObservation(date(2023, 8, 1), 0.8)], season_year=2023
        )
        assert 0.0 < window.senescence_loss_avoided <= 1.0
