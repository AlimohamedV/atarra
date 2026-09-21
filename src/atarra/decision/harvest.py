"""Harvest-window optimisation.

The remediation logic this project is built around: *Phragmites* takes up heavy
metals and agricultural nutrients through the growing season, holds them in living
tissue, and then releases them back into the water as it senesces in autumn. So the
harvest has to happen when accumulated content is at its maximum but before
translocation back to the rhizome and litterfall begin. Cutting too early removes
nutrients but wastes the season's uptake; cutting too late puts them straight back
into the canal.

Two things are modelled deliberately separately:

  **Peak standing biomass** -- from the fitted growth curve.
  **Senescence risk** -- a physical process that starts once growth stops, not
  something the curve knows about, so it gets its own curve.

The window is placed to end at peak biomass and start ``lead_days`` earlier,
because crews need time to mobilise. That lead time is the one genuinely
operational input here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import numpy as np

from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger
from atarra.forecast.growth import AreaObservation, fit_logistic

log = get_logger("decision.harvest")

# Accumulated nutrient content as a fraction of the season's maximum, by month.
# August is the peak, and the decline after it is the senescence release the
# operation exists to prevent. Derived from the published phenology of
# Phragmites australis in Mediterranean coastal wetlands: nutrient standing stock
# peaks in late summer, then declines through autumn as it is translocated to
# rhizomes and lost with litterfall.
MONTHLY_NUTRIENT_CONTENT = {
    1: 0.05, 2: 0.05, 3: 0.08, 4: 0.15,
    5: 0.30, 6: 0.50, 7: 0.75, 8: 1.00,
    9: 0.72, 10: 0.45, 11: 0.20, 12: 0.08,
}

# Fraction of the season's accumulated nutrient content returned to the water when
# senescence proceeds unharvested. This is the loss the harvest window avoids.
SENESCENCE_RELEASE_FRACTION = 0.60


@dataclass(frozen=True)
class BiomassObservation:
    """A dated biomass proxy observation."""

    observed_on: Date
    value: float
    kind: str = "ndvi_integral"


@dataclass
class HarvestWindow:
    """A recommended cutting window and the reasoning behind it."""

    area_key: str
    season_year: int
    window_start: Date
    window_end: Date
    peak_date: Date
    lead_days: int
    expected_nutrient_fraction: float
    senescence_loss_avoided: float
    rationale: str
    curve: dict | None = None
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    @property
    def days(self) -> int:
        return (self.window_end - self.window_start).days + 1

    def as_dict(self) -> dict:
        return {
            "area": self.area_key,
            "season_year": self.season_year,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "peak_date": self.peak_date.isoformat(),
            "days": self.days,
            "lead_days": self.lead_days,
            "expected_nutrient_fraction": round(self.expected_nutrient_fraction, 4),
            "senescence_loss_avoided": round(self.senescence_loss_avoided, 4),
            "rationale": self.rationale,
            "curve": self.curve,
            "warnings": self.warnings,
        }


def nutrient_content_fraction(day: Date) -> float:
    """Interpolated accumulated nutrient content for a calendar date, 0..1."""
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_values = np.array([MONTHLY_NUTRIENT_CONTENT[m] for m in range(1, 13)], dtype=np.float64)

    # Place each month's value at its midpoint so a date interpolates sensibly
    # rather than jumping at month boundaries.
    offsets, cumulative = [], 0
    for length in days_in_month:
        offsets.append(cumulative + length / 2.0)
        cumulative += length
    day_of_year = day.timetuple().tm_yday
    return float(np.interp(day_of_year, offsets, month_values, period=365))


def senescence_risk_curve(start: Date, days: int = 400) -> list[dict]:
    """Nutrient content against date, for plotting or export."""
    return [
        {
            "date": (start + timedelta(days=offset)).isoformat(),
            "nutrient_fraction": round(nutrient_content_fraction(start + timedelta(days=offset)), 4),
        }
        for offset in range(days)
    ]


def harvest_window(
    area_key: str,
    observations: list[BiomassObservation],
    *,
    season_year: int | None = None,
    lead_days: int = 14,
    peak_fraction: float = 0.95,
) -> HarvestWindow:
    """Recommend a cutting window for one season.

    The window closes on the projected peak standing biomass and opens
    ``lead_days`` before it, so crews arrive while the plants still hold their
    accumulated metals and nutrients.
    """
    if not observations:
        raise AtarraError("harvest_window needs at least one biomass observation")
    if lead_days < 0:
        raise AtarraError(f"lead_days must be >= 0, got {lead_days}")

    ordered = sorted(observations, key=lambda o: o.observed_on)
    year = season_year or ordered[-1].observed_on.year
    warnings: list[str] = []

    # Cross-check the fitted peak against the biological expectation. A curve fit
    # to a partial season can place the peak in the wrong month entirely, and a
    # harvest scheduled for December would be worse than no recommendation.
    curve_dict: dict | None = None
    peak_date: Date | None = None

    if len(ordered) >= 4:
        try:
            fit = fit_logistic(
                [AreaObservation(o.observed_on, o.value, o.kind) for o in ordered],
                peak_fraction=peak_fraction,
            )
            curve_dict = fit.as_dict()
            origin = ordered[0].observed_on
            peak_day = fit.day_at_fraction(peak_fraction)
            candidate = origin + timedelta(days=int(round(peak_day)))
            if candidate.year == year or abs((candidate - origin).days) < 400:
                peak_date = candidate
            if fit.r_squared < 0.5:
                warnings.append(
                    f"growth curve fit is weak (R^2={fit.r_squared:.2f}); the peak date "
                    "is driven mainly by the biological prior below"
                )
        except AtarraError as exc:
            warnings.append(f"could not fit a growth curve ({exc}); using the phenological prior")

    if peak_date is None:
        latest = ordered[-1]
        peak_date = _prior_peak_date(year)
        warnings.append(
            "no growth curve available, so the peak is placed from the published "
            f"phenology (August {year}) rather than from this season's observations "
            f"(latest observation {latest.observed_on})"
        )

    # The biological prior is authoritative about which month is plausible; the
    # curve is authoritative about this season's timing within it. Blending them
    # keeps a bad fit from scheduling a midwinter harvest.
    prior_peak = _prior_peak_date(year)
    if abs((peak_date - prior_peak).days) > 60:
        warnings.append(
            f"fitted peak {peak_date} is more than 60 days from the published "
            f"August phenology ({prior_peak}); using {prior_peak}. Check the season's "
            "observations before trusting this window."
        )
        peak_date = prior_peak

    start = peak_date - timedelta(days=lead_days)
    nutrient_fraction = nutrient_content_fraction(peak_date)

    if start.year != peak_date.year:
        warnings.append(
            "the lead time pushes the window start into the previous calendar year"
        )

    avoided = nutrient_fraction * SENESCENCE_RELEASE_FRACTION
    rationale = (
        f"Standing nutrient content peaks around {peak_date} "
        f"({nutrient_fraction * 100:.0f}% of the season's maximum). Harvesting in the "
        f"{lead_days}-day lead-up captures that peak while avoiding senescence, which "
        f"would otherwise return roughly {SENESCENCE_RELEASE_FRACTION * 100:.0f}% of the "
        f"accumulated load to the water column. Cutting earlier sacrifices uptake; "
        f"cutting later re-releases it."
    )

    return HarvestWindow(
        area_key=area_key,
        season_year=year,
        window_start=start,
        window_end=peak_date,
        peak_date=peak_date,
        lead_days=lead_days,
        expected_nutrient_fraction=nutrient_fraction,
        senescence_loss_avoided=avoided,
        rationale=rationale,
        curve=curve_dict,
        warnings=warnings,
    )


def _prior_peak_date(year: int) -> Date:
    """Published peak-nutrient date: mid-August."""
    return Date(year, 8, 15)
