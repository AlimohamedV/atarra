"""Growth modelling for reed patches.

The honest framing, stated up front because it shapes everything here: **canopy
biomass cannot be measured from orbit.** What a satellite sees is a spectral
signal strongly correlated with leaf area and chlorophyll, not kilograms per
square metre. So growth is fitted to an NDVI/NDRE-derived proxy and reported as a
proxy. The fitted parameters are informative (when growth accelerates, when it
saturates, doubling time) but the absolute level is not a mass.

Two models are provided deliberately. A logistic curve is interpretable and needs
few observations; a plain exponential rate estimate needs no fitting at all and
degrades gracefully when a season has gaps. Where they disagree is itself
informative, so :meth:`LogisticFit.r_squared` and the observation count travel with
every result rather than being hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import numpy as np

from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger

log = get_logger("forecast.growth")

MIN_OBSERVATIONS = 4


@dataclass(frozen=True)
class AreaObservation:
    """One dated observation of a patch."""

    observed_on: Date
    value: float
    kind: str = "ndvi_integral"

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise AtarraError(f"non-finite observation on {self.observed_on}")


@dataclass
class LogisticFit:
    """A fitted logistic curve ``y = K / (1 + exp(-r (t - t0)))``."""

    K: float
    r: float
    t0: float
    r_squared: float
    n_observations: int
    converged: bool
    proxy_kind: str = "ndvi_integral"
    note: str = ""

    def predict(self, day: float) -> float:
        return float(self.K / (1.0 + np.exp(-self.r * (day - self.t0))))

    @property
    def max_growth_day(self) -> float:
        """Day index of peak growth rate (the curve's inflection point)."""
        return float(self.t0)

    def day_at_fraction(self, fraction: float = 0.95) -> float:
        """Day index at which the curve first reaches ``fraction`` of carrying capacity.

        Standing biomass peaks well before the asymptote is actually reached, so
        "when is it essentially fully grown" is the question that matters for
        scheduling, not "when does it reach K" (which is never, mathematically).
        """
        fraction = min(max(fraction, 0.01), 0.999)
        return float(self.t0 + np.log(fraction / (1.0 - fraction)) / self.r)

    def doubling_time_days(self) -> float:
        """Doubling time during exponential phase: ``ln(2) / r``."""
        return float(np.log(2.0) / self.r) if self.r > 0 else float("inf")

    def as_dict(self) -> dict:
        return {
            "carrying_capacity": round(self.K, 4),
            "growth_rate_per_day": round(self.r, 5),
            "inflection_day": round(self.t0, 2),
            "r_squared": round(self.r_squared, 4),
            "doubling_time_days": round(self.doubling_time_days(), 2),
            "n_observations": self.n_observations,
            "converged": self.converged,
            "proxy_kind": self.proxy_kind,
            "note": self.note,
        }


def fit_logistic(
    observations: list[AreaObservation],
    *,
    peak_fraction: float = 0.95,
) -> LogisticFit:
    """Fit a logistic curve to a season of proxy observations."""
    if len(observations) < MIN_OBSERVATIONS:
        raise AtarraError(
            f"logistic fit needs at least {MIN_OBSERVATIONS} observations, "
            f"got {len(observations)}. Use `expansion_rate` instead, which needs two."
        )

    ordered = sorted(observations, key=lambda o: o.observed_on)
    origin = ordered[0].observed_on
    days = np.array([(o.observed_on - origin).days for o in ordered], dtype=np.float64)
    values = np.array([o.value for o in ordered], dtype=np.float64)

    if np.ptp(days) == 0:
        raise AtarraError("all observations fall on one date; cannot fit a trend")

    kinds = {o.kind for o in ordered}
    kind = kinds.pop() if len(kinds) == 1 else "mixed"

    try:
        from scipy.optimize import curve_fit

        def logistic(t, K, r, t0):
            return K / (1.0 + np.exp(-r * (t - t0)))

        # Bounds keep the fit physical: a positive capacity, a positive growth
        # rate, and an inflection somewhere in the observed window. Without them
        # curve_fit happily returns r < 0, i.e. a patch that is shrinking while
        # the data shows it growing.
        span = float(np.ptp(days))
        initial = (float(values.max() * 1.2) or 1.0, 1.0 / max(span / 4.0, 1.0), float(np.median(days)))
        bounds = ([0.0, 1e-6, float(days.min()) - span], [np.inf, 1.0, float(days.max()) + span])
        params, _ = curve_fit(logistic, days, values, p0=initial, bounds=bounds, maxfev=20000)
        K, r, t0 = (float(v) for v in params)

        predicted = logistic(days, K, r, t0)
        residual = float(np.sum((values - predicted) ** 2))
        total = float(np.sum((values - values.mean()) ** 2))
        r_squared = 1.0 - residual / total if total > 0 else 0.0

        if r_squared < 0.5:
            log.warning(
                "logistic fit is weak (R^2=%.3f over %d observations); treat the "
                "projected dates as indicative",
                r_squared,
                len(ordered),
            )
        return LogisticFit(
            K=K,
            r=r,
            t0=t0,
            r_squared=r_squared,
            n_observations=len(ordered),
            converged=True,
            proxy_kind=kind,
            note=(
                "Fitted to a spectral proxy, not to measured biomass."
                if r_squared >= 0.5
                else "Weak fit (R^2 < 0.5): the projected dates carry little confidence."
            ),
        )
    except Exception as exc:  # scipy missing, or the fit did not converge
        log.warning("logistic fit failed (%s); falling back to the exponential estimate", exc)
        rate = expansion_rate(ordered)
        return LogisticFit(
            K=float(values.max()),
            r=rate.daily_rate,
            t0=float(np.median(days)),
            r_squared=float("nan"),
            n_observations=len(ordered),
            converged=False,
            proxy_kind=kind,
            note=f"Logistic fit did not converge ({exc}); rate taken from the exponential estimate.",
        )


@dataclass
class ExpansionEstimate:
    """Exponential expansion statistics."""

    daily_rate: float
    doubling_time_days: float
    span_days: int
    first_value: float
    last_value: float

    @property
    def relative_gain(self) -> float:
        return self.last_value / self.first_value if self.first_value else float("inf")

    def as_dict(self) -> dict:
        return {
            "daily_rate": round(self.daily_rate, 5),
            "doubling_time_days": round(self.doubling_time_days, 2),
            "span_days": self.span_days,
            "relative_gain": round(self.relative_gain, 3),
        }


def expansion_rate(observations: list[AreaObservation]) -> ExpansionEstimate:
    """Estimate exponential expansion from the first and last observation.

    Two points make the estimate fragile to a single bad observation, which is
    why the full span is reported alongside it: a rate derived from a 9-day span
    deserves far less trust than one from 240 days, and the caller can see which
    they have.
    """
    if len(observations) < 2:
        raise AtarraError("expansion_rate needs at least two observations")

    ordered = sorted(observations, key=lambda o: o.observed_on)
    first, last = ordered[0], ordered[-1]
    span = (last.observed_on - first.observed_on).days
    if span <= 0:
        raise AtarraError("observations must span more than one day")

    if first.value <= 0 or last.value <= 0:
        # A non-positive proxy makes logarithms undefined; fall back to an absolute
        # rate so the caller still gets something usable.
        rate = (last.value - first.value) / span
        return ExpansionEstimate(rate, float("inf"), span, first.value, last.value)

    rate = float(np.log(last.value / first.value) / span)
    doubling = float(np.log(2.0) / rate) if rate > 0 else float("inf")
    return ExpansionEstimate(
        daily_rate=rate,
        doubling_time_days=doubling,
        span_days=span,
        first_value=first.value,
        last_value=last.value,
    )


def time_to_blockage(
    current_coverage: float,
    daily_rate: float,
    *,
    threshold: float = 0.75,
) -> dict:
    """Days until coverage crosses the functional-capacity threshold.

    Reported as a dict rather than a bare number because the interesting cases are
    the ones where no answer exists: already blocked, or not growing.
    """
    if not 0.0 < current_coverage <= 1.0:
        raise AtarraError(f"coverage must be in (0, 1], got {current_coverage}")
    if not 0.0 < threshold <= 1.0:
        raise AtarraError(f"threshold must be in (0, 1], got {threshold}")

    if current_coverage >= threshold:
        return {
            "days": 0,
            "already_blocked": True,
            "daily_rate": daily_rate,
            "current_coverage": current_coverage,
            "threshold": threshold,
        }

    if daily_rate <= 0:
        return {
            "days": None,
            "already_blocked": False,
            "daily_rate": daily_rate,
            "current_coverage": current_coverage,
            "threshold": threshold,
            "reason": "coverage is not growing, so the threshold is never reached",
        }

    days = float(np.log(threshold / current_coverage) / daily_rate)
    return {
        "days": round(days, 1),
        "already_blocked": False,
        "daily_rate": daily_rate,
        "current_coverage": current_coverage,
        "threshold": threshold,
        "projected_date": None,
    }


def project_coverage(current_coverage: float, daily_rate: float, days: float) -> float:
    """Coverage after ``days``, saturating at 1.0."""
    projected = current_coverage * float(np.exp(daily_rate * days))
    return float(min(1.0, projected))
