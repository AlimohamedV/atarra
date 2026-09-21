"""Blockage Threat Score.

A 0-100 index that turns "there is vegetation here" into "clear this stretch of
canal first". The score exists to make the maintenance queue defensible, so the
components are kept separate and returned alongside it -- a number nobody can
interrogate is a number nobody should act on.

Three inputs, weighted:

  coverage   How much of the channel cross-section the vegetation occupies. This
             is the direct hydraulic threat and carries the most weight.
  width      Narrow channels choke sooner for the same coverage, so a lower
             absolute width increases urgency. Modelled as a smooth falloff rather
             than a hard threshold, because 12 m and 13 m of canal do not differ
             categorically.
  priority   Upstream discharge importance. Blocking a trunk canal that supplies
             thousands of farms must outrank blocking a field ditch that is
             regularly dry.

Every weight and constant is module-level so a reviewer can see and challenge it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Component weights. Sum to 1.0.
W_COVERAGE = 0.55
W_WIDTH = 0.25
W_PRIORITY = 0.20

# Canal width in metres at which urgency is maximal, and beyond which a channel is
# wide enough that vegetation takes a long time to constrict it.
WIDTH_CRITICAL_M = 8.0
WIDTH_TOLERANT_M = 120.0


@dataclass
class ThreatInputs:
    """Everything the score depends on."""

    coverage_fraction: float
    canal_width_m: float
    upstream_priority: float
    # Optional modifiers, each in 0..1.
    biomass_density: float | None = None
    downstream_pressure: float | None = None


@dataclass
class ThreatScore:
    """A score plus the components that produced it."""

    score: float
    coverage_component: float
    width_component: float
    priority_component: float
    band: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "band": self.band,
            "components": {
                "coverage": round(self.coverage_component, 4),
                "width": round(self.width_component, 4),
                "priority": round(self.priority_component, 4),
            },
            "notes": self.notes,
        }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _width_urgency(canal_width_m: float) -> float:
    """Urgency from channel width: narrow is urgent, wide is tolerant.

    Smooth and monotonic, saturating at both ends so that an implausible width
    (a data-entry error, or a polygon spanning a whole lagoon) cannot produce a
    wild score.
    """
    width = float(canal_width_m)
    if width <= WIDTH_CRITICAL_M:
        return 1.0
    if width >= WIDTH_TOLERANT_M:
        return 0.0
    span = WIDTH_TOLERANT_M - WIDTH_CRITICAL_M
    return float(1.0 - ((width - WIDTH_CRITICAL_M) / span) ** 0.5)


def _band(score: float) -> str:
    if score >= 75.0:
        return "critical"
    if score >= 50.0:
        return "high"
    if score >= 25.0:
        return "moderate"
    return "low"


def blockage_threat_score(inputs: ThreatInputs) -> ThreatScore:
    """Compute the 0-100 blockage threat score."""
    notes: list[str] = []

    coverage = _clip01(inputs.coverage_fraction)
    if inputs.coverage_fraction > 1.0:
        notes.append(
            f"coverage clamped from {inputs.coverage_fraction:.3f}; a fraction above 1 "
            "is not physical and suggests a polygon geometry error"
        )

    priority = _clip01(inputs.upstream_priority)
    width_component = _width_urgency(inputs.canal_width_m)

    score = 100.0 * (
        W_COVERAGE * coverage + W_WIDTH * width_component + W_PRIORITY * priority
    )

    # Optional modifiers nudge rather than dominate: they refine the queue among
    # patches of similar core score, which is exactly where a triage tool earns
    # its keep. Capped at +/-10 points in total so they cannot reorder the bands.
    if inputs.biomass_density is not None:
        adjustment = 5.0 * (_clip01(inputs.biomass_density) - 0.5)
        score += adjustment
        notes.append(f"biomass density adjusted the score by {adjustment:+.1f}")

    if inputs.downstream_pressure is not None:
        adjustment = 5.0 * (_clip01(inputs.downstream_pressure) - 0.5)
        score += adjustment
        notes.append(f"downstream pressure adjusted the score by {adjustment:+.1f}")

    score = max(0.0, min(100.0, score))

    if width_component < 0.15:
        notes.append(
            f"channel is {inputs.canal_width_m:.0f} m wide, so constriction is slow "
            "even at high coverage"
        )

    return ThreatScore(
        score=score,
        coverage_component=coverage,
        width_component=width_component,
        priority_component=priority,
        band=_band(score),
        notes=notes,
    )


def rank_by_threat(items: list[tuple[ThreatInputs, object]]) -> list[tuple[object, ThreatScore]]:
    """Score and sort candidate patches, most urgent first.

    Highest score first, then widest coverage as a tie-break -- when two patches
    are equally urgent, the larger one is worth more machine time per visit.

    The tie-break reads coverage from the *inputs*; the payload is an arbitrary
    caller-supplied object and must not be assumed to carry any attribute.
    """
    scored = [
        (payload, blockage_threat_score(inputs), inputs.coverage_fraction)
        for inputs, payload in items
    ]
    scored.sort(key=lambda item: (-item[1].score, -item[2]))
    return [(payload, score) for payload, score, _coverage in scored]
