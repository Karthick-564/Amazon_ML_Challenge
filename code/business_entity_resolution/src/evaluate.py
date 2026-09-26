"""Challenge-aligned macro F_0.5 evaluation at the Source-1 entity level."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .io_utils import read_id_lists


@dataclass(frozen=True)
class EvaluationResult:
    macro_f05: float
    entities: int
    exact_lists: int
    true_singletons: int
    correct_singletons: int
    false_merges_on_singletons: int


def f05_for_entity(truth: set[str], predicted: set[str]) -> float:
    """Return the competition's F_0.5 value for one S1 entity.

    Empty/empty is a correct singleton and scores 1.0. Any nonempty prediction
    for an empty truth scores 0.0. This avoids undefined precision/recall cases.
    """
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    true_positives = len(truth & predicted)
    if not true_positives:
        return 0.0
    precision = true_positives / len(predicted)
    recall = true_positives / len(truth)
    beta_squared = 0.25
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def evaluate_predictions(
    truth_by_s1: dict[str, set[str]], prediction_by_s1: dict[str, set[str]]
) -> EvaluationResult:
    """Compute macro F_0.5; every truth S1 must have exactly one prediction row."""
    truth_ids, prediction_ids = set(truth_by_s1), set(prediction_by_s1)
    missing, extra = truth_ids - prediction_ids, prediction_ids - truth_ids
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} S1 IDs")
        if extra:
            detail.append(f"unexpected {len(extra)} S1 IDs")
        raise ValueError("Prediction IDs do not match truth IDs: " + "; ".join(detail))

    scores = []
    exact_lists = true_singletons = correct_singletons = false_merges = 0
    for source_id, truth in truth_by_s1.items():
        predicted = prediction_by_s1[source_id]
        scores.append(f05_for_entity(truth, predicted))
        exact_lists += predicted == truth
        if not truth:
            true_singletons += 1
            correct_singletons += not predicted
            false_merges += bool(predicted)
    return EvaluationResult(
        macro_f05=sum(scores) / len(scores) if scores else 0.0,
        entities=len(scores),
        exact_lists=exact_lists,
        true_singletons=true_singletons,
        correct_singletons=correct_singletons,
        false_merges_on_singletons=false_merges,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute challenge macro F_0.5 locally.")
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    args = parser.parse_args()

    truth = read_id_lists(args.truth, "matched_entity_ids")
    prediction = read_id_lists(args.prediction, "matched_entity_ids")
    result = evaluate_predictions(truth, prediction)
    print(f"entities: {result.entities}")
    print(f"macro_f0.5: {result.macro_f05:.8f}")
    print(f"exact_match_lists: {result.exact_lists}")
    print(f"true_singletons: {result.true_singletons}")
    print(f"correct_singletons: {result.correct_singletons}")
    print(f"false_merges_on_singletons: {result.false_merges_on_singletons}")


if __name__ == "__main__":
    main()
