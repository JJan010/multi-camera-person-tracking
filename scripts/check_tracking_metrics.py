"""Check identity metrics against small examples with known answers."""

import math
from importlib.metadata import version

import motmetrics as mm
import numpy as np


METRICS = ["idf1", "num_switches", "num_false_positives", "num_misses"]


def check(name, frames, expected):
    accumulator = mm.MOTAccumulator(auto_id=False)

    for frame_index, (gt_ids, pred_ids, distances) in enumerate(frames):
        distances = np.asarray(distances, dtype=float).reshape(
            len(gt_ids), len(pred_ids)
        )
        accumulator.update(gt_ids, pred_ids, distances, frameid=frame_index)

    result = mm.metrics.create().compute(
        accumulator, metrics=METRICS, name=name
    ).loc[name]

    for metric, target in zip(METRICS, expected):
        actual = float(result[metric])
        if not math.isclose(actual, target, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(
                f"{name}: {metric}={actual}, expected {target}"
            )

    print(
        f"{name:20} IDF1={result['idf1']:7.2%} "
        f"IDSW={int(result['num_switches'])} "
        f"FP={int(result['num_false_positives'])} "
        f"FN={int(result['num_misses'])}: OK"
    )


def main():
    for package in ("motmetrics", "numpy", "scipy", "pandas"):
        print(f"{package}: {version(package)}")

    mm.lap.default_solver = "scipy"

    # Expected values: IDF1, ID switches, false positives, misses.
    check(
        "correct_id",
        [([1], [7], [[0]])] * 4,
        (1, 0, 0, 0),
    )
    check(
        "split_id",
        [([1], [i], [[0]]) for i in (7, 7, 8, 8)],
        (0.5, 1, 0, 0),
    )

    diagonal = [[0, np.nan], [np.nan, 0]]
    check(
        "swapped_ids",
        [
            ([1, 2], ids, diagonal)
            for ids in ([7, 8], [7, 8], [8, 7], [8, 7])
        ],
        (0.5, 2, 0, 0),
    )
    check(
        "merged_id",
        [([i], [7], [[0]]) for i in (1, 1, 2, 2)],
        (0.5, 0, 0, 0),
    )
    check(
        "missed_observations",
        [
            ([1], [7], [[0]]),
            ([1], [], []),
            ([1], [], []),
            ([1], [7], [[0]]),
        ],
        (2 / 3, 0, 0, 2),
    )
    check(
        "false_observations",
        [([1], [7, 99], [[0, np.nan]])] * 4,
        (2 / 3, 0, 4, 0),
    )
    check(
        "no_spatial_match",
        [([1], [7], [[np.nan]])] * 4,
        (0, 0, 4, 4),
    )
    check(
        "empty_frame",
        [([], [], []), ([1], [7], [[0]])],
        (1, 0, 0, 0),
    )

    print("Tracking metrics smoke test: PASSED")


if __name__ == "__main__":
    main()
