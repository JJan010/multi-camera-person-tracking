"""Small known-answer tests for causal appearance memory; no models or CUDA."""

from fractions import Fraction

import numpy as np

from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch


def unit(index):
    result = np.zeros(512, dtype=np.float32)
    result[index] = 1
    return result


def batch(frame, entries, time=None):
    time = Fraction(frame, 30) if time is None else time
    return ReIDBatch(tuple(ObservationKey(camera, identity, frame) for camera, identity, _ in entries),
                     (time,) * len(entries), np.asarray([v for _, _, v in entries], dtype=np.float32).reshape(-1, 512))


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    x, y, z = unit(0), unit(1), unit(2)
    memory = AppearanceHistory("run-a", max_observations=2)
    first = batch(0, [(4, 7, x), (8, 7, z)])
    result = memory.update(0, Fraction(0), first)
    # Neither caller input mutation nor returned-array mutation can corrupt cache.
    first.embeddings[:] = 0
    result.latest.embeddings[:] = 0
    result.mean.embeddings[:] = 0
    result = memory.update(1, Fraction(1, 30), batch(1, [(8, 7, z), (4, 7, y)]))
    np.testing.assert_allclose(result.mean.embeddings[0], z, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.mean.embeddings[1], (x + y) / np.sqrt(2), rtol=0, atol=1e-7)
    require(result.source_frames == ((0, 1), (0, 1)), "History/key mapping differs")
    result = memory.update(2, Fraction(2, 30), batch(2, [(4, 7, y)]))
    np.testing.assert_array_equal(result.mean.embeddings[0], y)
    require(result.source_frames == ((1, 2),), "Oldest sample was not removed")
    require(memory.active_tracks == 2 and memory.stored_vectors == 4, "Absent track state was lost")
    print("Known mean, camera isolation, row order, owned memory and observation limit: OK")

    latest_only = AppearanceHistory("latest", max_observations=1)
    for frame, vector in enumerate((x, y, z)):
        output = latest_only.update(frame, Fraction(frame, 30), batch(frame, [(4, 7, vector)]))
        np.testing.assert_array_equal(output.latest.embeddings[0], vector)
        np.testing.assert_allclose(output.mean.embeddings[0], vector, rtol=0, atol=1e-7)
    print("Window size 1 agrees with the latest descriptor: OK")

    expiry = AppearanceHistory("expiry", max_age=Fraction(1))
    expiry.update(0, Fraction(0), batch(0, [(4, 7, x)]))
    output = expiry.update(30, Fraction(1), batch(30, []))
    require(output.mean.embeddings.shape == (0, 512) and expiry.active_tracks == 1,
            "Sample at exact age limit must remain; absent track must not be emitted")
    expiry.update(31, Fraction(31, 30), batch(31, []))
    require(expiry.active_tracks == 0, "Expired history was not removed on an empty round")
    output = expiry.update(32, Fraction(32, 30), batch(32, [(4, 7, y)]))
    require(output.source_frames == ((32,),), "Expired samples leaked into return")
    fresh = AppearanceHistory("new-run")
    require(fresh.update(0, Fraction(0), batch(0, [(4, 7, z)])).source_frames == ((0,),), "Run isolation failed")
    print("Scene-time expiry, exact boundary, empty rounds and new-run isolation: OK")

    guarded = AppearanceHistory("guarded")
    guarded.update(0, Fraction(0), batch(0, [(4, 7, x)]))
    invalid = [batch(1, [(4, 7, y), (4, 7, y)]), batch(2, [(4, 7, y)]),
               batch(1, [(4, 7, np.zeros(512))]), batch(1, [(4, 7, y)], Fraction(2, 30))]
    for value in invalid:
        try:
            guarded.update(1, Fraction(1, 30), value)
        except ValueError:
            continue
        raise AssertionError("Invalid round accepted")
    output = guarded.update(1, Fraction(1, 30), batch(1, [(4, 7, y)]))
    require(output.source_frames == ((0, 1),), "Rejected input changed state")
    try:
        guarded.update(1, Fraction(1, 30), batch(1, []))
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated round accepted")
    frozen = output.mean.embeddings.copy()
    guarded.update(2, Fraction(2, 30), batch(2, [(4, 7, z)]))
    np.testing.assert_array_equal(output.mean.embeddings, frozen)
    print("Invalid/repeated rounds rejected atomically; future updates preserve past output: OK")

    cancellation = AppearanceHistory("cancellation")
    cancellation.update(0, Fraction(0), batch(0, [(4, 7, x)]))
    output = cancellation.update(1, Fraction(1, 30), batch(1, [(4, 7, -x)]))
    require(output.used_latest_fallback == (True,), "Cancellation fallback not reported")
    np.testing.assert_array_equal(output.mean.embeddings[0], -x)
    print("Zero-mean cancellation uses an explicitly reported latest-vector fallback: OK")
    print("Appearance history checks: PASSED")


if __name__ == "__main__":
    main()
