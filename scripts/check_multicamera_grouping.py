"""Known-answer grouping checks. All supplied scores are synthetic."""

from dataclasses import replace
from fractions import Fraction
from itertools import combinations, permutations

import numpy as np

from mtmc.association.grouping import group_pair_associations, edge_key
from mtmc.association.pairwise import CameraAppearance, PairAssociation, PairMatch, UnmatchedObservation, associate_camera_pair
from mtmc.reid.osnet import ObservationKey, ReIDBatch


def fixture(nodes, edges):
    results = []
    for a, b in combinations((4, 5, 8), 2):
        matches = []
        for left, right, score in edges:
            left, right = edge_key(left, right)
            if (left.camera_id, right.camera_id) == (a, b):
                matches.append(PairMatch(left, right, score))
        used = {k for m in matches for k in (m.left, m.right)}
        unmatched = {c: tuple(UnmatchedObservation(k, "no_candidate_above_threshold", .1)
                              for k in nodes if k.camera_id == c and k not in used) for c in (a, b)}
        results.append(PairAssociation("synthetic-group-test", 2, Fraction(1, 15), "mean", a, b, .8,
                                       tuple(matches), unmatched[a], unmatched[b]))
    return tuple(results)


def groups(result):
    return {frozenset(group) for group in result.groups}


def reject(call):
    try:
        call()
    except (TypeError, ValueError):
        return
    raise AssertionError("Invalid grouping input was accepted")


def main():
    a, a2, b, c = [ObservationKey(camera, identity, 2) for camera, identity in ((4, 1), (4, 2), (5, 1), (8, 1))]
    triangle = fixture([a, b, c], [(a, b, .95), (b, c, .94), (a, c, .93)])
    complete = group_pair_associations(triangle)
    assert groups(complete) == {frozenset((a, b, c))}
    assert [d.outcome for d in complete.decisions] == ["merged", "merged", "already_in_same_group"]
    print("Three mutually supported observations form one group: OK")

    chain = group_pair_associations(fixture([a, b, c], [(a, b, .95), (b, c, .94)]))
    assert groups(chain) == {frozenset((a, b)), frozenset((c,))}
    assert chain.decisions[-1].outcome == "rejected_missing_pair_support"
    assert chain.decisions[-1].missing_support == ((a, c),)
    # Assume all three are one true person: this conservative policy keeps only
    # one of their three possible within-person pairs. GT is not a module input.
    retained_pairs = sum(len(group) * (len(group) - 1) // 2 for group in chain.groups)
    assert retained_pairs == 1
    print("Open chain remains a pair plus singleton; missing support is recorded: OK")
    print("Known tradeoff: a true three-view person can be split by this policy: DEMONSTRATED")

    conflict = group_pair_associations(fixture([a, a2, b, c], [(a, b, .99), (a2, c, .98), (b, c, .97)]))
    assert groups(conflict) == {frozenset((a, b)), frozenset((a2, c))}
    assert conflict.decisions[-1].outcome == "rejected_camera_conflict"
    assert conflict.decisions[-1].conflicting_cameras == (4,)
    print("Joining groups containing two local IDs from camera 4 is rejected: OK")

    none = group_pair_associations(fixture([a, a2, b, c], []))
    assert len(none.groups) == 4 and all(len(x) == 1 for x in none.groups)
    assert group_pair_associations(fixture([], [])).groups == ()
    assert groups(group_pair_associations(fixture([a], []))) == {frozenset((a,))}
    print("Unmatched observations, empty cameras and empty rounds are preserved: OK")

    for order in permutations(triangle):
        assert group_pair_associations(order) == complete
    reversed_pairs = tuple(replace(r, left_camera=r.right_camera, right_camera=r.left_camera,
                                   matches=tuple(PairMatch(m.right, m.left, m.cosine_similarity) for m in r.matches),
                                   unmatched_left=r.unmatched_right, unmatched_right=r.unmatched_left) for r in triangle)
    assert group_pair_associations(reversed_pairs) == complete
    equal = fixture([c, b, a], [(b, c, .9), (a, b, .9)])
    assert groups(group_pair_associations(equal)) == {frozenset((a, b)), frozenset((c,))}
    for order in permutations(equal):
        assert group_pair_associations(order) == group_pair_associations(equal)
    print("Pair order, orientation and exact-score ties use a canonical result: OK")

    # A greedy choice can be suboptimal even though its output is structurally valid.
    suboptimal = group_pair_associations(fixture([a, a2, b, c], [(b, c, .99), (a, b, .98), (a2, c, .97)]))
    assert groups(suboptimal) == {frozenset((b, c)), frozenset((a,)), frozenset((a2,))}
    assert (.98 - .8) + (.97 - .8) > (.99 - .8)
    print("Greedy policy does not claim a globally optimal partition: DEMONSTRATED")

    cameras = []
    for camera, axes in ((4, [0, 1]), (5, [1, 0]), (8, [0, 1])):
        features = np.zeros((2, 512), dtype=np.float32)
        features[np.arange(2), axes] = 1
        batch = ReIDBatch(tuple(ObservationKey(camera, i, 2) for i in (1, 2)),
                          (Fraction(1, 15),) * 2, features)
        cameras.append(CameraAppearance("adapter-test", camera, 2, Fraction(1, 15), "mean", batch))
    actual_pairs = [associate_camera_pair(x, y, min_similarity=.8) for x, y in combinations(cameras, 2)]
    integrated = group_pair_associations(actual_pairs)
    assert {frozenset((k.camera_id, k.local_id) for k in group) for group in integrated.groups} == {
        frozenset(((4, 1), (5, 2), (8, 1))), frozenset(((4, 2), (5, 1), (8, 2)))}
    print("Synthetic ReIDBatch -> pairwise API -> two complete three-camera groups: OK")

    reject(lambda: group_pair_associations(()))
    reject(lambda: group_pair_associations(triangle[:2]))
    reject(lambda: group_pair_associations(triangle + triangle[:1]))
    for field, value in (("run_id", "other"), ("frame_index", 3), ("timestamp", Fraction(2, 15)),
                         ("descriptor_variant", "latest"), ("min_similarity", .9)):
        bad = (replace(triangle[0], **{field: value}), *triangle[1:])
        reject(lambda bad=bad: group_pair_associations(bad))
    bad_match = replace(triangle[0].matches[0], cosine_similarity=.8)
    reject(lambda: group_pair_associations((replace(triangle[0], matches=(bad_match,)), *triangle[1:])))
    reject(lambda: group_pair_associations((replace(triangle[0], matches=()), *triangle[1:])))
    reject(lambda: group_pair_associations((replace(triangle[0], matches=triangle[0].matches * 2), *triangle[1:])))
    print("Mixed scopes, incomplete camera pairs, missing observations and invalid links rejected: OK")
    print("Output groups describe one scene time; they are not persistent global IDs.")
    print("Multi-camera grouping smoke test: PASSED")


if __name__ == "__main__":
    main()
