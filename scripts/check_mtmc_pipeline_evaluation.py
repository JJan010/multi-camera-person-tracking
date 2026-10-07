"""Known-answer joins and coverage checks for the full-pipeline evaluator."""
from copy import deepcopy
from fractions import Fraction
import numpy as np

import evaluate_mtmc_pipeline as e


CONFIG = {"appearance_variant":"mean","appearance_threshold":.7,
          "geometry":{"max_distance":2.,"unavailable_policy":"appearance_only"},"coordinate_space":"synthetic-plane"}


def fixture(frame=2, outside=False):
    time = str(Fraction(frame,30))
    cameras,observations,assignments = [],[],[]
    for index,camera in enumerate((4,5)):
        box = [-20,10,-10,30] if outside and camera == 5 else [10,10,20,30]
        encoded = not (outside and camera == 5)
        key = {"camera_id":camera,"local_id":1,"frame_index":frame}
        cameras.append({"camera":camera,"local_ids":[1],"xyxy":[box]})
        observations.append({"camera":camera,"local_id":1,"frame_index":frame,"source_xyxy":box,
                             "status":"encoded" if encoded else "skipped_fully_outside",
                             "embedding_row":index if encoded else None,"crop_xyxy_int":box if encoded else None})
        assignments.append({"key":key,"global_id":1,"embedding_row":index if encoded else None,
                            "has_current_embedding":encoded})
    cameras.append({"camera":8,"local_ids":[],"xyxy":[]})
    local = {"run_id":"test","frame_index":frame,"timestamp":time,"cameras":cameras,"reid_observations":observations}
    record = {"run_id":"test","identity_scope":"test","frame_index":frame,"timestamp":time,
              "policy":e.controlled_merge.POLICY,"association_policy":e.geometry.POLICY,
              "descriptor_variant":"mean","min_similarity":.7,"geometry_configuration":CONFIG["geometry"],
              "coordinate_space":CONFIG["coordinate_space"],"assignments":assignments,
              "unencoded_singletons":[assignments[1]["key"]] if outside else []}
    return local,record


def check(local,record):
    return e.validate_round(local,record,frame=2,run="test",config=CONFIG,next_row=0)


def rejected(function):
    try:
        function()
    except (ValueError,TypeError):
        return
    raise AssertionError("Invalid trace accepted")


def main():
    local,record = fixture()
    cameras,assignments,rows = check(local,record)
    full,control = e.metric.IdentityCounts(),e.metric.IdentityCounts()
    for i,camera in enumerate((4,5)):
        keys,boxes = cameras[camera]
        gt,mask,unique,excluded,outside = e.spatial_slot({7:[10,10,20,30]},keys,boxes)
        full.update(gt,[assignments[k] for k in keys],mask)
        control.update(gt,[i+1],mask)
    assert rows == 2 and full.result()[0]["idf1"] == 1 and control.result()[0]["idf1"] == .5
    reordered = deepcopy(record);reordered["assignments"].reverse()
    shuffled = deepcopy(local);shuffled["cameras"].reverse()
    assert check(shuffled,reordered)[1:] == (assignments,rows)
    print("Shared global identity gives 100%, camera-scoped control 50%; order-independent joins: OK")
    local,record = fixture(outside=True)
    cameras,assignments,rows = check(local,record)
    assert rows == 1 and len(assignments) == 2
    keys,boxes = cameras[5]
    gt,mask,unique,excluded,outside = e.spatial_slot({7:[10,10,20,30]},keys,boxes)
    counts = e.metric.IdentityCounts();counts.update(gt,[assignments[k] for k in keys],mask)
    assert outside == 1 and not unique and counts.result()[0]["idfp"] == counts.result()[0]["idfn"] == 1
    print("Unencoded fully outside prediction remains in denominators; it cannot spatially match: OK")
    gt,mask,unique,excluded,outside = e.spatial_slot({7:[-20,10,-10,30]},(),np.empty((0,4)))
    assert gt == [] and excluded == 1 and mask.shape == (0,0)
    gt,mask,*_ = e.spatial_slot({7:[0,0,10,10]},(),np.empty((0,4)))
    assert gt == [7] and mask.shape == (1,0)
    print("Zero-area GT exclusion and empty prediction camera: OK")
    local,record = fixture()
    for edit in (
        lambda r:r.update(timestamp="1/30"),
        lambda r:r.update(identity_scope="other"),
        lambda r:r.update(coordinate_space="other"),
        lambda r:r["assignments"][0].update(embedding_row=3),
        lambda r:r["assignments"].pop(),
        lambda r:r["assignments"].append(deepcopy(r["assignments"][0])),
        lambda r:r["assignments"][0].update(has_current_embedding=False),
    ):
        broken = deepcopy(record);edit(broken)
        rejected(lambda:check(local,broken))
    broken = deepcopy(local);broken["reid_observations"][0]["source_xyxy"] = [11,10,20,30]
    rejected(lambda:check(broken,record))
    print("Wrong time/scope/geometry, row mapping, coverage, duplicate assignment and changed box rejected: OK")
    print("MTMC pipeline evaluation checks: PASSED")


if __name__ == "__main__":
    main()
