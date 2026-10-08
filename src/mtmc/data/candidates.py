"""Scene-aware detector candidate crops and persisted embedding-row contract."""
from fractions import Fraction

import numpy as np

from mtmc.data.scene import RuntimeScene, require
from mtmc.reid.crops import crop_geometry
from mtmc.reid.osnet import ObservationKey, PersonCrop


def validate_frame_batch(scene, batch, *, expected_frame):
    require(isinstance(scene, RuntimeScene), 'Expected runtime-only scene')
    require(type(expected_frame) is int and 0 <= expected_frame < scene.rounds
            and type(batch.frame_index) is int and batch.frame_index == expected_frame
            and isinstance(batch.timestamp, Fraction)
            and batch.timestamp == Fraction(expected_frame, scene.fps), 'Invalid scene frame/time')
    images = {}
    sizes = {c.camera_id: (c.width,c.height) for c in scene.cameras}
    for packet in batch.frames:
        c = packet.camera_id
        require(type(c) is int and c in sizes and c not in images, 'Invalid/duplicate frame camera')
        require(type(packet.frame_index) is int and packet.frame_index == expected_frame
                and isinstance(packet.timestamp, Fraction) and packet.timestamp == batch.timestamp,
                'Packet frame/time differs')
        w,h = sizes[c]
        require(isinstance(packet.rgb,np.ndarray) and packet.rgb.dtype == np.uint8
                and packet.rgb.shape == (h,w,3), 'RGB dimensions/type differ from scene')
        images[c] = packet
    require(set(images) == set(sizes), 'Missing camera frame')
    return images


def prepare_candidate_crops(scene, batch, detections, *, row_offset):
    """Preserve every candidate and borrow its original RGB pixels, without GT.

    ObservationKey.local_id temporarily carries the detection index for the
    existing encoder API. Persisted keys explicitly call it detection_index.
    This index is frame-local; it is not a tracker identity.
    """
    require(type(row_offset) is int and row_offset >= 0, 'Invalid embedding offset')
    images = validate_frame_batch(scene, batch, expected_frame=batch.frame_index)
    by_camera = {}
    for item in detections:
        c = item.camera_id
        require(type(c) is int and c in images and c not in by_camera, 'Invalid/duplicate detector camera')
        require(type(item.frame_index) is int and item.frame_index == batch.frame_index
                and isinstance(item.timestamp,Fraction) and item.timestamp == batch.timestamp, 'Detector frame/time differs')
        boxes, scores = item.xyxy, item.confidence
        require(isinstance(boxes,np.ndarray) and boxes.dtype == np.float32 and boxes.ndim == 2
                and boxes.shape[1] == 4 and np.isfinite(boxes).all()
                and np.all(boxes[:,2:] > boxes[:,:2]), 'Invalid detector boxes')
        require(isinstance(scores,np.ndarray) and scores.dtype == np.float32
                and scores.shape == (len(boxes),) and np.isfinite(scores).all()
                and np.all((scores >= 0) & (scores <= 1)), 'Invalid detector confidences')
        by_camera[c] = item
    require(set(by_camera) == set(images), 'Missing detector camera, including empty cameras')
    crops, records = [], []
    for c in sorted(images):
        rgb, detection = images[c].rgb, by_camera[c]
        h,w,_ = rgb.shape
        for index,(box,score) in enumerate(zip(detection.xyxy,detection.confidence)):
            bounds,fraction = crop_geometry(box,w,h)
            row = row_offset + len(crops) if bounds is not None else None
            records.append({'camera_id':c,'detection_index':index,'frame_index':batch.frame_index,
                'xyxy':box.tolist(),'confidence':float(score),'crop_xyxy_int':bounds,
                'inside_image_fraction':fraction,'embedding_row':row})
            if bounds is not None:
                x1,y1,x2,y2 = bounds
                view = rgb[y1:y2,x1:x2].view(); view.setflags(write=False)
                crops.append(PersonCrop(ObservationKey(c,index,batch.frame_index),batch.timestamp,view))
    return tuple(crops), records


def cache_record(run_id, scene, batch, records):
    require(isinstance(run_id,str) and bool(run_id.strip()), 'Missing cache scope')
    return {'cache_run_id':run_id,'scene':scene.scene,'frame_index':batch.frame_index,
            'timestamp':str(batch.timestamp),'camera_ids':list(scene.camera_ids),'detections':records}


def validate_cache_round(record, scene, vectors, *, run_id, frame, row_offset):
    """Validate persisted records after writing; return next row and outside count.

    All cameras are explicitly listed, including empty ones. Candidate indices
    restart at zero per camera/frame; embedding rows advance across the run.
    """
    require(record['cache_run_id'] == run_id and record['scene'] == scene.scene
            and type(record['frame_index']) is int and record['frame_index'] == frame
            and record['timestamp'] == str(Fraction(frame,scene.fps))
            and record['camera_ids'] == list(scene.camera_ids), 'Persisted candidate scope differs')
    require(isinstance(vectors,np.ndarray) and vectors.dtype == np.float32
            and vectors.ndim == 2 and vectors.shape[1] == 512, 'Invalid feature archive')
    sizes = {c.camera_id:(c.width,c.height) for c in scene.cameras}
    counts = dict.fromkeys(scene.camera_ids,0)
    previous_camera = -1; outside = 0; next_row = row_offset
    for item in record['detections']:
        c,i = item['camera_id'],item['detection_index']
        require(type(c) is int and c in sizes and c >= previous_camera
                and type(i) is int and i == counts[c]
                and type(item['frame_index']) is int and item['frame_index'] == frame, 'Candidate order/key differs')
        previous_camera = c; counts[c] += 1
        box = np.asarray(item['xyxy'],np.float64); score = item['confidence']
        require(box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2])
                and type(score) in (int,float) and np.isfinite(score) and 0 <= score <= 1,
                'Invalid persisted detection')
        require(np.array_equal(box.astype(np.float32).astype(np.float64),box)
                and float(np.float32(score)) == score, 'Persisted detection lost float32 parity')
        bounds,fraction = crop_geometry(box,*sizes[c])
        require(item['crop_xyxy_int'] == (list(bounds) if bounds is not None else None)
                and item['inside_image_fraction'] == fraction, 'Persisted crop geometry differs')
        if bounds is None:
            require(item['embedding_row'] is None,'Outside candidate has an embedding')
            outside += 1
        else:
            require(type(item['embedding_row']) is int and item['embedding_row'] == next_row
                    and 0 <= next_row < len(vectors), 'Invalid/reused embedding row')
            vector = vectors[next_row]
            require(np.isfinite(vector).all() and abs(float(np.linalg.norm(vector))-1) <= 1e-5,
                    'Invalid normalized embedding')
            next_row += 1
    return next_row, outside, counts
