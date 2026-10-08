"""Verified scene inputs; runtime data is separate from offline evaluation data."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'Invalid {name}')
    return value


def local_path(root, value):
    require(isinstance(value, str), 'Path must be a string')
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and '..' not in relative.parts and bool(relative.parts), 'Expected project-relative path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Path escapes project root')
    return path


@dataclass(frozen=True)
class FileRef:
    path: Path
    sha256: str

    def verify(self):
        require(sha256(self.path) == self.sha256, f'Checksum mismatch: {self.path}')
        return self.path


def file_ref(root, value):
    digest = value['sha256']
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid SHA-256')
    return FileRef(local_path(root, value['path']), digest)


@dataclass(frozen=True)
class CameraInput:
    camera_id: int
    video: FileRef
    width: int
    height: int
    homography: np.ndarray


@dataclass(frozen=True)
class RuntimeScene:
    scene: str
    dataset: str
    revision: str
    fps: int
    rounds: int
    cameras: tuple[CameraInput, ...]
    calibration: FileRef
    coordinate_space: str

    @property
    def camera_ids(self):
        return tuple(c.camera_id for c in self.cameras)

    @property
    def sources(self):
        return {c.camera_id: c.video.path for c in self.cameras}

    @property
    def matrices(self):
        return {c.camera_id: c.homography for c in self.cameras}


@dataclass(frozen=True)
class EvaluationSpec:
    ground_truth: FileRef
    camera_ids: tuple[int, ...]
    first_frame: int
    last_frame: int
    gt_to_video_offset: int
    min_iou: float
    empty_slot_policy: str


@dataclass(frozen=True)
class SceneInputs:
    runtime: RuntimeScene
    evaluation: EvaluationSpec
    configuration: FileRef
    source_manifest: FileRef
    video_manifest: FileRef


def load_scene(config_path, *, project_root):
    """Verify manifests, calibration and videos; never open GT contents.

    The caller passes only .runtime to tracking. Offline evaluation explicitly
    opens .evaluation.ground_truth. Native calibration scale is preserved.
    """
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    require(config['schema_version'] == 1, 'Unsupported scene schema')
    scene = config['scene']; parts = PurePosixPath(scene).parts
    require(len(parts) == 3 and parts[0] == 'MTMC_Tracking_2024' and
            parts[1] in ('train','val','test','eval') and re.fullmatch(r'scene_\d+', parts[2]), 'Invalid scene scope')
    require(config['dataset'] == 'nvidia/PhysicalAI-SmartSpaces' and
            isinstance(config['revision'],str) and re.fullmatch('[0-9a-f]{40}',config['revision']), 'Invalid dataset/revision')
    ids = config['camera_ids']
    require(isinstance(ids,list) and ids and all(type(c) is int and c>=0 for c in ids)
            and ids == sorted(set(ids)), 'Cameras must be unique, sorted nonnegative integers')
    fps = integer(config['fps'], 'fps', 1)
    runtime = config['runtime']; evaluation = config['evaluation']
    require(type(runtime['first_frame']) is int and runtime['first_frame'] == 0, 'Runtime must start at frame zero')
    rounds = integer(runtime['rounds'], 'rounds', 1)
    first = integer(evaluation['first_frame'], 'evaluation first frame')
    last = integer(evaluation['last_frame'], 'evaluation last frame')
    require(first <= last < rounds, 'Evaluation interval outside runtime')
    offset = evaluation['gt_to_video_offset']; require(type(offset) is int, 'Invalid GT offset')
    gate = evaluation['min_iou']
    require(type(gate) in (int,float) and np.isfinite(gate) and 0 < gate <= 1, 'Invalid IoU gate')
    require(evaluation['empty_slot_policy'] == 'retain_slot_and_count_predictions', 'Unsupported empty-slot policy')
    source_ref = file_ref(root,config['source_manifest']); video_ref = file_ref(root,config['video_manifest'])
    source = json.loads(source_ref.verify().read_text()); videos = json.loads(video_ref.verify().read_text())
    for manifest in (source,videos):
        require(all(manifest[k] == config[k] for k in ('dataset','revision','scene')), 'Mixed scene/revision manifests')
    require(videos['camera_ids'] == ids and len(videos['files']) == len(ids)
            and sorted(x['camera'] for x in videos['files']) == ids, 'Video camera coverage differs')
    def annotation(name):
        items = [x for x in source['files'] if x['remote_path'] == f'{scene}/{name}']
        require(len(items) == 1, f'Missing/duplicate annotation reference: {name}')
        item = items[0]
        return file_ref(root,{'path':item['local_path'],'sha256':item['sha256']})
    calibration_ref = annotation('calibration_2025_format.json')
    # Constructing the GT reference does not read that file.
    ground_ref = annotation('ground_truth.txt')
    calibration = json.loads(calibration_ref.verify().read_text())
    cameras = []
    for camera in ids:
        item = next(v for v in videos['files'] if v['camera']==camera)
        require(item['remote_path']==f'{scene}/camera_{camera:04d}/video.mp4', 'Video remote scope differs')
        video = file_ref(root,{'path':item['local_path'],'sha256':item['sha256']})
        require(video.path.stat().st_size == integer(item['size_bytes'],'video size',1), 'Video size mismatch')
        video.verify()
        sensors = [s for s in calibration['sensors'] if s.get('id','').lower()==f'camera_{camera:04d}']
        require(len(sensors)==1, 'Missing/duplicate calibrated camera')
        sensor=sensors[0];attrs={a['name']:a['value'] for a in sensor['attributes']}
        width,height=int(attrs['frameWidth']),int(attrs['frameHeight'])
        require(width>0 and height>0 and Fraction(attrs['fps'])==fps, 'Calibration dimensions/FPS differ')
        matrices=[]
        for name,shape in [('intrinsicMatrix',(3,3)),('extrinsicMatrix',(3,4)),('cameraMatrix',(3,4)),('homography',(3,3))]:
            a=np.asarray(sensor[name],dtype=np.float64)
            require(a.shape==shape and np.isfinite(a).all(), f'Invalid {name}')
            matrices.append(a)
        k,e,p,h=matrices
        for a,b in ((p,k@e),(h,p[:,[0,1,3]])):
            require(np.linalg.norm(a)>0 and np.linalg.norm(b)>0, 'Zero projective matrix')
            a,b=a/np.linalg.norm(a),b/np.linalg.norm(b)
            require(min(np.linalg.norm(a-b),np.linalg.norm(a+b))<=1e-6,'Projection convention differs')
        require(np.linalg.matrix_rank(h)==3,'Singular ground-plane homography')
        h.setflags(write=False)
        cameras.append(CameraInput(camera,video,width,height,h))
    # Preserve the scene_001 namespace convention; calibration digest disambiguates scope.
    space=f'{parts[2]}/Z0/native/calibration={calibration_ref.sha256}'
    return SceneInputs(
        RuntimeScene(scene,config['dataset'],config['revision'],fps,rounds,tuple(cameras),calibration_ref,space),
        EvaluationSpec(ground_ref,tuple(ids),first,last,offset,float(gate),evaluation['empty_slot_policy']),
        FileRef(config_path,sha256(config_path)),source_ref,video_ref)
