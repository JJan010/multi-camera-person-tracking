"""CPU-only known-answer checks for candidate keys, crops and feature caching."""
from fractions import Fraction
from types import SimpleNamespace as NS
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from cache_detection_embeddings import prepare_candidates, encode_chunks, parity_sample, EmbeddingArchive
from mtmc.reid.osnet import ReIDBatch


def fixture():
    image = np.arange(1080 * 1920 * 3, dtype=np.uint8).reshape(1080, 1920, 3)
    packets = tuple(NS(camera_id=c, frame_index=0, timestamp=Fraction(0), rgb=image) for c in (8, 4, 5))
    local = {'run_id': 'fixture', 'frame_index': 0, 'timestamp': '0', 'cameras': [
        {'camera': 5, 'detector_xyxy': [], 'detector_confidence': []},
        {'camera': 8, 'detector_xyxy': [[1, 2, 4, 6]], 'detector_confidence': [.2]},
        {'camera': 4, 'detector_xyxy': [[-.2, 1.2, 3.1, 5.8], [2000, 0, 2001, 1],
                                       [-.2, 1.2, 3.1, 5.8]], 'detector_confidence': [.8, .9, .1]}]}
    return local, NS(frame_index=0, timestamp=Fraction(0), frames=packets)


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, crops):
        self.calls.append(len(crops))
        a = np.zeros((len(crops), 512), np.float32)
        for i, crop in enumerate(crops):
            a[i, int(crop.rgb[0, 0, 0])] = 1
        return ReIDBatch(tuple(c.key for c in crops), tuple(c.timestamp for c in crops), a)


def rejected(callback):
    try:
        callback()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Invalid input accepted')


def main():
    local, batch = fixture()
    crops, records = prepare_candidates(local, batch, source_run='fixture', row_offset=10)
    assert [(c.key.camera_id, c.key.local_id) for c in crops] == [(4, 0), (4, 2), (8, 0)]
    assert [r['embedding_row'] for r in records] == [10, None, 11, 12]
    assert records[0]['crop_xyxy_int'] == (0, 1, 4, 6)
    assert np.array_equal(crops[0].rgb, batch.frames[1].rgb[1:6, :4])
    assert np.array_equal(crops[0].rgb, crops[1].rgb)
    assert not crops[0].rgb.flags.writeable and batch.frames[1].rgb.flags.writeable
    print('Camera/key mapping, duplicate detections, outside-box rows and exact clipped RGB: OK')
    local['cameras'].reverse()
    reordered, reordered_records = prepare_candidates(local, batch, source_run='fixture', row_offset=10)
    assert tuple(c.key for c in reordered) == tuple(c.key for c in crops) and reordered_records == records
    encoder = FakeEncoder()
    features = encode_chunks(encoder, crops, 2)
    assert encoder.calls == [2, 1]
    assert encode_chunks(encoder, (), 2).shape == (0, 512) and encoder.calls == [2, 1]
    assert np.array_equal(features, encode_chunks(FakeEncoder(), crops, 1))
    print('Bounded batches preserve order; empty batches skip encoder; partition parity: OK')
    local['reid_observations'] = [{'camera': 4, 'local_id': 11, 'crop_xyxy_int': [0, 1, 4, 6],
                                  'embedding_row': 0}]
    samples = parity_sample(local, records, features, 10, features[:1])
    assert len(samples) == 1 and samples[0]['cosine'] == 1 and samples[0]['cache_embedding_row'] == 10
    wrong = np.zeros((1, 512), np.float32)
    wrong[0, (np.argmax(features[0]) + 1) % 512] = 1
    rejected(lambda: parity_sample(local, records, features, 10, wrong))
    rejected(lambda: prepare_candidates(local, batch, source_run='another-run', row_offset=0))
    local['cameras'][0]['detector_confidence'] = [-1]
    rejected(lambda: prepare_candidates(local, batch, source_run='fixture', row_offset=0))
    print('Exact-crop reference mapping, wrong embeddings and invalid source inputs: OK')
    with TemporaryDirectory() as directory:
        archive = EmbeddingArchive(Path(directory) / 'features.npy')
        archive.append(features[:1]); archive.append(features[1:]); archive.finalize()
        assert np.array_equal(np.load(archive.path, allow_pickle=False), features)
    print('Streamed archive preserves every feature row: OK')
    print('Detection embedding cache checks: PASSED (synthetic encoder; no GPU inference)')


if __name__ == '__main__':
    main()
