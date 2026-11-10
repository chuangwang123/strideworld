"""Shared, explicit official AlayaWorld input contract."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

COMMIT = 'ea03cfbb2e4c4e9102ed8ea8562e0b5370ca9b79'
HOST_ROOT = Path('<path>/anon')
REPRO_ROOT = HOST_ROOT / 'work/alaya_world_official_repro_20260918'
ROUTES = {f'atomic_{a}': [a] * 16 for a in ('W', 'S', 'A', 'D', 'YL', 'YR')}
ROUTES['G4_003_A'] = ['YL', 'A', 'A', 'D', 'W', 'W', 'YL', 'S', 'W', 'W', 'W', 'A', 'A', 'A']
PROFILE = {
    'id': 'alaya-native8-official-contract-v2-2x-20260919',
    'status': 'OFFICIAL_CONTRACT_ALIGNED_AMPLITUDE_ACCEPTANCE_PENDING',
    'frames_per_unit': 8, 'translation_per_unit': 1.27,
    'lateral_translation_per_unit': 0.0846, 'yaw_degrees_per_unit': 2.0,
    'full_dose': 16, 'inspection_doses': [4, 8, 12, 16],
    'dose12_rounds': 3, 'dose12_output_frames': 97,
    'dose16_rounds': 4, 'dose16_output_frames': 129,
    'rationale': 'same official native8 converter and 8-frame sampling; amplitudes are a documented 2x acceptance candidate to clear short-horizon motion gates',
    'cross_model_metric_unit_claim': False,
}
# Explicit per-model values; no integration function may read PROFILE globally.
REFERENCE_PROFILES = {
    'lyra-2.0': {'frames_per_unit': 10, 'translation_per_unit': 0.003125,
                 'lateral_translation_per_unit': 0.003125, 'yaw_degrees_per_unit': 0.25},
    'lingbot-world-v2': {'frames_per_unit': 4, 'translation_per_unit': 0.1,
                       'lateral_translation_per_unit': 0.1, 'yaw_degrees_per_unit': 3.0},
}


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def load(path: Path):
    return json.loads(path.read_text())


def integrate(actions, frames_per_unit: int, translation: float, yaw_degrees: float,
              *, lateral_translation: float | None = None) -> np.ndarray:
    """Camera-local post-multiplied SE(3), with independent model parameters."""
    lateral = translation if lateral_translation is None else lateral_translation
    n = int(frames_per_unit)
    if n < 1 or min(translation, lateral, yaw_degrees) <= 0:
        raise ValueError('positive amplitudes and frames_per_unit required')
    pose = np.eye(4, dtype=np.float64)
    poses = [pose.copy()]
    for action in actions:
        delta = np.eye(4, dtype=np.float64)
        delta[:3, 3] = {'W': (0, 0, translation/n), 'S': (0, 0, -translation/n),
                       'A': (-lateral/n, 0, 0), 'D': (lateral/n, 0, 0),
                       'YL': (0, 0, 0), 'YR': (0, 0, 0)}[action]
        if action in ('YL', 'YR'):
            angle = math.radians(yaw_degrees / n) * (-1 if action == 'YL' else 1)
            c, s = math.cos(angle), math.sin(angle)
            delta[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        for _ in range(n):
            pose = pose @ delta
            poses.append(pose.copy())
    return np.asarray(poses, dtype=np.float32)


def profile_poses(actions, profile):
    return integrate(actions, profile['frames_per_unit'], profile['translation_per_unit'],
                     profile['yaw_degrees_per_unit'], lateral_translation=profile['lateral_translation_per_unit'])


def camera_for_alaya(actions, profile=PROFILE):
    controls = profile_poses(actions, profile)
    rounds = math.ceil(len(actions) / 4)
    tail = 32 * rounds - 8 * len(actions)
    cameras = np.concatenate([np.repeat(np.eye(4, dtype=np.float32)[None], 128, axis=0),
                              controls, np.repeat(controls[-1:], 8 + tail, axis=0)])
    assert len(cameras) == 137 + 32 * rounds
    return cameras, rounds, tail


def image_transform(width: int, height: int, target_width=960, target_height=544):
    """Cover resize then center crop; actual rounded x/y resize ratios matter."""
    scale = max(target_width/width, target_height/height)
    resized = (round(width*scale), round(height*scale))
    left = int(round((resized[0] - target_width)/2.0))
    top = int(round((resized[1] - target_height)/2.0))
    transform = np.array([[resized[0]/width, 0, -left],
                          [0, resized[1]/height, -top], [0, 0, 1]], dtype=np.float64)
    return transform, resized, (left, top)


def transformed_intrinsic(k_pixel, width, height, target_width=960, target_height=544):
    k = np.asarray(k_pixel, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('K must be finite 3x3 with homogeneous row [0,0,1]')
    transform, resized, crop = image_transform(width, height, target_width, target_height)
    k_output = transform @ k  # scales only image-coordinate rows, never the homogeneous row
    normalized = np.diag([1/target_width, 1/target_height, 1]) @ k_output
    return normalized.astype(np.float32), {'source_size_wh': [width, height],
        'resized_size_wh': list(resized), 'crop_left_top': list(crop),
        'output_size_wh': [target_width, target_height], 'pixel_transform': transform.tolist(),
        'source_k_pixel': k.tolist(), 'output_k_pixel': k_output.tolist()}


def input_files(prefix: str | Path):
    return [Path(str(prefix) + suffix) for suffix in ('_image.png', '_camera.pt', '_prompt.txt')]


def file_records(prefix):
    return [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in input_files(prefix)]
