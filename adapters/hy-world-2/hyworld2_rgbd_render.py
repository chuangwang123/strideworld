"""Render one frozen I0-predicted point cloud using official point_rendering.

Geometry is normalized once per input image by its frozen valid-depth median.
No target views, benchmark depth/GT, trajectory-dependent scale, or history.
Torch and the native renderer are imported only by the rendering entry point.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class GeometryContractError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GeometryContractError(message)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def rgb_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_intrinsics(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    require(K.shape == (3, 3) and np.isfinite(K).all(), 'K_cv must be finite 3x3')
    require(np.allclose(K[2], [0, 0, 1], atol=1e-8, rtol=0), 'K_cv must use ordinary positive-Z perspective convention')
    require(K[0, 0] > 0 and K[1, 1] > 0 and abs(np.linalg.det(K)) > 1e-12, 'K_cv is singular or has non-positive focal length')
    return K


def backproject_frozen_depth(image: np.ndarray, K_cv: np.ndarray, depth: np.ndarray,
                            valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pure numpy positive-Z backprojection on integer OpenCV pixel centers.

    Returns world points in the initial camera frame. Their fixed valid median
    depth is 1. The exact same points and scale are used for all trajectories.
    """
    image = np.asarray(image)
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, 'condition must be RGB uint8 HWC')
    K = validate_intrinsics(K_cv)
    depth, mask = np.asarray(depth), np.asarray(valid_mask)
    require(depth.shape == image.shape[:2] and depth.dtype.kind == 'f', 'depth must be floating positive-Z HxW at model image resolution')
    require(mask.shape == depth.shape and mask.dtype == np.bool_, 'valid_mask must be frozen bool HxW')
    require(bool(mask.any()), 'frozen geometry has no valid depth')
    valid_depth = depth[mask].astype(np.float64)
    require(np.isfinite(valid_depth).all() and bool((valid_depth > 0).all()), 'valid_mask selects nonfinite/nonpositive depth')
    median = float(np.median(valid_depth))
    require(math.isfinite(median) and median > 0, 'valid depth median is invalid')
    # Pixel centers are (u,v)=(column,row), not half-shifted native Plucker rays.
    v, u = np.nonzero(mask)
    pixels = np.stack([u, v, np.ones_like(u)], axis=1).astype(np.float64)
    rays = np.linalg.solve(K, pixels.T).T
    normalized_z = valid_depth / median
    points64 = rays * normalized_z[:, None]
    require(np.isfinite(points64).all() and bool((np.abs(points64) <= np.finfo(np.float32).max).all()), 'normalized point cloud is not finite float32')
    points = np.ascontiguousarray(points64, dtype=np.float32)
    colors = np.ascontiguousarray(image[mask].astype(np.float32) / 127.5 - 1.0)
    metadata = {'valid_depth_median_raw': median, 'normalization_multiplier': 1.0/median,
        'normalization_reference': 'one_frozen_valid_mask_median_per_I0', 'normalized_valid_depth_median': 1.0,
        'raw_valid_depth_min': float(valid_depth.min()), 'raw_valid_depth_max': float(valid_depth.max()),
        'normalized_valid_depth_min': float(normalized_z.min()), 'normalized_valid_depth_max': float(normalized_z.max()),
        'point_count': len(points), 'source_valid_pixel_fraction': float(mask.mean()),
        'pointcloud_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
        'pointcloud_color_sha256': hashlib.sha256(colors.tobytes()).hexdigest(),
        'pointcloud_coordinate_frame': 'I0_OpenCV_right_handed_x_right_y_down_z_forward',
        'backprojection_pixel_centers': 'integer_column_row_using_K_cv',
        'depth_model_metric_scale_claim': False,
        'trajectory_dependent_normalization': False, 'pointcloud_updated_after_I0': False}
    return points, colors, metadata


def load_frozen_geometry(image: np.ndarray, K_cv: np.ndarray, geometry_case_dir: str | Path,
                         profile: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Verify cache identity/hash/resolution and construct its fixed point cloud."""
    directory = Path(geometry_case_dir).resolve()
    metadata_path, depth_path = directory/'metadata.json', directory/'depth.npz'
    metadata = json.loads(metadata_path.read_text())
    require(isinstance(metadata, dict), 'geometry metadata must be a JSON object')
    expected_metadata_sha = profile.get('expected_geometry_metadata_sha256')
    if expected_metadata_sha is not None:
        require(is_sha256(expected_metadata_sha) and file_sha256(metadata_path) == expected_metadata_sha,
                'geometry metadata differs from the generation-bound cache version')
    for field in ('condition_image_sha256', 'condition_rgb_sha256', 'source_image_sha256', 'checkpoint_sha256', 'depth_npz_sha256'):
        require(is_sha256(metadata.get(field)), f'geometry metadata lacks a valid {field}')
    require(metadata.get('geometry_source') == 'predicted_from_condition_RGB_only', 'geometry must be predicted from this I0 RGB only')
    require(metadata.get('depth_type') == 'positive_z_camera', 'geometry must be camera Z-depth, not range/disparity')
    require(metadata.get('depth_scale_applied') == 1.0 and metadata.get('normalization_applied') is False,
            'geometry cache must contain raw unnormalized predicted depth')
    require(profile.get('camera_normalize') is False, 'render and Plucker controls require camera_normalize=False')
    require(is_sha256(profile.get('expected_source_image_sha256')), 'profile.expected_source_image_sha256 must bind the canonical source RGB')
    require(metadata['source_image_sha256'] == profile['expected_source_image_sha256'], 'geometry/source RGB identity mismatch')
    expected_checkpoint = profile.get('geometry_checkpoint_sha256')
    if expected_checkpoint is not None:
        require(is_sha256(expected_checkpoint) and expected_checkpoint == metadata['checkpoint_sha256'], 'geometry depth-model checkpoint mismatch')
    image = np.asarray(image)
    require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, 'condition must be RGB uint8 HWC')
    H, W = image.shape[:2]
    require((H, W) == (metadata.get('height'), metadata.get('width')), 'geometry and condition image dimensions differ')
    require((H, W) == (profile.get('height'), profile.get('width')), 'geometry and generation profile dimensions differ')
    require(rgb_sha256(image) == metadata['condition_rgb_sha256'], 'geometry condition RGB array SHA mismatch')
    K = validate_intrinsics(K_cv)
    metadata_K = validate_intrinsics(metadata.get('K_cv'))
    require(np.allclose(K, metadata_K, rtol=1e-6, atol=1e-5), 'geometry and render K_cv mismatch')
    # The official PerspectiveCameras construction consumes fx/fy/cx/cy only.
    require(abs(K[0, 1]) <= 1e-8 and abs(K[1, 0]) <= 1e-8,
            'native point_rendering does not support skewed camera intrinsics')
    # Retain the exact cache PNG as file-level evidence, separately from RGB.
    condition_value = metadata.get('condition_image_path', metadata.get('condition_image_file', 'conditioning_frame.png'))
    condition_path = Path(condition_value)
    if condition_path.is_absolute():
        require(condition_path.name == 'conditioning_frame.png', 'absolute cache condition path must name conditioning_frame.png')
        # A cache snapshot can retain its original absolute path in metadata.
        # Read only the current snapshot's pinned PNG, never that original path.
        condition_path = directory/'conditioning_frame.png'
    else:
        condition_path = directory/condition_path
    condition_path = condition_path.resolve()
    require(condition_path.is_relative_to(directory), 'cache condition image must be contained in geometry_case_dir')
    require(file_sha256(condition_path) == metadata['condition_image_sha256'], 'geometry condition PNG file SHA mismatch')
    from PIL import Image
    condition_rgb = np.asarray(Image.open(condition_path).convert('RGB'))
    require(condition_rgb.shape == image.shape and np.array_equal(condition_rgb, image), 'geometry condition PNG pixels differ from current I0')
    require(file_sha256(depth_path) == metadata['depth_npz_sha256'], 'geometry depth.npz SHA mismatch')
    with np.load(depth_path, allow_pickle=False) as archive:
        require('depth' in archive and 'valid_mask' in archive, 'depth.npz must contain depth and valid_mask')
        points, colors, scale = backproject_frozen_depth(image, K, archive['depth'], archive['valid_mask'])
    if 'valid_depth_median_raw' in metadata:
        require(math.isclose(scale['valid_depth_median_raw'], float(metadata['valid_depth_median_raw']), rel_tol=1e-6, abs_tol=1e-8),
                'geometry frozen valid-depth median mismatch')
    output = {'schema_version': 'worldstereo-I0-predicted-depth-render-v1',
        'geometry_case_dir': str(directory), 'geometry_metadata_sha256': file_sha256(metadata_path),
        'depth_npz_sha256': metadata['depth_npz_sha256'],
        'condition_image_sha256': metadata['condition_image_sha256'], 'condition_rgb_sha256': metadata['condition_rgb_sha256'],
        'condition_image_metadata_original_path': str(condition_value),
        'condition_image_resolved_snapshot_path': str(condition_path),
        'source_image_sha256': metadata['source_image_sha256'], 'geometry_checkpoint_sha256': metadata['checkpoint_sha256'],
        'geometry_model_id': metadata.get('model_id'), 'geometry_model_revision': metadata.get('model_revision'),
        'height': H, 'width': W, 'K_cv': K.tolist(), 'geometry_source': metadata['geometry_source'],
        'depth_type': metadata['depth_type'], 'cache_depth_scale_applied': metadata['depth_scale_applied'],
        'cache_normalization_applied': False, 'GT_depth_input': False, 'source_world_pose_input': False,
        'generated_history_input': False, 'target_or_future_RGB_input': False, **scale}
    return points, colors, output


def build_render_inputs(image_uint8_HWC, K_cv_np, w2c_torch, geometry_case_dir, device, profile):
    """Return rv[1,3,F,H,W], rm[1,1,F,H,W], JSON-safe audit metadata.

    The caller has placed official HY-World-2.0/hyworld2/worldgen on sys.path.
    Native point_rendering is invoked in chunks of four. Mask 1 means unknown.
    Frame zero is explicitly supplied I0 RGB/mask0 as *model conditioning*; this
    function does not read or alter any generated video frame.
    """
    import torch
    from src.pointcloud import point_rendering

    points, colors, metadata = load_frozen_geometry(image_uint8_HWC, K_cv_np, geometry_case_dir, profile)
    require(isinstance(w2c_torch, torch.Tensor) and w2c_torch.ndim == 3
            and tuple(w2c_torch.shape[1:]) == (4, 4) and w2c_torch.shape[0] > 0,
            'w2c_torch must have shape [F,4,4]')
    w2c = w2c_torch.detach().to(device=device, dtype=torch.float32).contiguous()
    require(bool(torch.isfinite(w2c).all().item()), 'render camera contains nonfinite values')
    eye = torch.eye(4, device=device, dtype=torch.float32)
    require(torch.allclose(w2c[0], eye, atol=1e-5, rtol=1e-5), 'initial render camera must be I0 identity')
    require(torch.allclose(w2c[:, 3], eye[3].expand(w2c.shape[0], -1), atol=1e-6, rtol=0), 'render camera homogeneous row is invalid')
    R = w2c[:, :3, :3]
    require(torch.allclose(R@R.transpose(1, 2), eye[:3, :3].expand(w2c.shape[0], -1, -1), atol=1e-4, rtol=1e-4)
            and torch.allclose(torch.linalg.det(R), torch.ones(w2c.shape[0], device=device), atol=1e-4, rtol=1e-4),
            'render camera rotations must be proper orthonormal matrices')
    radius = float(profile.get('render_radius', 0.008))
    points_per_pixel = profile.get('render_points_per_pixel', profile.get('points_per_pixel', 8))
    require(profile.get('render_chunk_frames', profile.get('render_chunk_size', 4)) == 4,
            'native render chunks are frozen at four frames')
    require(math.isfinite(radius) and radius > 0, 'render_radius must be finite positive')
    require(isinstance(points_per_pixel, int) and not isinstance(points_per_pixel, bool) and points_per_pixel > 0,
            'render_points_per_pixel must be a positive integer')
    F, H, W = int(w2c.shape[0]), metadata['height'], metadata['width']
    K = torch.as_tensor(np.asarray(K_cv_np).copy(), device=device, dtype=torch.float32).unsqueeze(0).repeat(F, 1, 1)
    p = torch.from_numpy(points).to(device=device)
    c = torch.from_numpy(colors).to(device=device)
    rv = torch.empty((1, 3, F, H, W), device=device, dtype=torch.float32)
    rm = torch.empty((1, 1, F, H, W), device=device, dtype=torch.float32)
    native_coverage = []
    with torch.no_grad():
        for start in range(0, F, 4):
            end = min(start+4, F)
            rgb, mask = point_rendering(K=K[start:end], w2cs=w2c[start:end], points=p, colors=c,
                device=device, h=H, w=W, background_color=[0, 0, 0], render_radius=radius,
                points_per_pixel=points_per_pixel, return_depth=False)
            require(tuple(rgb.shape) == (end-start, 3, H, W) and tuple(mask.shape) == (end-start, 1, H, W),
                    'native point renderer returned unexpected shapes')
            require(bool(torch.isfinite(rgb).all().item()) and bool(torch.isfinite(mask).all().item()), 'native renderer returned nonfinite values')
            require(bool(((mask == 0) | (mask == 1)).all().item()), 'native renderer mask must be binary 1=unknown')
            require(float(rgb.min()) >= -1.0001 and float(rgb.max()) <= 1.0001, 'native rendered RGB is outside normalized [-1,1]')
            # Native background should already be zero; enforce/verify exact
            # unknown-pixel semantics while preserving all observed colors.
            unknown = mask.bool().expand_as(rgb)
            require(bool((rgb[unknown].abs() <= 1e-6).all().item()), 'native unknown render pixels must equal zero background')
            rv[0, :, start:end] = rgb.clamp(-1, 1).permute(1, 0, 2, 3)
            rm[0, :, start:end] = mask.permute(1, 0, 2, 3)
            native_coverage.extend((1-mask).mean(dim=(1, 2, 3)).detach().cpu().tolist())
        supplied = torch.from_numpy(np.ascontiguousarray(image_uint8_HWC)).to(device=device, dtype=torch.float32)
        rv[0, :, 0] = (supplied/127.5-1).permute(2, 0, 1)
        rm[0, :, 0] = 0
    metadata.update({'frame_count': F, 'render_backend': 'official src.pointcloud.point_rendering',
        'render_chunk_frames': 4, 'render_radius': radius, 'render_points_per_pixel': points_per_pixel,
        'native_render_valid_fraction_per_frame': native_coverage,
        'render_valid_fraction_per_frame': [1.0, *native_coverage[1:]],
        'unknown_mask_value': 1, 'unknown_render_RGB_value': 0, 'render_tensor_dtype': 'float32',
        'frame_zero_policy': 'supplied_condition_RGB_mask0; conditioning only, generated video unchanged',
        'camera_normalize': False,
        'render_camera_w2c_sha256': hashlib.sha256(w2c.detach().cpu().numpy().tobytes()).hexdigest()})
    return rv.contiguous(), rm.contiguous(), metadata

