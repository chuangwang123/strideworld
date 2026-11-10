"""Small, dependency-light SE(2) primitives used by all StrideWorld tasks.

The matrix convention is the one frozen by the implementation specification:

    [[cos(theta),  sin(theta), tx],
     [-sin(theta), cos(theta), tz],
     [0,           0,          1 ]]

``theta`` is radians internally.  Translation is in centimetres.  No metric
in this module combines centimetres and degrees.
"""

from __future__ import annotations

import math

import numpy as np

try:  # scipy is already a dependency of the existing PnP refinement path.
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover - a clear error is raised when needed.
    Rotation = None  # type: ignore[assignment]


EPS = 1e-9
SMALL_ANGLE = 1e-6

ACTION_DIRECTIONS: dict[str, np.ndarray] = {
    "W": np.array([0.0, 1.0, 0.0]),
    "S": np.array([0.0, -1.0, 0.0]),
    "A": np.array([-1.0, 0.0, 0.0]),
    "D": np.array([1.0, 0.0, 0.0]),
    "YL": np.array([0.0, 0.0, -1.0]),
    "YR": np.array([0.0, 0.0, 1.0]),
}
_ACTION_ALIASES = {
    "w": "W", "forward": "W", "move_forward": "W",
    "s": "S", "backward": "S", "move_backward": "S",
    "a": "A", "left": "A", "move_left": "A",
    "d": "D", "right": "D", "move_right": "D",
    "yl": "YL", "yaw_left": "YL", "turn_left": "YL",
    "yr": "YR", "yaw_right": "YR", "turn_right": "YR",
}


def _validate_se2(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"SE(2) matrix must be finite [3,3], got {matrix.shape}")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("invalid SE(2) homogeneous row")
    rotation = matrix[:2, :2]
    if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-8):
        raise ValueError("SE(2) rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("SE(2) rotation determinant is not +1")
    return matrix


def wrap_angle_rad(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi, abs_tol=1e-15) else wrapped


def se2_matrix(tx: float, tz: float, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s, tx], [-s, c, tz], [0.0, 0.0, 1.0]], dtype=np.float64)


def _v_matrix(theta: float) -> np.ndarray:
    if abs(theta) < SMALL_ANGLE:
        # The first-order limit is sufficient at the frozen numerical scale;
        # retain a few terms to make round trips stable at 1e-9.
        theta2 = theta * theta
        a = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        b = theta / 2.0 - theta * theta2 / 24.0 + theta * theta2 * theta2 / 720.0
        return np.array([[a, b], [-b, a]], dtype=np.float64)
    return np.array([
        [math.sin(theta) / theta, (1.0 - math.cos(theta)) / theta],
        [-(1.0 - math.cos(theta)) / theta, math.sin(theta) / theta],
    ], dtype=np.float64)


def log_se2(matrix: np.ndarray) -> np.ndarray:
    """Return ``[rho_x, rho_z, theta]`` from one SE(2) matrix."""
    matrix = _validate_se2(matrix)
    theta = wrap_angle_rad(math.atan2(float(matrix[0, 1]), float(matrix[0, 0])))
    rho = np.linalg.solve(_v_matrix(theta), matrix[:2, 2])
    return np.array([rho[0], rho[1], theta], dtype=np.float64)


def exp_se2(twist: np.ndarray) -> np.ndarray:
    """Return the SE(2) exponential of ``[rho_x, rho_z, theta]``."""
    twist = np.asarray(twist, dtype=np.float64)
    if twist.shape != (3,) or not np.isfinite(twist).all():
        raise ValueError("SE(2) twist must be one finite [3] vector")
    theta = float(twist[2])
    translation = _v_matrix(theta) @ twist[:2]
    return se2_matrix(float(translation[0]), float(translation[1]), theta)


def inverse_se2(matrix: np.ndarray) -> np.ndarray:
    matrix = _validate_se2(matrix)
    rotation = matrix[:2, :2]
    result = np.eye(3, dtype=np.float64)
    result[:2, :2] = rotation.T
    result[:2, 2] = -rotation.T @ matrix[:2, 2]
    return result


def compose_se2(*matrices: np.ndarray) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    for matrix in matrices:
        result = result @ _validate_se2(matrix)
    return result


def project_se3_to_se2(transform: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Project a camera-relative SE(3) transform and return its leakage.

    The yaw extraction and residual rotation exactly follow §0.3 of the final
    specification.  ``ty`` is in cm; pitch and roll are returned in degrees.
    """
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("SE(3) transform must be finite [4,4]")
    rotation = transform[:3, :3]
    theta = math.atan2(float(rotation[0, 2]), float(rotation[0, 0]))
    c, s = math.cos(theta), math.sin(theta)
    yaw_rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    residual = yaw_rotation.T @ rotation
    if Rotation is None:  # pragma: no cover
        raise RuntimeError("SE(3) projection requires scipy")
    residual_rotvec = Rotation.from_matrix(residual).as_rotvec()
    planar = se2_matrix(
        float(transform[0, 3]), float(transform[2, 3]), float(theta)
    )
    leakage = {
        "ty_cm": float(transform[1, 3]),
        "pitch_deg": float(np.degrees(residual_rotvec[0])),
        "roll_deg": float(np.degrees(residual_rotvec[2])),
    }
    return planar, leakage


def distance_se2(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Return separate planar translation (cm) and yaw (deg) distances."""
    delta = compose_se2(inverse_se2(first), second)
    twist = log_se2(delta)
    return {
        "translation_cm": float(np.linalg.norm(delta[:2, 2])),
        "yaw_deg": abs(math.degrees(wrap_angle_rad(float(twist[2])))),
    }


def se2_power(matrix: np.ndarray, power: int | float) -> np.ndarray:
    """Compute a positive integer power through the group logarithm."""
    if isinstance(power, bool) or float(power) < 0 or not float(power).is_integer():
        raise ValueError("SE(2) power must be a non-negative integer")
    return exp_se2(log_se2(matrix) * int(power))


def action_name(action: str) -> str:
    token = str(action).strip()
    return _ACTION_ALIASES.get(token.lower(), token.upper())


def action_direction(action: str) -> np.ndarray:
    name = action_name(action)
    if name not in ACTION_DIRECTIONS:
        raise KeyError(f"unknown StrideWorld action: {action}")
    return ACTION_DIRECTIONS[name].copy()


def action_scalar(twist: np.ndarray, action: str) -> float:
    twist = np.asarray(twist, dtype=np.float64).reshape(3)
    return float(np.dot(twist, action_direction(action)))


__all__ = [
    "ACTION_DIRECTIONS", "action_direction", "action_scalar", "action_name",
    "compose_se2", "distance_se2", "exp_se2", "inverse_se2", "log_se2",
    "project_se3_to_se2", "se2_matrix", "se2_power", "wrap_angle_rad",
]
