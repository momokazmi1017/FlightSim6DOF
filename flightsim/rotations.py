"""Quaternion utilities. Quaternions are [w, x, y, z] and rotate body vectors
into the inertial frame: v_inertial = R(q) v_body.

Attitude is propagated with q_dot = 1/2 q (x) [0, omega_body]; unlike Euler
angles, this has no singularity when the rocket points straight up.
"""
import math

import numpy as np


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_dcm(q):
    """Rotation matrix (body -> inertial) of a unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dcm_to_quat(R):
    """Unit quaternion of a rotation matrix (Shepperd's method, numerically robust)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * math.sqrt(tr + 1)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def attitude_from_axis(x_body_inertial):
    """Quaternion whose body x axis points along the given inertial direction, with
    body y horizontal (a rocket on a rail; roll is arbitrary)."""
    x = np.asarray(x_body_inertial, float)
    x = x / np.linalg.norm(x)
    up = np.array([0.0, 0.0, 1.0])
    y = np.cross(up, x)
    if np.linalg.norm(y) < 1e-9:           # pointing straight up
        y = np.array([0.0, 1.0, 0.0])
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return dcm_to_quat(np.column_stack([x, y, z]))
