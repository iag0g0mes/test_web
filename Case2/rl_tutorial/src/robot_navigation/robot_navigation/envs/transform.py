import numpy as np
import math
from rclpy.time import Time
from rclpy.duration import Duration
import tf2_ros

def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
   
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    return np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    ], dtype=np.float64)

def transform_points(
    tf_buffer: tf2_ros.Buffer,
    points: np.ndarray,         
    source_frame: str,               
    target_frame: str,               
    stamp: Time | None = None,      
    timeout_sec: float = 0.1,
) -> np.ndarray:

    if stamp is None:
        stamp = Time()  # latest

    try:
        T = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            stamp,
            timeout=Duration(seconds=float(timeout_sec)),
        )
    except tf2_ros.TransformException as e:
        raise RuntimeError(f"TF lookup failed {source_frame}->{target_frame}: {e}")

    t = T.transform.translation
    q = T.transform.rotation

    R = _quat_to_rotmat(q.x, q.y, q.z, q.w)    # 3x3
    p = points.astype(np.float64)              # Nx3
    out = (p @ R.T) + np.array([t.x, t.y, t.z],
                            dtype=np.float64)  # Nx3
    return out.astype(points.dtype, copy=False)