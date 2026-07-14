import numpy as np

from .backend import get_backend


def _ensure_3xN(values, backend):
    arr = backend.asarray(values)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError('Vector must have length 3.')
        arr = backend.reshape(arr, (3, 1))
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr
    raise ValueError('Vector must have shape (3,) or (3,N).')


def vector(x=0.0, y=0.0, z=0.0, backend=None):
    backend = backend or get_backend('numpy')
    return backend.reshape(backend.asarray([x, y, z], dtype=float), (3, 1))


def zeros(shape, backend=None):
    backend = backend or get_backend('numpy')
    return backend.zeros(shape)


def cross(a, b, backend=None):
    backend = backend or get_backend('numpy')
    a = _ensure_3xN(a, backend)
    b = _ensure_3xN(b, backend)
    x = a[1, :] * b[2, :] - a[2, :] * b[1, :]
    y = a[2, :] * b[0, :] - a[0, :] * b[2, :]
    z = a[0, :] * b[1, :] - a[1, :] * b[0, :]
    return backend.stack([x, y, z], axis=0)


def dot(a, b, backend=None):
    backend = backend or get_backend('numpy')
    a = _ensure_3xN(a, backend)
    b = _ensure_3xN(b, backend)
    return backend.sum(a * b, axis=0)


def norm(vectors, axis=None, keepdims=False, backend=None):
    backend = backend or get_backend('numpy')
    return backend.norm(vectors, axis=axis, keepdims=keepdims)


def euler_to_quaternion(roll, pitch, yaw, backend=None):
    backend = backend or get_backend('numpy')
    half_roll = backend.asarray(roll, dtype=float) * 0.5
    half_pitch = backend.asarray(pitch, dtype=float) * 0.5
    half_yaw = backend.asarray(yaw, dtype=float) * 0.5
    cr = backend.cos(half_roll)
    sr = backend.sin(half_roll)
    cp = backend.cos(half_pitch)
    sp = backend.sin(half_pitch)
    cy = backend.cos(half_yaw)
    sy = backend.sin(half_yaw)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return backend.stack([w, x, y, z], axis=0)


def quaternion_to_euler(q, backend=None):
    backend = backend or get_backend('numpy')
    q = backend.asarray(q)
    if q.ndim == 1:
        q = backend.reshape(q, (4, 1))
    w, x, y, z = q[0:1, :], q[1:2, :], q[2:3, :], q[3:4, :]
    roll = backend.arctan2(2 * (y * z + w * x), 2 * (w * w - 0.5 + z * z))
    pitch = -backend.arcsin(backend.clip(2 * (x * z - w * y), backend.asarray(-1.0, dtype=float), backend.asarray(1.0, dtype=float)))
    yaw = backend.arctan2(2 * (x * y + w * z), 2 * (w * w - 0.5 + x * x))
    return backend.concatenate([roll, pitch, yaw], axis=0)


def quaternion_to_axis_angle(q, backend=None):
    backend = backend or get_backend('numpy')
    q = backend.asarray(q)
    if q.ndim == 1:
        q = backend.reshape(q, (4, 1))
    w = q[0:1, :]
    angle = 2 * backend.arccos(backend.clip(w, backend.asarray(-1.0, dtype=float), backend.asarray(1.0, dtype=float)))
    sin_half_angle = backend.sqrt(backend.clip(1.0 - w * w, backend.asarray(0.0, dtype=float), backend.asarray(1.0, dtype=float)))
    default_axis = backend.reshape(backend.asarray([1.0, 0.0, 0.0], dtype=float), (3, 1))
    axis = backend.where(
        sin_half_angle == 0.0,
        backend.broadcast_to(default_axis, (3, q.shape[1])),
        q[1:4, :] / sin_half_angle,
    )
    return axis, angle


def fix_continuity(quaternions):
    q = np.asarray(quaternions)
    if q.ndim != 2 or (q.shape[0] != 4 and q.shape[1] != 4):
        raise ValueError('Quaternions input must be shape (4,N) or (N,4).')

    if q.shape[1] == 4:
        q = q.copy()
        for index in range(1, q.shape[0]):
            prev = q[index - 1]
            current = q[index]
            a = np.linalg.norm(current - prev)
            b = np.linalg.norm(-current - prev)
            if b < a:
                q[index] = -current
        return q

    q = q.copy()
    for index in range(1, q.shape[1]):
        prev = q[:, index - 1]
        current = q[:, index]
        a = np.linalg.norm(current - prev)
        b = np.linalg.norm(-current - prev)
        if b < a:
            q[:, index] = -current
    return q


class Quaternion:
    def __init__(self, values=None, backend=None):
        self.backend = backend or get_backend('numpy')
        if values is None:
            values = [1.0, 0.0, 0.0, 0.0]
        q = self.backend.asarray(values, dtype=float)
        if q.ndim == 1:
            q = self.backend.reshape(q, (4, 1))
        if q.shape[0] != 4:
            raise ValueError('Quaternion must have shape (4,) or (4,N).')
        self._quat = q

    @property
    def values(self):
        return self._quat

    def normalized(self):
        magnitude = self.backend.norm(self._quat, axis=0, keepdims=True)
        magnitude = self.backend.where(magnitude == 0.0, self.backend.ones(magnitude.shape), magnitude)
        return Quaternion(self._quat / magnitude, backend=self.backend)

    def conjugate(self):
        q = self._quat
        return Quaternion(self.backend.concatenate([q[0:1, :], -q[1:4, :]], axis=0), backend=self.backend)

    def __mul__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError('Can only multiply by another Quaternion.')
        a = self._quat
        b = other._quat
        w1, x1, y1, z1 = a[0:1, :], a[1:2, :], a[2:3, :], a[3:4, :]
        w2, x2, y2, z2 = b[0:1, :], b[1:2, :], b[2:3, :], b[3:4, :]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return Quaternion(self.backend.concatenate([w, x, y, z], axis=0), backend=self.backend)

    def rotate_vector(self, vectors):
        backend = self.backend
        v = _ensure_3xN(vectors, backend)
        q = self._quat
        qv = q[1:4, :]
        t = cross(qv, v, backend) * 2.0
        term = q[0:1, :] * t
        return v + term + cross(qv, t, backend)

    @staticmethod
    def from_euler(roll, pitch, yaw, backend=None):
        backend = backend or get_backend('numpy')
        q = euler_to_quaternion(roll, pitch, yaw, backend=backend)
        return Quaternion(q, backend=backend)

    @staticmethod
    def from_rotation_vector(rotvec, backend=None):
        backend = backend or get_backend('numpy')
        rv = _ensure_3xN(rotvec, backend)
        theta = backend.norm(rv, axis=0, keepdims=True)
        half = theta * 0.5
        safe_theta = backend.where(theta == 0.0, backend.ones(theta.shape), theta)
        axis = rv / safe_theta
        w = backend.cos(half)
        s = backend.sin(half)
        xyz = axis * s
        q = backend.concatenate([w, xyz], axis=0)
        q = backend.where(backend.equal(theta, 0.0), backend.concatenate([backend.ones((1, 1)), backend.zeros((3, 1))], axis=0), q)
        return Quaternion(q, backend=backend)

    def to_euler(self):
        return quaternion_to_euler(self._quat, backend=self.backend)

    def to_axis_angle(self):
        return quaternion_to_axis_angle(self._quat, backend=self.backend)

    def to_rotation_matrix(self):
        q = self._quat
        w, x, y, z = q[0:1, :], q[1:2, :], q[2:3, :], q[3:4, :]
        m00 = 1 - 2 * (y * y + z * z)
        m01 = 2 * (x * y - w * z)
        m02 = 2 * (x * z + w * y)
        m10 = 2 * (x * y + w * z)
        m11 = 1 - 2 * (x * x + z * z)
        m12 = 2 * (y * z - w * x)
        m20 = 2 * (x * z - w * y)
        m21 = 2 * (y * z + w * x)
        m22 = 1 - 2 * (x * x + y * y)
        return self.backend.stack([
            self.backend.stack([m00, m01, m02], axis=0),
            self.backend.stack([m10, m11, m12], axis=0),
            self.backend.stack([m20, m21, m22], axis=0),
        ], axis=1)

    def __repr__(self):
        return f'Quaternion(values={self._quat}, backend={self.backend.name})'
