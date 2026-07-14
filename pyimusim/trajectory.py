import numpy as np
from .math import vector, zeros, Quaternion, cross, dot
from .backend import get_backend


class AbstractTrajectory:
    @property
    def start_time(self):
        return 0.0

    @property
    def end_time(self):
        return np.inf

    def to_backend(self, backend):
        raise NotImplementedError('Trajectory subclasses must implement to_backend.')


class PositionTrajectory(AbstractTrajectory):
    def position(self, t):
        raise NotImplementedError

    def velocity(self, t):
        raise NotImplementedError

    def acceleration(self, t):
        raise NotImplementedError


class RotationTrajectory(AbstractTrajectory):
    def rotation(self, t):
        raise NotImplementedError

    def rotational_velocity(self, t):
        raise NotImplementedError

    def rotational_acceleration(self, t):
        raise NotImplementedError


class StaticTrajectory(PositionTrajectory, RotationTrajectory):
    def __init__(self, position=None, rotation=None, backend=None):
        self.backend = backend or get_backend('numpy')
        self.position0 = vector(0.0, 0.0, 0.0, backend=self.backend) if position is None else position
        self.rotation0 = Quaternion(backend=self.backend) if rotation is None else rotation

    def to_backend(self, backend):
        return StaticTrajectory(
            position=backend.asarray(self.position0),
            rotation=Quaternion(self.rotation0.values, backend=backend),
            backend=backend,
        )

    def position(self, t):
        if hasattr(t, 'shape') and len(t.shape) > 0:
            return self.backend.broadcast_to(self.position0, (3, t.shape[0]))
        return self.position0

    def velocity(self, t):
        return zeros((3, 1), backend=self.backend) if not hasattr(t, 'shape') or len(t.shape) == 0 else zeros((3, t.shape[0]), backend=self.backend)

    def acceleration(self, t):
        return self.velocity(t)

    def rotation(self, t):
        return self.rotation0

    def rotational_velocity(self, t):
        return self.velocity(t)

    def rotational_acceleration(self, t):
        return self.velocity(t)


class ConstantVelocityTrajectory(PositionTrajectory, RotationTrajectory):
    def __init__(self, position, velocity, rotation=None, rotational_velocity=None, backend=None):
        self.backend = backend or get_backend('numpy')
        self.position0 = position
        self.velocity0 = velocity
        self.rotation0 = Quaternion(backend=self.backend) if rotation is None else rotation
        self.rotational_velocity0 = vector(0.0, 0.0, 0.0, backend=self.backend) if rotational_velocity is None else rotational_velocity

    def to_backend(self, backend):
        return ConstantVelocityTrajectory(
            position=backend.asarray(self.position0),
            velocity=backend.asarray(self.velocity0),
            rotation=Quaternion(self.rotation0.values, backend=backend),
            rotational_velocity=backend.asarray(self.rotational_velocity0),
            backend=backend,
        )

    def position(self, t):
        t = self.backend.asarray(t)
        if t.ndim == 0:
            return self.position0 + self.velocity0 * t
        return self.position0 + self.velocity0 * t.reshape(1, -1)

    def velocity(self, t):
        if hasattr(t, 'shape') and len(t.shape) > 0:
            return self.backend.broadcast_to(self.velocity0, (3, t.shape[0]))
        return self.velocity0

    def acceleration(self, t):
        return self.velocity(t) * 0.0

    def rotation(self, t):
        t = self.backend.asarray(t)
        if t.ndim == 0:
            t = t.reshape(1)
        delta = self.rotational_velocity0 * t.reshape(1, -1)
        delta_q = Quaternion.from_rotation_vector(delta, backend=self.backend)
        return self.rotation0 * delta_q

    def rotational_velocity(self, t):
        if hasattr(t, 'shape') and len(t.shape) > 0:
            return self.backend.broadcast_to(self.rotational_velocity0, (3, t.shape[0]))
        return self.rotational_velocity0

    def rotational_acceleration(self, t):
        return self.rotational_velocity(t) * 0.0


class OffsetTrajectory(PositionTrajectory, RotationTrajectory):
    def __init__(self, parent, position_offset=None, rotation_offset=None):
        self.parent = parent
        self.backend = parent.backend
        self.position_offset = vector(0.0, 0.0, 0.0, backend=self.backend) if position_offset is None else position_offset
        self.rotation_offset = Quaternion(backend=self.backend) if rotation_offset is None else rotation_offset

    def to_backend(self, backend):
        return OffsetTrajectory(
            self.parent.to_backend(backend),
            position_offset=backend.asarray(self.position_offset),
            rotation_offset=Quaternion(self.rotation_offset.values, backend=backend),
        )

    def position(self, t):
        parent_position = self.parent.position(t)
        parent_rotation = self.parent.rotation(t)
        offset = parent_rotation.rotate_vector(self.position_offset)
        if hasattr(t, 'shape') and len(t.shape) > 0:
            return parent_position + offset
        return parent_position + offset

    def velocity(self, t):
        parent_velocity = self.parent.velocity(t)
        parent_rotation = self.parent.rotation(t)
        offset = parent_rotation.rotate_vector(self.position_offset)
        omega = self.parent.rotational_velocity(t)
        rv = cross(omega, offset, backend=self.backend)
        return parent_velocity + rv

    def acceleration(self, t):
        parent_acceleration = self.parent.acceleration(t)
        parent_rotation = self.parent.rotation(t)
        offset = parent_rotation.rotate_vector(self.position_offset)
        omega = self.parent.rotational_velocity(t)
        alpha = self.parent.rotational_acceleration(t)
        lt = cross(alpha, offset, backend=self.backend)
        lr = offset * dot(offset, omega, backend=self.backend) - offset * self.backend.norm(omega, axis=0, keepdims=True) ** 2
        return parent_acceleration + lt + lr

    def rotation(self, t):
        return self.parent.rotation(t) * self.rotation_offset

    def rotational_velocity(self, t):
        return self.parent.rotational_velocity(t)

    def rotational_acceleration(self, t):
        return self.parent.rotational_acceleration(t)

    @property
    def start_time(self):
        return self.parent.start_time

    @property
    def end_time(self):
        return self.parent.end_time
