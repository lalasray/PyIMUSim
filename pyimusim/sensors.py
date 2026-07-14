import numpy as np

from .platform import Component
from .backend import get_backend
from .math import vector, zeros, cross, dot, Quaternion
from .trajectory import OffsetTrajectory


def _ensure_3xN(values, backend):
    arr = backend.asarray(values)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError('Vector must have length 3.')
        arr = backend.reshape(arr, (3, 1))
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr
    raise ValueError('Vector must have shape (3,) or (3,N).')


class Sensor(Component):
    def __init__(self, platform=None, position_offset=None, rotation_offset=None,
                 backend=None):
        self.backend = backend or get_backend(None)
        self.position_offset = vector(0.0, 0.0, 0.0, backend=self.backend) if position_offset is None else position_offset
        self.rotation_offset = Quaternion(backend=self.backend) if rotation_offset is None else rotation_offset
        super().__init__(platform)

    def _trajectory_change(self):
        if self.platform is None or self.platform.trajectory is None:
            self.trajectory = None
        else:
            self.trajectory = OffsetTrajectory(self.platform.trajectory,
                                               self.position_offset,
                                               self.rotation_offset)

    def _simulation_change(self):
        self.simulation = self.platform.simulation if self.platform is not None else None

    def true_values(self, t):
        raise NotImplementedError

    def sensed_voltages(self, t):
        return self.true_values(t)

    def noise_voltages(self, t):
        if hasattr(t, 'shape') and len(t.shape) > 0:
            shape = (3, t.shape[0])
        else:
            shape = (3, 1)
        return zeros(shape, backend=self.backend)

    def voltages(self, t):
        return self.sensed_voltages(t) + self.noise_voltages(t)


class IdealSensor(Sensor):
    pass


class NoisySensor(Sensor):
    def __init__(self,
                 platform=None,
                 noise_std=0.0,
                 offset=None,
                 sensitivity_error=None,
                 noise_density=None,
                 cross_axis=None,
                 misalignment=0.0,
                 measurement_range=None,
                 deterministic=True,
                 **kwargs):
        super().__init__(platform=platform, **kwargs)
        self.noise_std = noise_std
        self.offset = _ensure_3xN(offset, self.backend) if offset is not None else vector(0.0, 0.0, 0.0, backend=self.backend)
        self.sensitivity_error = _ensure_3xN(sensitivity_error, self.backend) if sensitivity_error is not None else vector(0.0, 0.0, 0.0, backend=self.backend)
        self.noise_density = _ensure_3xN(noise_density, self.backend) if noise_density is not None else vector(0.0, 0.0, 0.0, backend=self.backend)
        self.cross_axis = _ensure_3xN(cross_axis, self.backend) if cross_axis is not None else vector(0.0, 0.0, 0.0, backend=self.backend)
        self.misalignment = misalignment
        self.measurement_range = measurement_range
        self._deterministic = deterministic

    def _sample_interval(self, t):
        if hasattr(t, 'shape') and len(t.shape) > 0 and t.shape[0] > 1:
            dt = t[1] - t[0]
            if self.backend.name == 'torch':
                return float(dt.item())
            return float(dt)
        return 1.0

    def noise_voltages(self, t):
        if not self._deterministic:
            if hasattr(t, 'shape') and len(t.shape) > 0:
                shape = (3, t.shape[0])
            else:
                shape = (3, 1)
            return self.backend.normal(shape, mean=0.0, std=self.noise_std)
        return zeros((3, 1) if not (hasattr(t, 'shape') and len(t.shape) > 0) else (3, t.shape[0]), backend=self.backend)

    def _apply_sensor_errors(self, sensor, t):
        sensor = sensor + self.offset
        sensor = sensor * (1.0 + self.sensitivity_error / 100.0)

        dt = self._sample_interval(t)
        if dt > 0.0 and not self._deterministic:
            scale = self.backend.sqrt(1.0 / dt)
            density_noise = self.backend.normal(sensor.shape, mean=0.0, std=self.noise_density * scale)
            sensor = sensor + density_noise

        if self.cross_axis is not None and self.misalignment != 0.0:
            cross_axis_matrix = self.backend.asarray([
                [1.0, self.cross_axis[1, 0] / 100.0, self.cross_axis[2, 0] / 100.0],
                [self.cross_axis[0, 0] / 100.0, 1.0, self.cross_axis[2, 0] / 100.0],
                [self.cross_axis[0, 0] / 100.0, self.cross_axis[1, 0] / 100.0, 1.0],
            ], dtype=float)
            misalignment_vector = vector(1.0, 1.0, 1.0, backend=self.backend) * float(np.radians(self.misalignment))
            misalignment_quaternion = Quaternion.from_rotation_vector(misalignment_vector, backend=self.backend)
            misalignment_matrix = misalignment_quaternion.to_rotation_matrix()
            combined_matrix = self.backend.matmul(cross_axis_matrix, misalignment_matrix)
            sensor = self.backend.matmul(combined_matrix, sensor)

        if self.measurement_range is not None and self.measurement_range != 0 and self._deterministic:
            sensor = self.backend.clip(sensor, -self.measurement_range, self.measurement_range)

        return sensor

    def voltages(self, t):
        true = self.true_values(t)
        noisy = true + self.noise_voltages(t)
        return self._apply_sensor_errors(noisy, t)

    def set_deterministic(self, deterministic=True):
        self._deterministic = deterministic
        return self


class Accelerometer(Sensor):
    def true_values(self, t):
        if self.trajectory is None:
            return zeros((3, 1), backend=self.backend)
        position = self.trajectory.position(t)
        acceleration = self.trajectory.acceleration(t)
        gravity = self.platform.simulation.environment.gravity(position, t)
        measurement = acceleration - gravity
        return self.trajectory.rotation(t).conjugate().rotate_vector(measurement)


class IdealAccelerometer(Accelerometer, IdealSensor):
    pass


class NoisyAccelerometer(NoisySensor, Accelerometer):
    pass


class Gyroscope(Sensor):
    def true_values(self, t):
        if self.trajectory is None:
            return zeros((3, 1), backend=self.backend)
        omega = self.trajectory.rotational_velocity(t)
        return self.trajectory.rotation(t).conjugate().rotate_vector(omega)


class IdealGyroscope(Gyroscope, IdealSensor):
    pass


class NoisyGyroscope(NoisySensor, Gyroscope):
    pass


class Magnetometer(Sensor):
    def __init__(self, platform=None, field_vector=None, **kwargs):
        self.field_vector = vector(0.0, 0.0, 1.0, backend=get_backend('numpy')) if field_vector is None else field_vector
        super().__init__(platform=platform, **kwargs)

    def true_values(self, t):
        if self.trajectory is None:
            return zeros((3, 1), backend=self.backend)
        field = self.field_vector
        if hasattr(t, 'shape') and len(t.shape) > 0:
            field = self.backend.broadcast_to(field, (3, t.shape[0]))
        return self.trajectory.rotation(t).conjugate().rotate_vector(field)


class IdealMagnetometer(Magnetometer, IdealSensor):
    pass


class NoisyMagnetometer(NoisySensor, Magnetometer):
    pass
