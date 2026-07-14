from .platform import Platform
from .math import vector, Quaternion
from .sensors import IdealAccelerometer, IdealGyroscope, IdealMagnetometer, NoisyAccelerometer, NoisyGyroscope, NoisyMagnetometer


class IMU(Platform):
    @property
    def sensors(self):
        return []

    def set_deterministic(self, deterministic=True):
        for sensor in self.sensors:
            if hasattr(sensor, 'set_deterministic'):
                sensor.set_deterministic(deterministic)
        return self

    def set_trajectory(self, trajectory):
        super().set_trajectory(trajectory)
        return self

    def set_simulation(self, simulation):
        super().set_simulation(simulation)
        return self

    def collect_measurements(self, times):
        return {type(sensor).__name__: sensor.voltages(times) for sensor in self.sensors}


class StandardIMU(IMU):
    def __init__(self, simulation=None, trajectory=None):
        if not hasattr(self, 'accelerometer'):
            self.accelerometer = IdealAccelerometer(self)
        if not hasattr(self, 'gyroscope'):
            self.gyroscope = IdealGyroscope(self)
        if not hasattr(self, 'magnetometer'):
            self.magnetometer = IdealMagnetometer(self)
        super().__init__(simulation=simulation, trajectory=trajectory)

    @property
    def sensors(self):
        return [self.accelerometer, self.gyroscope, self.magnetometer]


class IdealIMU(StandardIMU):
    pass


class IdealIMUOffset(StandardIMU):
    def __init__(self, position_offset=None, rotation_offset=None, simulation=None, trajectory=None):
        self.accelerometer = IdealAccelerometer(self, position_offset=position_offset, rotation_offset=rotation_offset)
        self.gyroscope = IdealGyroscope(self, position_offset=position_offset, rotation_offset=rotation_offset)
        self.magnetometer = IdealMagnetometer(self, position_offset=position_offset, rotation_offset=rotation_offset)
        super().__init__(simulation=simulation, trajectory=trajectory)


class MagicIMU(StandardIMU):
    def __init__(self, simulation=None, trajectory=None):
        self.accelerometer = IdealAccelerometer(self)
        self.gyroscope = IdealGyroscope(self)
        self.magnetometer = IdealMagnetometer(self, field_vector=vector(0.0, 0.0, 9.81))
        super().__init__(simulation=simulation, trajectory=trajectory)


class Orient3IMU(StandardIMU):
    def __init__(self, simulation=None, trajectory=None, noise_std=0.01):
        self.accelerometer = NoisyAccelerometer(self, noise_std=noise_std)
        self.gyroscope = NoisyGyroscope(self, noise_std=noise_std)
        self.magnetometer = NoisyMagnetometer(self, noise_std=noise_std)
        super().__init__(simulation=simulation, trajectory=trajectory)
