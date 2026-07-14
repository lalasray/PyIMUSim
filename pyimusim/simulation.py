from .backend import get_backend
from .environment import Environment


class Simulation:
    def __init__(self, backend=None, device='cpu', environment=None, seed=None):
        self.backend = backend or get_backend(None if device == 'cpu' else 'torch', device=device)
        self.environment = environment or Environment(backend=self.backend)
        self.seed = seed
        if self.backend.name == 'torch':
            try:
                import torch
                if seed is not None:
                    torch.manual_seed(seed)
            except ImportError:
                pass

    def run(self, imu, duration, dt, start_time=0.0, trajectory=None):
        if trajectory is not None:
            imu.set_trajectory(trajectory)
        imu.simulation = self
        if imu.trajectory is not None and hasattr(imu.trajectory, 'to_backend'):
            imu.set_trajectory(imu.trajectory.to_backend(self.backend))
        times = self.backend.arange(start_time, duration + dt, dt)
        measurements = imu.collect_measurements(times)
        return {
            'times': times,
            'measurements': measurements,
        }
