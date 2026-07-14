import pyimusim
from pyimusim.backend import get_backend


def test_idealimu_run():
    sim = pyimusim.Simulation()
    trajectory = pyimusim.StaticTrajectory()
    imu = pyimusim.IdealIMU(simulation=sim, trajectory=trajectory)
    result = sim.run(imu, duration=0.5, dt=0.1)

    assert 'IdealAccelerometer' in result['measurements']
    assert 'IdealGyroscope' in result['measurements']
    assert 'IdealMagnetometer' in result['measurements']
    assert result['times'].shape[0] == 6
    assert result['measurements']['IdealAccelerometer'].shape == (3, 6)


try:
    import torch  # noqa: F401

    def test_torch_simulation_backward():
        backend = get_backend('torch', device='cpu')
        rot = backend.asarray([0.0, 0.0, 1.0], dtype=float)
        rot = backend.reshape(rot, (3, 1))
        rot.requires_grad_(True)

        trajectory = pyimusim.ConstantVelocityTrajectory(
            backend.zeros((3, 1), dtype=float),
            backend.zeros((3, 1), dtype=float),
            backend=backend,
            rotational_velocity=rot,
        )
        sim = pyimusim.Simulation(backend=backend)
        imu = pyimusim.IdealIMU(simulation=sim, trajectory=trajectory).set_deterministic(True)

        result = sim.run(imu, duration=0.1, dt=0.1)
        loss = result['measurements']['IdealGyroscope'].sum()
        loss.backward()

        assert rot.grad is not None
        assert torch.any(rot.grad != 0)
except ImportError:
    pass
