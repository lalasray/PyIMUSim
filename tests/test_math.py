import numpy as np
from pyimusim.math import vector, cross, dot, norm, Quaternion, fix_continuity
from pyimusim.backend import get_backend


def _run_math_tests(backend):
    v1 = vector(1.0, 0.0, 0.0, backend=backend)
    v2 = vector(0.0, 1.0, 0.0, backend=backend)

    cross_prod = cross(v1, v2, backend=backend)
    assert cross_prod.shape == (3, 1)
    assert np.allclose(cross_prod if backend.name == 'numpy' else cross_prod.cpu().numpy(), np.array([[0.0], [0.0], [1.0]]))

    dot_prod = dot(v1, v2, backend=backend)
    dot_val = dot_prod if backend.name == 'numpy' else dot_prod.cpu().numpy()
    assert np.allclose(dot_val, np.array([0.0]))

    q = Quaternion([1.0, 0.0, 0.0, 0.0], backend=backend)
    rotated = q.rotate_vector(v1)
    rotated_val = rotated if backend.name == 'numpy' else rotated.cpu().numpy()
    assert np.allclose(rotated_val, np.array([[1.0], [0.0], [0.0]]))

    q2 = Quaternion.from_rotation_vector(vector(0.0, 0.0, np.pi / 2, backend=backend), backend=backend)
    rotated2 = q2.rotate_vector(v1)
    rotated2_val = rotated2 if backend.name == 'numpy' else rotated2.cpu().numpy()
    assert np.allclose(rotated2_val, np.array([[0.0], [1.0], [0.0]]), atol=1e-6)


def test_quaternion_conversions():
    backend = get_backend('numpy')
    q = Quaternion.from_euler(np.pi / 6, np.pi / 4, np.pi / 3, backend=backend)
    euler = q.to_euler()
    euler_val = euler if backend.name == 'numpy' else euler.cpu().numpy()
    assert np.allclose(euler_val.flatten(), np.array([np.pi / 6, np.pi / 4, np.pi / 3]), atol=1e-6)

    axis, angle = q.to_axis_angle()
    axis_val = axis if backend.name == 'numpy' else axis.cpu().numpy()
    angle_val = angle if backend.name == 'numpy' else angle.cpu().numpy()
    assert np.allclose(np.linalg.norm(axis_val, axis=0), 1.0, atol=1e-6)
    assert np.allclose(angle_val.flatten(), np.array([2.0 * np.arccos(np.clip(q.values[0, 0], -1.0, 1.0))]), atol=1e-6)

    q2 = Quaternion.from_euler(euler_val[0, 0], euler_val[1, 0], euler_val[2, 0], backend=backend)
    assert np.allclose(q2.values if backend.name == 'numpy' else q2.values.cpu().numpy(), q.values, atol=1e-6)


def test_quaternion_fix_continuity():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = -q1
    stacked = np.stack([q1, q2], axis=1)
    fixed = fix_continuity(stacked)
    assert np.allclose(fixed[:, 1], q1)


def test_math_numpy():
    backend = get_backend('numpy')
    _run_math_tests(backend)


try:
    import torch  # noqa: F401

    def test_math_torch():
        backend = get_backend('torch', device='cpu')
        _run_math_tests(backend)

    def test_torch_simulation_backward():
        import pyimusim

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
