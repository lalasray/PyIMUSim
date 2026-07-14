import numpy as np
from pyimusim.trajectory import StaticTrajectory, ConstantVelocityTrajectory, OffsetTrajectory
from pyimusim.math import vector, Quaternion
from pyimusim.backend import get_backend


def test_static_trajectory():
    backend = get_backend('numpy')
    trajectory = StaticTrajectory(backend=backend)
    t = np.array([0.0, 1.0, 2.0])
    position = trajectory.position(t)
    assert position.shape == (3, 3)
    assert np.allclose(position, np.zeros((3, 3)))
    assert np.allclose(trajectory.velocity(t), np.zeros((3, 3)))
    assert np.allclose(trajectory.acceleration(t), np.zeros((3, 3)))


def test_constant_velocity_trajectory():
    backend = get_backend('numpy')
    position0 = vector(0.0, 0.0, 0.0, backend=backend)
    velocity0 = vector(1.0, 0.0, 0.0, backend=backend)
    trajectory = ConstantVelocityTrajectory(position0, velocity0, backend=backend)
    t = np.array([0.0, 1.0, 2.0])
    position = trajectory.position(t)
    assert np.allclose(position, np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(trajectory.velocity(t), np.tile(velocity0, (1, 3)))
    assert np.allclose(trajectory.acceleration(t), np.zeros((3, 3)))


def test_offset_trajectory():
    backend = get_backend('numpy')
    base = StaticTrajectory(backend=backend)
    offset = vector(1.0, 0.0, 0.0, backend=backend)
    parent = ConstantVelocityTrajectory(vector(0.0, 0.0, 0.0, backend=backend), vector(0.0, 1.0, 0.0, backend=backend), backend=backend)
    offset_traj = OffsetTrajectory(parent, position_offset=offset, rotation_offset=Quaternion(backend=backend))
    t = np.array([0.0, 1.0])
    position = offset_traj.position(t)
    assert np.allclose(position, np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
    assert position.shape == (3, 2)
