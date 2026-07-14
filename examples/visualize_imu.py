"""Optional interactive 3D visualization example for PyIMUSim.

This example uses Plotly to show a moving IMU frame, a trajectory path,
and the gravity/acceleration vectors that the sensor sees.
It is intentionally optional and should be installed separately:

    pip install plotly
"""

import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Install plotly to run this example: pip install plotly") from exc

import pyimusim


def _rotation_matrix_from_quaternion(q):
    w, x, y, z = q[0, 0], q[1, 0], q[2, 0], q[3, 0]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _body_frame_axes(q, scale=1.0):
    rotation = _rotation_matrix_from_quaternion(q)
    axes = np.eye(3) * scale
    return rotation @ axes


def build_demo():
    sim = pyimusim.Simulation()
    trajectory = pyimusim.ConstantVelocityTrajectory(
        position=pyimusim.vector(0.0, 0.0, 0.0),
        velocity=pyimusim.vector(0.3, 0.0, 0.0),
        rotation=pyimusim.Quaternion.from_euler(0.0, 0.0, 0.0),
        rotational_velocity=pyimusim.vector(0.0, 0.0, 0.4),
    )
    imu = pyimusim.IdealIMU(simulation=sim, trajectory=trajectory)
    result = sim.run(imu, duration=2.0, dt=0.05)

    times = np.asarray(result['times']).reshape(-1)
    accel = np.asarray(result['measurements']['IdealAccelerometer'])
    gyro = np.asarray(result['measurements']['IdealGyroscope'])

    # smooth sine-wave trajectory for the example
    trajectory_points = np.stack([
        np.linspace(0.0, 1.0, len(times)),
        0.6 * np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, len(times))),
        0.6 * np.cos(2.0 * np.pi * np.linspace(0.0, 1.0, len(times))),
    ], axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=trajectory_points[:, 0],
        y=trajectory_points[:, 1],
        z=trajectory_points[:, 2],
        mode='lines+markers',
        name='trajectory',
        line=dict(color='rgba(80, 120, 255, 0.95)', width=4),
        marker=dict(size=4, color='rgba(80, 120, 255, 1.0)'),
    ))

    fig.add_trace(go.Scatter3d(
        x=times,
        y=accel[0, :],
        z=accel[2, :],
        mode='lines+markers',
        name='accelerometer signal',
        line=dict(color='rgba(255, 90, 90, 0.95)', width=3),
        marker=dict(size=3, color='rgba(255, 90, 90, 1.0)'),
    ))

    fig.add_trace(go.Scatter3d(
        x=times,
        y=gyro[0, :],
        z=gyro[2, :],
        mode='lines+markers',
        name='gyroscope signal',
        line=dict(color='rgba(40, 180, 120, 0.95)', width=3),
        marker=dict(size=3, color='rgba(40, 180, 120, 1.0)'),
    ))

    # add a moving IMU frame and gravity arrows at a few sample points
    for idx in range(0, len(times), 10):
        q = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=float)
        if idx < len(times):
            q = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=float)
        axes = _body_frame_axes(q, scale=0.2)
        origin = np.array([trajectory_points[idx, 0], trajectory_points[idx, 1], trajectory_points[idx, 2]], dtype=float)
        x_axis = origin + axes[:, 0]
        y_axis = origin + axes[:, 1]
        z_axis = origin + axes[:, 2]
        fig.add_trace(go.Scatter3d(
            x=[origin[0], x_axis[0]], y=[origin[1], x_axis[1]], z=[origin[2], x_axis[2]],
            mode='lines', line=dict(color='red', width=2), showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[origin[0], y_axis[0]], y=[origin[1], y_axis[1]], z=[origin[2], y_axis[2]],
            mode='lines', line=dict(color='green', width=2), showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[origin[0], z_axis[0]], y=[origin[1], z_axis[1]], z=[origin[2], z_axis[2]],
            mode='lines', line=dict(color='blue', width=2), showlegend=False,
        ))
        gravity_end = origin + np.array([0.0, 0.0, -0.3])
        fig.add_trace(go.Scatter3d(
            x=[origin[0], gravity_end[0]], y=[origin[1], gravity_end[1]], z=[origin[2], gravity_end[2]],
            mode='lines', line=dict(color='orange', width=2), showlegend=False,
        ))

    fig.update_layout(
        title='PyIMUSim optional interactive view',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
        ),
        width=1100,
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [[]]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None]]},
            ],
        }],
    )

    frames = []
    for idx in range(len(times)):
        frame = go.Frame(data=[go.Scatter3d(
            x=[trajectory_points[idx, 0]], y=[trajectory_points[idx, 1]], z=[trajectory_points[idx, 2]],
            mode='markers', marker=dict(size=8, color='black'), showlegend=False,
        )], traces=[0])
        frames.append(frame)
    fig.frames = frames

    return fig


if __name__ == '__main__':
    fig = build_demo()
    fig.show()
