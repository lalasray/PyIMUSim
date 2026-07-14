# PyIMUSim

A Python-only differentiable IMU simulator with both CPU and optional GPU backends.

## Features

- Pure Python IMU simulation framework
- `numpy` backend for CPU execution
- Optional `torch` backend for GPU-accelerated tensor operations
- Trajectory, environment, and sensor models
- `IdealIMU`, `Orient3IMU`, and platform simulation support

## Installation

### Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -U pip
pip install numpy
```

### Optional GPU backend

```bash
pip install torch
```

## Usage

```python
import pyimusim

sim = pyimusim.Simulation()
trajectory = pyimusim.StaticTrajectory()
imu = pyimusim.IdealIMU(simulation=sim, trajectory=trajectory)
result = sim.run(imu, duration=1.0, dt=0.01)

print(result['times'])
print(result['measurements']['IdealAccelerometer'])
```

## Torch backend

```python
import pyimusim

backend = pyimusim.get_backend('torch')
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
print(rot.grad)
```

## Deterministic vs noisy sensor mode

```python
import pyimusim

sim = pyimusim.Simulation()
trajectory = pyimusim.StaticTrajectory()
imu = pyimusim.Orient3IMU(simulation=sim, trajectory=trajectory, noise_std=0.01)

imu.set_deterministic(False)
noisy = sim.run(imu, duration=0.1, dt=0.1)['measurements']['NoisyAccelerometer']

imu.set_deterministic(True)
clean = sim.run(imu, duration=0.1, dt=0.1)['measurements']['NoisyAccelerometer']

print('noisy', noisy)
print('clean', clean)
```

## Package metadata

Install the package from source:

```bash
pip install -e .
```

## Optional visualization example

A lightweight 3D visualization example is available at `examples/visualize_imu.py`. It uses Plotly for an interactive view of IMU signals and is intentionally optional.

```bash
pip install plotly
python examples/visualize_imu.py
```

## GPU example

See `examples/demo_torch_gpu.py` for a backend example using PyTorch (if installed).

## Testing

```bash
pip install pytest
pytest
```
