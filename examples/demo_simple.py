import pyimusim


def run_demo():
    sim = pyimusim.Simulation()
    trajectory = pyimusim.StaticTrajectory()
    imu = pyimusim.IdealIMU(simulation=sim, trajectory=trajectory)
    result = sim.run(imu, duration=1.0, dt=0.1)

    print('times:', result['times'])
    print('accelerometer measurements:', result['measurements']['IdealAccelerometer'])


if __name__ == '__main__':
    run_demo()
