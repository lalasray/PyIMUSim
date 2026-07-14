class Component:
    def __init__(self, platform=None):
        self.platform = platform
        self.simulation = getattr(platform, 'simulation', None) if platform is not None else None
        self.trajectory = getattr(platform, 'trajectory', None) if platform is not None else None
        if platform is not None and hasattr(platform, 'attach_component'):
            platform.attach_component(self)

    def _simulation_change(self):
        self.simulation = self.platform.simulation if self.platform is not None else None

    def _trajectory_change(self):
        pass


class Platform:
    def __init__(self, simulation=None, trajectory=None):
        self.simulation = simulation
        self.trajectory = trajectory
        if not hasattr(self, '_components'):
            self._components = []
        if simulation is not None:
            self.set_simulation(simulation)
        if trajectory is not None:
            self.set_trajectory(trajectory)

    def attach_component(self, component):
        if not hasattr(self, '_components'):
            self._components = []
        self._components.append(component)

    def set_simulation(self, simulation):
        self.simulation = simulation
        for component in self._components:
            if hasattr(component, '_simulation_change'):
                component._simulation_change()

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory
        for component in self._components:
            if hasattr(component, '_trajectory_change'):
                component._trajectory_change()
