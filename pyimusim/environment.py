from .math import vector
from .backend import get_backend


class ConstantGravityField:
    def __init__(self, magnitude=9.81, backend=None):
        self.backend = backend or get_backend('numpy')
        self._value = vector(0.0, 0.0, magnitude, backend=self.backend)

    def __call__(self, position, t=None):
        length = 1
        if hasattr(position, 'shape') and len(position.shape) > 1:
            length = position.shape[1]
        return self._value * self.backend.ones((1, length), dtype=float)


class Environment:
    def __init__(self, gravity=None, backend=None):
        self.backend = backend or get_backend('numpy')
        self.gravity = gravity or ConstantGravityField(backend=self.backend)
