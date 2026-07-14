from abc import ABC, abstractmethod

try:
    import torch
except ImportError:
    torch = None

import numpy as np


class Backend(ABC):
    """Abstract numeric backend for PyIMUSim."""

    name = 'base'

    def __init__(self, device='cpu'):
        self.device = device

    @abstractmethod
    def array(self, data, dtype=None):
        pass

    @abstractmethod
    def asarray(self, data, dtype=None):
        pass

    @abstractmethod
    def zeros(self, shape, dtype=None):
        pass

    @abstractmethod
    def ones(self, shape, dtype=None):
        pass

    @abstractmethod
    def eye(self, size, dtype=None):
        pass

    @abstractmethod
    def arange(self, start, stop=None, step=1, dtype=None):
        pass

    @abstractmethod
    def sin(self, value):
        pass

    @abstractmethod
    def cos(self, value):
        pass

    @abstractmethod
    def sqrt(self, value):
        pass

    @abstractmethod
    def arctan2(self, y, x):
        pass

    @abstractmethod
    def arcsin(self, value):
        pass

    @abstractmethod
    def arccos(self, value):
        pass

    @abstractmethod
    def clip(self, value, min_value, max_value):
        pass

    @abstractmethod
    def stack(self, arrays, axis=0):
        pass

    @abstractmethod
    def concatenate(self, arrays, axis=0):
        pass

    @abstractmethod
    def sum(self, value, axis=None, keepdims=False):
        pass

    @abstractmethod
    def matmul(self, a, b):
        pass

    @abstractmethod
    def reshape(self, tensor, shape):
        pass

    @abstractmethod
    def transpose(self, tensor, axes=None):
        pass

    @abstractmethod
    def where(self, condition, x, y):
        pass

    @abstractmethod
    def equal(self, a, b):
        pass

    @abstractmethod
    def broadcast_to(self, array, shape):
        pass

    @abstractmethod
    def repeat(self, array, repeats, axis=0):
        pass

    @abstractmethod
    def zeros_like(self, value, dtype=None):
        pass

    @abstractmethod
    def normal(self, shape, mean=0.0, std=1.0):
        pass

    def norm(self, value, axis=None, keepdims=False):
        squared = value * value
        return self.sqrt(self.sum(squared, axis=axis, keepdims=keepdims))


class NumpyBackend(Backend):
    name = 'numpy'

    def __init__(self, device='cpu'):
        super().__init__(device)

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or np.float64)

    def asarray(self, data, dtype=None):
        return np.asarray(data, dtype=dtype or np.float64)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or np.float64)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or np.float64)

    def eye(self, size, dtype=None):
        return np.eye(size, dtype=dtype or np.float64)

    def arange(self, start, stop=None, step=1, dtype=None):
        if stop is None:
            return np.arange(start, dtype=dtype or np.float64)
        return np.arange(start, stop, step, dtype=dtype or np.float64)

    def sin(self, value):
        return np.sin(value)

    def cos(self, value):
        return np.cos(value)

    def sqrt(self, value):
        return np.sqrt(value)

    def arctan2(self, y, x):
        return np.arctan2(y, x)

    def arcsin(self, value):
        return np.arcsin(value)

    def arccos(self, value):
        return np.arccos(value)

    def clip(self, value, min_value, max_value):
        return np.clip(value, min_value, max_value)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def sum(self, value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def reshape(self, tensor, shape):
        return np.reshape(tensor, shape)

    def transpose(self, tensor, axes=None):
        return np.transpose(tensor, axes=axes)

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def equal(self, a, b):
        return a == b

    def broadcast_to(self, array, shape):
        return np.broadcast_to(array, shape)

    def repeat(self, array, repeats, axis=0):
        return np.repeat(array, repeats, axis=axis)

    def zeros_like(self, value, dtype=None):
        return np.zeros_like(value, dtype=dtype or np.float64)

    def normal(self, shape, mean=0.0, std=1.0):
        return np.random.normal(loc=mean, scale=std, size=shape)


class TorchBackend(Backend):
    name = 'torch'

    def __init__(self, device='cpu'):
        if torch is None:
            raise ImportError('PyTorch is required for the torch backend.')
        super().__init__(device)
        self.device = torch.device(device)

    def _dtype(self, dtype):
        if dtype is None or dtype == float:
            return torch.float64
        return dtype

    def array(self, data, dtype=None):
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=self.device, dtype=self._dtype(dtype))
            return tensor.requires_grad_(data.requires_grad) if data.requires_grad else tensor
        return torch.as_tensor(data, dtype=self._dtype(dtype), device=self.device)

    def asarray(self, data, dtype=None):
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=self.device, dtype=self._dtype(dtype))
            return tensor.requires_grad_(data.requires_grad) if data.requires_grad else tensor
        return torch.as_tensor(data, dtype=self._dtype(dtype), device=self.device)

    def zeros(self, shape, dtype=None):
        return torch.zeros(shape, dtype=self._dtype(dtype), device=self.device)

    def ones(self, shape, dtype=None):
        return torch.ones(shape, dtype=self._dtype(dtype), device=self.device)

    def eye(self, size, dtype=None):
        return torch.eye(size, dtype=self._dtype(dtype), device=self.device)

    def arange(self, start, stop=None, step=1, dtype=None):
        if stop is None:
            return torch.arange(start, dtype=self._dtype(dtype), device=self.device)
        return torch.arange(start, stop, step, dtype=self._dtype(dtype), device=self.device)

    def sin(self, value):
        return torch.sin(value)

    def cos(self, value):
        return torch.cos(value)

    def sqrt(self, value):
        return torch.sqrt(value)

    def arctan2(self, y, x):
        return torch.atan2(y, x)

    def arcsin(self, value):
        return torch.arcsin(value)

    def arccos(self, value):
        return torch.acos(value)

    def clip(self, value, min_value, max_value):
        return torch.clamp(value, min_value, max_value)

    def stack(self, arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    def concatenate(self, arrays, axis=0):
        return torch.cat(arrays, dim=axis)

    def sum(self, value, axis=None, keepdims=False):
        return torch.sum(value, dim=axis, keepdim=keepdims)

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def reshape(self, tensor, shape):
        return torch.reshape(tensor, shape)

    def transpose(self, tensor, axes=None):
        if axes is None:
            return tensor.T
        return tensor.permute(axes)

    def where(self, condition, x, y):
        return torch.where(condition, x, y)

    def equal(self, a, b):
        return a == b

    def broadcast_to(self, array, shape):
        if not isinstance(array, torch.Tensor):
            array = torch.as_tensor(array, dtype=self._dtype(None), device=self.device)
        return array.expand(shape)

    def repeat(self, array, repeats, axis=0):
        return array.repeat_interleave(repeats, dim=axis)

    def zeros_like(self, value, dtype=None):
        return torch.zeros_like(value, dtype=self._dtype(dtype))

    def normal(self, shape, mean=0.0, std=1.0):
        return torch.normal(mean=mean, std=std, size=shape, device=self.device, dtype=self._dtype(None))


def get_backend(name=None, device='cpu'):
    if name is None:
        if torch is not None:
            return TorchBackend(device=device)
        return NumpyBackend(device=device)

    name = name.lower()
    if name == 'numpy':
        return NumpyBackend(device=device)
    if name == 'torch':
        return TorchBackend(device=device)
    raise ValueError(f'Unknown backend: {name}')
