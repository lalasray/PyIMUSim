import pyimusim


def run_gpu_demo():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        from pyimusim.backend import get_backend
        backend = get_backend('torch', device='cuda')
        print('Using backend: torch_cuda')
    else:
        print('Torch CUDA backend unavailable; falling back to numpy.')
        from pyimusim.backend import get_backend
        backend = get_backend('numpy')

    quat = pyimusim.Quaternion([1.0, 0.0, 0.0, 0.0], backend=backend)
    vector = pyimusim.vector(1.0, 0.0, 0.0, backend=backend)
    rotated = quat.rotate_vector(vector)
    print('rotated vector:', rotated)
    print('shape:', rotated.shape)


if __name__ == '__main__':
    run_gpu_demo()
