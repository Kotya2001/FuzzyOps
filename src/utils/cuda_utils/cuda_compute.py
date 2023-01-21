from numba import cuda

# threadsperblock = 32
GrDim = 100 , 200
BlkDim = 16, 16


def compute_with_cuda(func):
    @cuda.jit()
    def inner(*args, **kwargs):
        assert cuda.is_available()

        new_args = [cuda.to_device(arg) for arg in args]
        # blockspergrid = (args[0].size + (threadsperblock - 1)) // threadsperblock

        func[GrDim, BlkDim](*new_args, **kwargs)

        cuda.synchronize()

        args = [arg.copy_to_host() for arg in new_args]

        return func(*args, **kwargs)

    return inner
