from numba import cuda

# threadsperblock = 32
# GrDim = 100, 200
# BlkDim = 16, 16


# def compute_with_cuda(func):
#     @cuda.jit()
#     def inner(*args, **kwargs):
#         assert cuda.is_available()
#
#         new_args = [cuda.to_device(arg) for arg in args]
#         # blockspergrid = (args[0].size + (threadsperblock - 1)) // threadsperblock
#
#         func[GrDim, BlkDim](*new_args, **kwargs)
#
#         cuda.synchronize()
#
#         args = [arg.copy_to_host() for arg in new_args]
#
#         return func(*args, **kwargs)
#
#     return inner


class CudaManager(object):
    def __init__(self, *values, func, GrDim=(100, 200), BlkDim=(16, 16)):
        assert cuda.is_available()
        self.cuda_values = [cuda.to_device(value) for value in values]
        self.func = func
        self.GrDim = GrDim
        self.BlkDim = BlkDim
        self.res = None

    def __enter__(self):

        self.res = self.func[*self.GrDim, *self.GrDim](*self.cuda_values)
        cuda.synchronize()
        return self.res

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.res = self.res.copy_to_host()
        return self.res








