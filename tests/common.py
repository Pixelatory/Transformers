import torch


class DebugTensor(torch.Tensor):
    # Override __torch_function__ to log calls
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        print(f"Called: {func.__name__}")
        print(f"Args: {args}, Kwargs: {kwargs}")
        return super().__torch_function__(func, types, args, kwargs)


t = DebugTensor(torch.ones(size=(1, 2)))
t2 = DebugTensor(torch.ones(size=(1, 2)))

t + t2
