import functools

import torch

_DEFAULT_DEVICE_OVERRIDE = None


def set_default_device(device):
    global _DEFAULT_DEVICE_OVERRIDE
    _DEFAULT_DEVICE_OVERRIDE = torch.device(device) if device is not None else None

def get_default_device():
    if _DEFAULT_DEVICE_OVERRIDE is not None:
        return _DEFAULT_DEVICE_OVERRIDE
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _find_device(value):
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, torch.device):
        return value
    if isinstance(value, str):
        try:
            return torch.device(value)
        except (RuntimeError, TypeError):
            return None
    if hasattr(value, "device"):
        try:
            return torch.device(value.device)
        except (RuntimeError, TypeError):
            return None
    return None


def _resolve_call_device(args, kwargs):
    for value in list(args) + list(kwargs.values()):
        device = _find_device(value)
        if device is not None:
            return device
    return get_default_device()


def _autocast_enabled(device, enabled):
    device = torch.device(device)
    return bool(enabled) and device.type == "cuda"

def safe_autocast_decorator(enabled=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = _resolve_call_device(args, kwargs)
            if _autocast_enabled(device, enabled):
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

import contextlib
@contextlib.contextmanager
def safe_autocast(enabled=True, device=None):
    device = torch.device(device) if device is not None else get_default_device()
    if _autocast_enabled(device, enabled):
        with torch.amp.autocast(device_type=device.type, enabled=True):
            yield
    else:
        yield
