"""Compatibility-only schema annotation bridge; no tensor/math changes.

Torch 2.4 infer_schema cannot interpret future-annotations strings/PEP604 unions
used by public Diffusers 0.39 custom operators. Resolve only annotations passed
to registration; retain Torch, original operator functions, and kernels.
"""
import types
import typing
import functools

def normalize(annotation):
    origin = typing.get_origin(annotation)
    args = tuple(normalize(x) for x in typing.get_args(annotation))
    if origin in (types.UnionType, typing.Union):
        return typing.Union[args]
    if origin is tuple:
        return typing.Tuple[args]
    if origin is list:
        return typing.List[args[0]]
    return annotation

def install():
    import torch
    if not torch.__version__.startswith('2.4.'):
        raise RuntimeError('annotation bridge is scoped to Torch 2.4')
    original = torch.library.custom_op
    if getattr(original, '_evoke_annotation_bridge', False):
        return

    @functools.wraps(original)
    def custom_op(name, fn=None, **kwargs):
        def register(function):
            function.__annotations__ = {k: normalize(v) for k,v in typing.get_type_hints(function).items()}
            return original(name, function, **kwargs)
        return register(fn) if fn is not None else register

    custom_op._evoke_annotation_bridge = True
    torch.library.custom_op = custom_op
