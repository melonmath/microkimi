def prepare_cu_seqlens_from_mask(*a, **kw):
    raise NotImplementedError("shim: padding masks unsupported; run with batch=1, mask=None")


def prepare_lens_from_mask(*a, **kw):
    raise NotImplementedError("shim: padding masks unsupported; run with batch=1, mask=None")
