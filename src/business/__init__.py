"""能见度业务化处理包。"""

from .config import DEFAULT_PROVINCES, BusinessConfig


def run_once(*args, **kwargs):
    from .pipeline import run_once as _run_once

    return _run_once(*args, **kwargs)


def run_forever(*args, **kwargs):
    from .pipeline import run_forever as _run_forever

    return _run_forever(*args, **kwargs)


def close_logging(*args, **kwargs):
    from .pipeline import close_logging as _close_logging

    return _close_logging(*args, **kwargs)

__all__ = ["BusinessConfig", "DEFAULT_PROVINCES", "run_once", "run_forever", "close_logging"]
