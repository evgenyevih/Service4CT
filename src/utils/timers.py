from __future__ import annotations
import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.time()
    try:
        yield lambda: time.time() - start
    finally:
        pass
