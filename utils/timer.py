import time
from contextlib import contextmanager


@contextmanager
def timer(label="Step"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{label}] 耗时: {elapsed:.4f} s")
