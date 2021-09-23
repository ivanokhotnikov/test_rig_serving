import time
import functools


def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f'[{elapsed*1e3:.3f} ms] {func.__name__}')
        return result

    return clocked