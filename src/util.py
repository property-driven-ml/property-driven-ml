from contextlib import contextmanager

@contextmanager
def maybe(context_manager, flag: bool):
    if flag:
        with context_manager as cm:
            yield cm
    else:
        yield None