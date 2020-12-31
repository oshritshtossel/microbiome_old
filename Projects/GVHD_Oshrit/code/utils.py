import sys


def in_debug() -> bool:
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        # We can't check this
        return False
    elif gettrace():
        return True
    else:
        return False
