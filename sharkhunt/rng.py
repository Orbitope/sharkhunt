"""mulberry32, mirrored exactly by docs/rng.js.

The article's widgets re-run these simulations in the browser, so the Python and
JavaScript sides have to agree bit for bit. numpy's generators don't port, but
mulberry32 does: it is ten lines of int32 arithmetic that JavaScript expresses
natively with ``Math.imul`` and ``>>>``, and Python can reproduce with masks.
``tests/test_rng_parity.py`` checks the two against each other.
"""

import math

_M32 = 0xFFFFFFFF


def _i32(x):
    """Wrap to signed 32-bit, matching JavaScript's ``x | 0``."""
    x &= _M32
    return x - 0x100000000 if x & 0x80000000 else x


def _imul(a, b):
    """Signed 32-bit multiply, matching ``Math.imul``."""
    return _i32((a & _M32) * (b & _M32))


def _ushr(x, k):
    """Unsigned right shift, matching ``x >>> k``."""
    return (x & _M32) >> k


class Rng:
    """Deterministic stream. Same seed, same numbers, in either language."""

    def __init__(self, seed):
        self.state = _i32(int(seed))

    def random(self):
        self.state = _i32(self.state + 0x6D2B79F5)
        a = self.state
        t = _imul(a ^ _ushr(a, 15), 1 | a)
        t = _i32(_i32(t + _imul(t ^ _ushr(t, 7), 61 | t)) ^ t)
        return _ushr(t ^ _ushr(t, 14), 0) / 4294967296.0

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.random()

    def randint(self, lo, hi):
        """Uniform integer in [lo, hi)."""
        return lo + int(self.random() * (hi - lo))

    def chance(self, p):
        return self.random() < p

    def choice_p(self, probs):
        """Index sampled from a probability vector (assumed to sum to 1)."""
        u = self.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if u < acc:
                return i
        return len(probs) - 1

    def normal(self):
        """Standard normal via Box-Muller. Uses two uniforms, no caching, so the
        stream position stays trivially predictable from the call sequence."""
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def spawn(self, salt):
        """A child stream, so each player can draw independently of match order."""
        return Rng(_i32(_imul(self.state ^ _i32(salt), 0x9E3779B1)))
