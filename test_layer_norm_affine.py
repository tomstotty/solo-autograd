import math

from autograd import Tensor


def expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:
        raise AssertionError(
            f"expected {exc.__name__}, got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"expected {exc.__name__}, no error raised")


x = Tensor([1.0, 2.0, 3.0, 4.0])
w = Tensor([1.0, 1.0, 1.0, 1.0])
b = Tensor([0.0, 0.0, 0.0, 0.0])

# --- eps boundary regression: only a positive finite float is legal ---
# The contract must not be relaxed: bool/int/str/None are TypeError, and
# zero, negative or non-finite floats are ValueError.
expect(TypeError, lambda: x.layer_norm_affine(w, b, True))
expect(TypeError, lambda: x.layer_norm_affine(w, b, False))
expect(TypeError, lambda: x.layer_norm_affine(w, b, 1))
expect(TypeError, lambda: x.layer_norm_affine(w, b, 0))
expect(TypeError, lambda: x.layer_norm_affine(w, b, "1e-5"))
expect(TypeError, lambda: x.layer_norm_affine(w, b, None))
expect(ValueError, lambda: x.layer_norm_affine(w, b, 0.0))
expect(ValueError, lambda: x.layer_norm_affine(w, b, -0.0))
expect(ValueError, lambda: x.layer_norm_affine(w, b, -1e-5))
expect(ValueError, lambda: x.layer_norm_affine(w, b, float("inf")))
expect(ValueError, lambda: x.layer_norm_affine(w, b, float("-inf")))
expect(ValueError, lambda: x.layer_norm_affine(w, b, float("nan")))

# The boundary holds even when the variance is zero: a constant input
# still rejects a non-positive eps instead of silently allowing eps=0.
const = Tensor([2.5, 2.5, 2.5, 2.5])
expect(ValueError, lambda: const.layer_norm_affine(w, b, 0.0))
expect(ValueError, lambda: const.layer_norm_affine(w, b, -1.0))

# Any positive finite float, however small, is accepted.
r = const.layer_norm_affine(w, b, 1e-300)
assert r.data == [0.0, 0.0, 0.0, 0.0]
r = x.layer_norm_affine(w, b, 1e-5)
mu = 2.5
var = sum((v - mu) ** 2 for v in [1.0, 2.0, 3.0, 4.0]) / 4
inv = 1.0 / math.sqrt(var + 1e-5)
expected = [(v - mu) * inv for v in [1.0, 2.0, 3.0, 4.0]]
assert all(
    math.isclose(a, e, rel_tol=1e-12, abs_tol=1e-12)
    for a, e in zip(r.data, expected)
)

# A failed eps validation leaves every input untouched.
x2 = Tensor([1.0, 2.0], True)
w2 = Tensor([1.0, 1.0], True)
b2 = Tensor([0.0, 0.0], True)
expect(ValueError, lambda: x2.layer_norm_affine(w2, b2, 0.0))
assert x2.data == [1.0, 2.0] and x2.grad is None
assert w2.data == [1.0, 1.0] and w2.grad is None
assert b2.data == [0.0, 0.0] and b2.grad is None

print("all layer_norm_affine eps boundary tests passed")
