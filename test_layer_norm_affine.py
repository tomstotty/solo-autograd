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


# Regression coverage for the layer_norm_affine eps boundary contract:
# only a positive finite float is legal. The contract must not be loosened.
x = Tensor([1.0, 2.0, 3.0, 4.0])
w = Tensor([1.0, 1.0, 1.0, 1.0])
b = Tensor([0.0, 0.0, 0.0, 0.0])

# The default eps is accepted; the boundary itself is legal.
assert x.layer_norm_affine(w, b).requires_grad is False
edge = x.layer_norm_affine(w, b, 1e-300)
assert all(math.isfinite(v) for v in edge.data)
big = x.layer_norm_affine(w, b, 1e300)
assert all(math.isfinite(v) for v in big.data)

# Non-float eps (including bool and int) is a TypeError.
expect(TypeError, lambda: x.layer_norm_affine(w, b, True))
expect(TypeError, lambda: x.layer_norm_affine(w, b, False))
expect(TypeError, lambda: x.layer_norm_affine(w, b, 1))
expect(TypeError, lambda: x.layer_norm_affine(w, b, 0))
expect(TypeError, lambda: x.layer_norm_affine(w, b, "1e-5"))
expect(TypeError, lambda: x.layer_norm_affine(w, b, None))

# Non-finite eps or eps <= 0.0 is a ValueError; zero is not legal.
expect(ValueError, lambda: x.layer_norm_affine(w, b, 0.0))
expect(ValueError, lambda: x.layer_norm_affine(w, b, -0.0))
expect(ValueError, lambda: x.layer_norm_affine(w, b, -1e-5))
expect(ValueError, lambda: x.layer_norm_affine(w, b, float("inf")))
expect(ValueError, lambda: x.layer_norm_affine(w, b, float("-inf")))
expect(ValueError, lambda: x.layer_norm_affine(w, b, float("nan")))

# A rejected eps aborts before the result exists: inputs stay unchanged and
# no graph is produced.
xr = Tensor([1.0, 2.0, 3.0, 4.0], True)
before = list(xr.data)
expect(ValueError, lambda: xr.layer_norm_affine(w, b, 0.0))
assert xr.data == before and xr.grad is None
expect(TypeError, lambda: xr.layer_norm_affine(w, b, False))
assert xr.data == before and xr.grad is None

# The same boundary holds through a grad-enabled graph.
xr = Tensor([1.0, 2.0, 3.0, 4.0], True)
wr = Tensor([1.0, 1.0, 1.0, 1.0], True)
br = Tensor([0.0, 0.0, 0.0, 0.0], True)
expect(ValueError, lambda: xr.layer_norm_affine(wr, br, 0.0))
expect(TypeError, lambda: xr.layer_norm_affine(wr, br, True))

print("all layer_norm_affine eps boundary tests passed")
