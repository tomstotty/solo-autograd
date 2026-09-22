import math

from autograd import Tensor, gradcheck


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


# --- forward basics ---
r = Tensor([3.0, 4.0]).l2_normalize()
assert len(r.data) == 2
assert math.isclose(r.data[0], 0.6, rel_tol=1e-11)
assert math.isclose(r.data[1], 0.8, rel_tol=1e-11)
assert abs(r.data[0] ** 2 + r.data[1] ** 2 - 1.0) < 1e-11
assert r.requires_grad is False and r._parents == ()

# custom eps
r = Tensor([3.0, 4.0]).l2_normalize(1.0)
inv = 1.0 / math.sqrt(26.0)
assert r.data == [3.0 * inv, 4.0 * inv]

# all-zero vector only works thanks to eps
z = Tensor([0.0, 0.0]).l2_normalize()
assert z.data == [0.0, 0.0]
z = Tensor([0.0, 0.0]).l2_normalize(1e-6)
assert z.data == [0.0, 0.0]

# --- data validation (mutated after construction where needed) ---
expect(ValueError, lambda: Tensor(1.0).l2_normalize())
expect(ValueError, lambda: Tensor([1.0]).l2_normalize(0.0))
bad = Tensor(1.0)
bad.data = 1
expect(ValueError, lambda: bad.l2_normalize())
bad.data = None
expect(ValueError, lambda: bad.l2_normalize())
bad.data = "x"
expect(ValueError, lambda: bad.l2_normalize())
bad.data = []
expect(ValueError, lambda: bad.l2_normalize())
bad.data = [1.0, 2]
expect(TypeError, lambda: bad.l2_normalize())
bad.data = [1.0, True]
expect(TypeError, lambda: bad.l2_normalize())
bad.data = [1.0, "x"]
expect(TypeError, lambda: bad.l2_normalize())
bad.data = [1.0, float("inf")]
expect(ValueError, lambda: bad.l2_normalize())
bad.data = [float("nan"), 1.0]
expect(ValueError, lambda: bad.l2_normalize())
bad.data = [1.0, 2.0]
bad.requires_grad = 1
expect(TypeError, lambda: bad.l2_normalize())
bad.requires_grad = "x"
expect(TypeError, lambda: bad.l2_normalize())

# --- eps validation ---
x = Tensor([1.0, 2.0])
expect(TypeError, lambda: x.l2_normalize(1))
expect(TypeError, lambda: x.l2_normalize(True))
expect(TypeError, lambda: x.l2_normalize("1e-12"))
expect(TypeError, lambda: x.l2_normalize(None))
expect(ValueError, lambda: x.l2_normalize(0.0))
expect(ValueError, lambda: x.l2_normalize(-1.0))
expect(ValueError, lambda: x.l2_normalize(float("inf")))
expect(ValueError, lambda: x.l2_normalize(float("nan")))

# --- non-finite forward intermediate -> V, input untouched ---
huge = Tensor([1e308, 1e308], True)
expect(ValueError, lambda: huge.l2_normalize())
assert huge.data == [1e308, 1e308] and huge.grad is None

# --- graph construction ---
assert Tensor([1.0, 2.0], False).l2_normalize().requires_grad is False
rg = Tensor([1.0, 2.0], True).l2_normalize()
assert rg.requires_grad is True
assert rg._parents == (rg._backward_fn.__closure__[0].cell_contents,) or len(
    rg._parents
) == 1

# --- backward: grad validation ---
out = Tensor([3.0, 4.0], True).l2_normalize()
expect(ValueError, lambda: out.backward())
expect(ValueError, lambda: out.backward(1.0))
expect(ValueError, lambda: out.backward([1.0]))
expect(ValueError, lambda: out.backward([1.0, 2.0, 3.0]))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward("x"))
expect(TypeError, lambda: out.backward([1.0, 2]))
expect(TypeError, lambda: out.backward([1.0, None]))
expect(ValueError, lambda: out.backward([1.0, float("inf")]))
expect(ValueError, lambda: out.backward([float("nan"), 1.0]))

# --- backward: value check against the closed form ---
x = Tensor([3.0, 4.0], True)
out = x.l2_normalize()
g = [0.7, -1.3]
out.backward(list(g))
A = 25.0 + 1e-12
r = 1.0 / math.sqrt(A)
y = [3.0 * r, 4.0 * r]
C = g[0] * y[0] + g[1] * y[1]
expected = [r * (g[0] - y[0] * C), r * (g[1] - y[1] * C)]
assert all(
    math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    for a, b in zip(x.grad, expected)
)

# --- snapshot: mutation after forward does not affect backward ---
x = Tensor([3.0, 4.0], True)
out = x.l2_normalize()
x.data[0] = 100.0
x.data = [9.0, 9.0]
out.backward(list(g))
assert all(
    math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    for a, b in zip(x.grad, expected)
)

# --- repeated backward accumulates ---
x = Tensor([3.0, 4.0], True)
out = x.l2_normalize()
out.backward(list(g))
out.backward(list(g))
assert all(
    math.isclose(a, 2.0 * b, rel_tol=1e-12, abs_tol=1e-12)
    for a, b in zip(x.grad, expected)
)

# --- shared path accumulates ---
x = Tensor([1.5, -2.0, 0.5], True)
n = x.l2_normalize()
n.sum().add(n.sum()).backward()
ref = Tensor([1.5, -2.0, 0.5], True)
ref.l2_normalize().sum().add(ref.l2_normalize().sum()).backward()
assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
           for a, b in zip(x.grad, ref.grad))

# --- non-finite backward intermediate -> V, grads untouched ---
# Four equal entries: y_i = 0.5, C = 4 * 1e308 * 0.5 = 2e308 -> inf.
x = Tensor([1.0, 1.0, 1.0, 1.0], True)
out = x.l2_normalize()
expect(ValueError, lambda: out.backward([1e308] * 4))
assert x.grad is None

# --- existing grad merge non-finite -> V, untouched ---
x = Tensor([1.0, 2.0], True)
x.grad = [float("inf"), 0.0]
expect(ValueError, lambda: x.l2_normalize().backward([1.0, 1.0]))
assert x.grad == [float("inf"), 0.0]

# --- gradcheck numeric agreement ---
passed, err = gradcheck(lambda t: t.l2_normalize().sum(), [1.5, -2.0, 0.5])
assert passed, err
passed, err = gradcheck(lambda t: t.l2_normalize(0.5).sum(), [0.3, -1.2])
assert passed, err
# a non-sum scalar reduction of the normalized output
passed, err = gradcheck(
    lambda t: t.l2_normalize().mul(Tensor([2.0, -1.0, 0.5])).sum(),
    [1.5, -2.0, 0.5],
)
assert passed, err
# near-zero input
passed, err = gradcheck(lambda t: t.l2_normalize(1e-4).sum(), [1e-6, -2e-6])
assert passed, err

print("all l2_normalize tests passed")
