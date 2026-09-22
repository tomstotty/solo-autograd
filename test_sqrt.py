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


# --- relu retained: requires_grad must be a bool at call time ---
t = Tensor(-2.0)
t.requires_grad = 1
expect(TypeError, t.relu)
t.requires_grad = "x"
expect(TypeError, t.relu)
t.requires_grad = False
assert t.relu().data == 0.0
assert Tensor(3.0, True).relu().backward() is None
# rest of relu behavior unchanged
assert Tensor([-1.0, 0.0, 2.0]).relu().data == [0.0, 0.0, 2.0]

# --- forward: data validation (mutated after construction) ---
bad = Tensor(1.0)
bad.data = 1
expect(TypeError, bad.sqrt)
bad.data = True
expect(TypeError, bad.sqrt)
bad.data = "1.0"
expect(TypeError, bad.sqrt)
bad.data = None
expect(TypeError, bad.sqrt)
bad.data = [[1.0]]
expect(TypeError, bad.sqrt)  # nested list
bad.data = [1.0, True]
expect(TypeError, bad.sqrt)
bad.data = [1.0, 2]
expect(TypeError, bad.sqrt)
bad.data = []
expect(ValueError, bad.sqrt)
bad.data = float("nan")
expect(ValueError, bad.sqrt)
bad.data = float("inf")
expect(ValueError, bad.sqrt)
bad.data = [1.0, float("nan")]
expect(ValueError, bad.sqrt)
bad.data = 2.0
bad.requires_grad = 1
expect(TypeError, bad.sqrt)
bad.requires_grad = False

# --- forward: x must be non-negative; zero is allowed ---
assert Tensor(0.0).sqrt().data == 0.0
assert Tensor(4.0).sqrt().data == 2.0
assert Tensor([0.0, 1.0, 9.0, 16.0]).sqrt().data == [0.0, 1.0, 3.0, 4.0]
expect(ValueError, lambda: Tensor(-0.5).sqrt())
expect(ValueError, lambda: Tensor(-1e-300).sqrt())
expect(ValueError, lambda: Tensor([1.0, -1.0]).sqrt())
expect(ValueError, lambda: Tensor([-1.0, 1.0]).sqrt())
# failure leaves the input untouched
x = Tensor([1.0, -2.0, 3.0])
expect(ValueError, x.sqrt)
assert x.data == [1.0, -2.0, 3.0] and x.grad is None
x = Tensor(-1.0, True)
expect(ValueError, x.sqrt)
assert x.data == -1.0 and x.grad is None

# --- no graph without requires_grad; one-parent graph otherwise ---
r = Tensor(4.0).sqrt()
assert r.data == 2.0 and r.requires_grad is False and r._parents == ()
expect(ValueError, r.backward)
r2 = Tensor(4.0, True).sqrt()
assert r2.requires_grad is True and len(r2._parents) == 1

# --- backward: scalar ---
a = Tensor(4.0, True)
out = a.sqrt()
assert out.data == 2.0
out.backward()
# dx = 1 / (2*2) = 0.25
assert a.grad == 0.25

# --- backward: grad validation ---
a = Tensor(4.0, True)
out = a.sqrt()
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward("x"))
expect(ValueError, lambda: out.backward(float("inf")))
expect(ValueError, lambda: out.backward(float("nan")))
expect(ValueError, lambda: out.backward([1.0]))
out.backward(4.0)
assert a.grad == 1.0

av = Tensor([1.0, 9.0, 16.0], True)
outv = av.sqrt()
expect(ValueError, outv.backward)  # vector needs explicit grad
expect(ValueError, lambda: outv.backward(1.0))
expect(ValueError, lambda: outv.backward([1.0, 1.0]))
expect(ValueError, lambda: outv.backward([1.0, 1.0, 1.0, 1.0]))
expect(TypeError, lambda: outv.backward([1.0, 2, 1.0]))
expect(TypeError, lambda: outv.backward([1.0, None, 1.0]))
expect(ValueError, lambda: outv.backward([1.0, float("nan"), 1.0]))
outv.backward([2.0, 6.0, 8.0])
# dx = g/(2*y): [2/2, 6/6, 8/8]
assert av.grad == [1.0, 1.0, 1.0]

# --- backward: zero input is not differentiable ---
z = Tensor(0.0, True)
expect(ValueError, z.sqrt().backward)
assert z.grad is None
zv = Tensor([0.0, 1.0], True)
expect(ValueError, lambda: zv.sqrt().backward([1.0, 1.0]))
assert zv.grad is None
# failure at a later index still changes nothing
zv = Tensor([4.0, 0.0, 9.0], True)
expect(ValueError, lambda: zv.sqrt().backward([1.0, 1.0, 1.0]))
assert zv.grad is None

# --- backward: non-finite intermediate (overflow), grads untouched ---
# Tiny x makes the denominator 2*y tiny, so g/(2*y) overflows.
tiny = Tensor(1e-300, True)
rb = tiny.sqrt()
assert rb.data == math.sqrt(1e-300)
expect(ValueError, lambda: rb.backward(1e308))
assert tiny.grad is None

# --- snapshot: mutation/replacement after forward does not affect backward ---
a = Tensor([1.0, 9.0], True)
out = a.sqrt()
a.data[0] = 100.0
a.data = [100.0, 100.0]
out.backward([2.0, 6.0])
assert a.grad == [1.0, 1.0]

a = Tensor(4.0, True)
out = a.sqrt()
a.data = 100.0
out.backward(4.0)
assert a.grad == 1.0

# replacing data with zero after forward still raises from the snapshot
a = Tensor(4.0, True)
out = a.sqrt()
a.data = 0.0
out.backward(1.0)
assert a.grad == 0.25

# --- repeated backward accumulates ---
a = Tensor(4.0, True)
out = a.sqrt()
out.backward(1.0)
out.backward(1.0)
assert a.grad == 0.5
av = Tensor([1.0, 9.0], True)
outv = av.sqrt()
outv.backward([2.0, 6.0])
outv.backward([2.0, 6.0])
assert av.grad == [2.0, 2.0]

# --- shared path accumulates: y = sqrt(x) * sqrt(x) = x, dy/dx = 1 ---
x = Tensor(4.0, True)
y = x.sqrt().mul(x.sqrt())
y.backward()
assert x.grad == 1.0

# --- alias: same child used twice through a shared node ---
x = Tensor(4.0, True)
s = x.sqrt()
y = s.mul(s)
y.backward()
assert x.grad == 1.0

# --- existing grad merge non-finite -> ValueError, untouched ---
a = Tensor(4.0, True)
a.grad = float("inf")
expect(ValueError, lambda: a.sqrt().backward(1.0))
assert a.grad == float("inf")

# --- gradcheck numeric agreement ---
passed, err = gradcheck(lambda t: t.sqrt().sum(), [0.25, 1.0, 4.0, 25.0])
assert passed, err
passed, err = gradcheck(lambda t: t.sqrt(), 2.0)
assert passed, err

print("all sqrt tests passed")
