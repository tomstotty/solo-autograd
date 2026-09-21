import math

from autograd import Tensor


def expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no error raised")


# --- forward: other type validation ---
t = Tensor(4.0)
expect(TypeError, lambda: t.div(True))
expect(TypeError, lambda: t.div(2))            # int -> T
expect(TypeError, lambda: t.div("2.0"))
expect(TypeError, lambda: t.div(None))
expect(ValueError, lambda: t.div(float("inf")))
expect(ValueError, lambda: t.div(float("nan")))

# --- forward: data validation (mutated after construction) ---
bad = Tensor(1.0)
bad.data = 1                      # int scalar -> T
expect(TypeError, lambda: bad.div(2.0))
bad.data = [1.0, True]            # bad element -> T
expect(TypeError, lambda: bad.div(2.0))
bad.data = []                     # empty -> V
expect(ValueError, lambda: bad.div(2.0))
bad.data = float("nan")           # non-finite -> V
expect(ValueError, lambda: bad.div(2.0))
bad.data = 2.0
bad.requires_grad = 1             # non-bool -> T
expect(TypeError, lambda: bad.div(2.0))
other_bad = Tensor(2.0)
other_bad.requires_grad = "x"
expect(TypeError, lambda: Tensor(1.0).div(other_bad))

# --- forward: broadcasting / shapes ---
assert Tensor(6.0).div(3.0).data == 2.0
assert Tensor([6.0, 9.0]).div(3.0).data == [2.0, 3.0]
assert Tensor(12.0).div(Tensor([3.0, 4.0])).data == [4.0, 3.0]
assert Tensor([6.0, 9.0]).div(Tensor([3.0, 3.0])).data == [2.0, 3.0]
expect(ValueError, lambda: Tensor([1.0, 2.0]).div(Tensor([1.0, 2.0, 3.0])))
expect(ValueError, lambda: Tensor(1.0).div(0.0))
expect(ValueError, lambda: Tensor(1.0).div(-0.0))
expect(ValueError, lambda: Tensor([1.0, 2.0]).div(Tensor([1.0, -0.0])))
expect(ValueError, lambda: Tensor(1e308).div(1e-308))   # overflow -> V
# failure leaves no state change
x = Tensor([1.0, 1e308])
expect(ValueError, lambda: x.div(Tensor([2.0, 1e-308])))
assert x.data == [1.0, 1e308] and x.grad is None

# --- no graph when neither side requires grad ---
r = Tensor(6.0).div(3.0)
assert r.requires_grad is False and r._parents == ()
expect(ValueError, lambda: r.backward())

# --- backward: scalar / scalar ---
a = Tensor(6.0, True)
b = Tensor(3.0, True)
out = a.div(b)
assert out.data == 2.0 and out.requires_grad is True
out.backward()
assert a.grad == 1.0 / 3.0
assert b.grad == -(1.0 * 6.0) / (3.0 * 3.0)

# --- backward: grad validation ---
a = Tensor(6.0, True); b = Tensor(3.0, True)
out = a.div(b)
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward("x"))
expect(ValueError, lambda: out.backward(float("inf")))
expect(ValueError, lambda: out.backward([1.0]))   # list for scalar -> V
out.backward()                                     # omitted -> 1.0
assert a.grad == 1.0 / 3.0

av = Tensor([6.0, 9.0], True)
bv = Tensor(3.0, True)
outv = av.div(bv)
expect(ValueError, lambda: outv.backward())           # vector omitted -> V
expect(ValueError, lambda: outv.backward(1.0))        # scalar for vector -> V
expect(ValueError, lambda: outv.backward([1.0]))      # wrong length -> V
expect(TypeError, lambda: outv.backward([1.0, 2]))    # int element -> T
expect(ValueError, lambda: outv.backward([1.0, float("nan")]))
outv.backward([1.0, 2.0])
assert av.grad == [1.0 / 3.0, 2.0 / 3.0]
assert bv.grad == -((1.0 * 6.0) / 9.0 + (2.0 * 9.0) / 9.0)

# --- backward: vector numerator, scalar denominator broadcast ---
a = Tensor([6.0, 9.0], True)
b = Tensor(3.0, True)
a.div(b).backward([1.0, 1.0])
assert a.grad == [1.0 / 3.0, 1.0 / 3.0]
assert b.grad == (-(1.0 * 6.0) / 9.0) + (-(1.0 * 9.0) / 9.0)

# --- backward: scalar numerator, vector denominator broadcast ---
a = Tensor(12.0, True)
b = Tensor([3.0, 4.0], True)
a.div(b).backward([1.0, 1.0])
assert a.grad == 1.0 / 3.0 + 1.0 / 4.0
assert b.grad == [-(12.0) / 9.0, -(12.0) / 16.0]

# --- backward: only requires-grad side gets grad ---
a = Tensor(6.0, True)
b = Tensor(3.0)
a.div(b).backward()
assert a.grad == 1.0 / 3.0 and b.grad is None

# --- same tensor on both sides merges ---
x = Tensor(4.0, True)
x.div(x).backward()
assert x.grad == 1.0 / 4.0 - (4.0) / 16.0

# --- snapshot: mutation/replacement after forward does not affect backward ---
a = Tensor([6.0, 9.0], True)
b = Tensor([3.0, 3.0], True)
out = a.div(b)
a.data[0] = 100.0
b.data = [1.0, 1.0]
out.backward([1.0, 1.0])
assert a.grad == [1.0 / 3.0, 1.0 / 3.0]
assert b.grad == [-(6.0) / 9.0, -(9.0) / 9.0]

# --- repeated backward accumulates ---
a = Tensor(6.0, True)
b = Tensor(3.0, True)
a.div(b).backward()
a.div(b).backward()
assert a.grad == 2.0 / 3.0
assert b.grad == 2.0 * (-(6.0) / 9.0)

# --- shared path accumulates ---
x = Tensor(4.0, True)
y = x.div(2.0).mul(x.div(2.0))
y.backward()
assert x.grad == 2.0 * (2.0 * (1.0 / 2.0))

# --- non-finite backward intermediate -> V, grads untouched ---
a = Tensor(1e308, True)
b = Tensor(1e-308, False)
out = a.mul(b)  # graph without div; force huge grad through div below
c = Tensor(1e308, True)
d = Tensor(1e-308, False)
r = c.div(Tensor(2.0, False))
try:
    r.backward(1e308)  # da = 1e308/2.0 fine; use a case that overflows
except ValueError:
    pass
# overflow in da: grad / tiny b
c2 = Tensor(1.0, True)
d2 = Tensor(1e-300, True)
r2 = c2.div(d2)
g = Tensor(1.0).grad
try:
    r2.backward(1e308)  # da = 1e308/1e-300 -> inf
    raise AssertionError("expected ValueError")
except ValueError:
    pass
assert c2.grad is None and d2.grad is None

# --- existing grad merge non-finite -> V, untouched ---
a = Tensor(6.0, True)
a.grad = float("inf")
expect(ValueError, lambda: a.div(Tensor(3.0, True)).backward())
assert a.grad == float("inf")

print("all div tests passed")
