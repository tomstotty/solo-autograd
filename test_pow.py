import math

from autograd import Tensor, gradcheck


def expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no error raised")


# --- forward: exponent type validation ---
t = Tensor(4.0)
expect(TypeError, lambda: t.pow(True))
expect(TypeError, lambda: t.pow(2))            # int -> T
expect(TypeError, lambda: t.pow("2.0"))
expect(TypeError, lambda: t.pow(None))
expect(ValueError, lambda: t.pow(float("inf")))
expect(ValueError, lambda: t.pow(float("nan")))

# --- forward: data validation (mutated after construction) ---
bad = Tensor(2.0)
bad.data = 1                      # int scalar -> T
expect(TypeError, lambda: bad.pow(2.0))
bad.data = [2.0, True]            # bad element -> T
expect(TypeError, lambda: bad.pow(2.0))
bad.data = []                     # empty -> V
expect(ValueError, lambda: bad.pow(2.0))
bad.data = float("nan")           # non-finite -> V
expect(ValueError, lambda: bad.pow(2.0))
bad.data = 2.0
bad.requires_grad = 1             # non-bool -> T
expect(TypeError, lambda: bad.pow(2.0))
other_bad = Tensor(2.0)
other_bad.requires_grad = "x"
expect(TypeError, lambda: Tensor(2.0).pow(other_bad))
exp_bad = Tensor(2.0)
exp_bad.data = [2.0, 1]           # bad exponent element -> T
expect(TypeError, lambda: Tensor(2.0).pow(exp_bad))
exp_bad.data = float("inf")       # non-finite exponent data -> V
expect(ValueError, lambda: Tensor(2.0).pow(exp_bad))

# --- forward: broadcasting / shapes ---
assert Tensor(4.0).pow(2.0).data == 16.0
assert Tensor([2.0, 3.0]).pow(2.0).data == [4.0, 9.0]
assert Tensor(2.0).pow(Tensor([2.0, 3.0])).data == [4.0, 8.0]
assert Tensor([2.0, 3.0]).pow(Tensor([3.0, 2.0])).data == [8.0, 9.0]
expect(ValueError, lambda: Tensor([2.0, 3.0]).pow(Tensor([1.0, 2.0, 3.0])))
expect(ValueError, lambda: Tensor(0.0).pow(2.0))      # base 0 -> V
expect(ValueError, lambda: Tensor(-0.0).pow(2.0))     # base -0.0 -> V
expect(ValueError, lambda: Tensor(-1.0).pow(2.0))     # negative base -> V
expect(ValueError, lambda: Tensor([2.0, -1.0]).pow(2.0))
expect(ValueError, lambda: Tensor(1e308).pow(2.0))    # overflow -> V
# failure leaves no state change
x = Tensor([2.0, 1e308])
expect(ValueError, lambda: x.pow(Tensor([2.0, 2.0])))
assert x.data == [2.0, 1e308] and x.grad is None

# --- no graph when neither side requires grad ---
r = Tensor(4.0).pow(2.0)
assert r.requires_grad is False and r._parents == ()
expect(ValueError, lambda: r.backward())

# --- backward: scalar / scalar ---
a = Tensor(4.0, True)
b = Tensor(3.0, True)
out = a.pow(b)
assert out.data == 64.0 and out.requires_grad is True
out.backward()
assert a.grad == 1.0 * 3.0 * 4.0**2.0
assert b.grad == 1.0 * 64.0 * math.log(4.0)

# --- backward: grad validation ---
a = Tensor(4.0, True); b = Tensor(3.0, True)
out = a.pow(b)
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward("x"))
expect(ValueError, lambda: out.backward(float("inf")))
expect(ValueError, lambda: out.backward([1.0]))   # list for scalar -> V
out.backward()                                     # omitted -> 1.0
assert a.grad == 1.0 * 3.0 * 4.0**2.0

av = Tensor([2.0, 3.0], True)
bv = Tensor(2.0, True)
outv = av.pow(bv)
expect(ValueError, lambda: outv.backward())           # vector omitted -> V
expect(ValueError, lambda: outv.backward(1.0))        # scalar for vector -> V
expect(ValueError, lambda: outv.backward([1.0]))      # wrong length -> V
expect(TypeError, lambda: outv.backward([1.0, 2]))    # int element -> T
expect(ValueError, lambda: outv.backward([1.0, float("nan")]))
outv.backward([1.0, 2.0])
assert av.grad == [1.0 * 2.0 * 2.0**1.0, 2.0 * 2.0 * 3.0**1.0]
assert bv.grad == (
    1.0 * 4.0 * math.log(2.0) + 2.0 * 9.0 * math.log(3.0)
)

# --- backward: vector base, scalar exponent broadcast ---
a = Tensor([2.0, 3.0], True)
b = Tensor(2.0, True)
a.pow(b).backward([1.0, 1.0])
assert a.grad == [1.0 * 2.0 * 2.0, 1.0 * 2.0 * 3.0]
assert b.grad == (1.0 * 4.0 * math.log(2.0)) + (1.0 * 9.0 * math.log(3.0))

# --- backward: scalar base, vector exponent broadcast ---
a = Tensor(2.0, True)
b = Tensor([2.0, 3.0], True)
a.pow(b).backward([1.0, 1.0])
assert a.grad == 1.0 * 2.0 * 2.0**1.0 + 1.0 * 3.0 * 2.0**2.0
assert b.grad == [1.0 * 4.0 * math.log(2.0), 1.0 * 8.0 * math.log(2.0)]

# --- backward: only requires-grad side gets grad ---
a = Tensor(4.0, True)
b = Tensor(3.0)
a.pow(b).backward()
assert a.grad == 1.0 * 3.0 * 4.0**2.0 and b.grad is None

# --- same tensor on both sides merges ---
x = Tensor(2.0, True)
x.pow(x).backward()
assert x.grad == 1.0 * 2.0 * 2.0**1.0 + 1.0 * 4.0 * math.log(2.0)

# --- snapshot: mutation/replacement after forward does not affect backward ---
a = Tensor([2.0, 3.0], True)
b = Tensor([2.0, 2.0], True)
out = a.pow(b)
a.data[0] = 100.0
b.data = [1.0, 1.0]
out.backward([1.0, 1.0])
assert a.grad == [1.0 * 2.0 * 2.0, 1.0 * 2.0 * 3.0]
assert b.grad == [1.0 * 4.0 * math.log(2.0), 1.0 * 9.0 * math.log(3.0)]

# --- repeated backward accumulates ---
a = Tensor(4.0, True)
b = Tensor(3.0, True)
a.pow(b).backward()
a.pow(b).backward()
assert a.grad == 2.0 * (1.0 * 3.0 * 4.0**2.0)
assert b.grad == 2.0 * (1.0 * 64.0 * math.log(4.0))

# --- shared path accumulates ---
x = Tensor(4.0, True)
y = x.pow(2.0).mul(x.pow(2.0))
y.backward()
assert x.grad == 2.0 * (16.0 * (1.0 * 2.0 * 4.0))

# --- non-finite backward intermediate -> V, grads untouched ---
# overflow in db: g * y overflows
c2 = Tensor(2.0, True)
d2 = Tensor(3.0, True)
r2 = c2.pow(d2)
try:
    r2.backward(1e308)  # db = 1e308 * 8.0 * log(2) -> inf
    raise AssertionError("expected ValueError")
except ValueError:
    pass
assert c2.grad is None and d2.grad is None

# overflow in da: g * b * a**(b-1) overflows
c4 = Tensor(2.0, True)
r4 = c4.pow(Tensor(3.0, False))
try:
    r4.backward(1e308)  # da = 1e308 * 3.0 * 4.0 -> inf
    raise AssertionError("expected ValueError")
except ValueError:
    pass
assert c4.grad is None

# --- existing grad merge non-finite -> V, untouched ---
a = Tensor(4.0, True)
a.grad = float("inf")
expect(ValueError, lambda: a.pow(Tensor(3.0, True)).backward())
assert a.grad == float("inf")

# --- gradcheck against central differences ---
passed, err = gradcheck(lambda t: t.pow(2.5), 3.0)
assert passed, err
passed, err = gradcheck(lambda t: t.pow(2.5).sum(), [1.5, 2.5])
assert passed, err
passed, err = gradcheck(lambda t: Tensor(2.0, True).pow(t).sum(), [1.5, 2.5])
assert passed, err

print("all pow tests passed")
