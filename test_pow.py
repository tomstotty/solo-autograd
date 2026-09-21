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


# --- forward: exponent type validation ---
t = Tensor(2.0)
expect(TypeError, lambda: t.pow(True))
expect(TypeError, lambda: t.pow(2))          # int -> T
expect(TypeError, lambda: t.pow("2.0"))
expect(TypeError, lambda: t.pow(None))
expect(TypeError, lambda: t.pow([2.0]))
expect(ValueError, lambda: t.pow(float("inf")))
expect(ValueError, lambda: t.pow(float("nan")))

# --- forward: data validation (mutated after construction) ---
bad = Tensor(1.0)
bad.data = 1
expect(TypeError, lambda: bad.pow(2.0))
bad.data = [1.0, True]
expect(TypeError, lambda: bad.pow(2.0))
bad.data = []
expect(ValueError, lambda: bad.pow(2.0))
bad.data = float("nan")
expect(ValueError, lambda: bad.pow(2.0))
bad.data = 2.0
bad.requires_grad = 1
expect(TypeError, lambda: bad.pow(2.0))
exp_bad = Tensor(2.0)
exp_bad.requires_grad = "x"
expect(TypeError, lambda: Tensor(1.0).pow(exp_bad))
# bad data on the exponent tensor side
eb = Tensor(2.0)
eb.data = 1
expect(TypeError, lambda: Tensor(2.0).pow(eb))
eb.data = []
expect(ValueError, lambda: Tensor(2.0).pow(eb))
eb.data = float("inf")
expect(ValueError, lambda: Tensor(2.0).pow(eb))

# --- forward: base must be positive ---
expect(ValueError, lambda: Tensor(0.0).pow(2.0))
expect(ValueError, lambda: Tensor(-1.0).pow(2.0))
expect(ValueError, lambda: Tensor([1.0, 0.0]).pow(2.0))
expect(ValueError, lambda: Tensor([1.0, -2.0]).pow(2.0))
expect(ValueError, lambda: Tensor(0.0).pow(Tensor([1.0, 2.0])))
expect(ValueError, lambda: Tensor(-1.0).pow(Tensor([1.0, 2.0])))

# --- forward: broadcasting / shapes ---
assert Tensor(2.0).pow(3.0).data == 8.0
assert Tensor([2.0, 3.0]).pow(2.0).data == [4.0, 9.0]
assert Tensor(2.0).pow(Tensor([3.0, 4.0])).data == [8.0, 16.0]
assert Tensor([2.0, 3.0]).pow(Tensor([3.0, 2.0])).data == [8.0, 9.0]
expect(
    ValueError,
    lambda: Tensor([1.0, 2.0]).pow(Tensor([1.0, 2.0, 3.0])),
)
# overflow / non-finite result -> V
expect(ValueError, lambda: Tensor(1e308).pow(100.0))
# failure leaves no state change
x = Tensor([1.0, 2.0])
expect(ValueError, lambda: x.pow(Tensor([1.0, 10000.0])))
assert x.data == [1.0, 2.0] and x.grad is None

# --- no graph when neither side requires grad ---
r = Tensor(2.0).pow(3.0)
assert r.requires_grad is False and r._parents == ()
expect(ValueError, lambda: r.backward())
# float exponent with a requires-grad base still builds a one-parent graph
r2 = Tensor(2.0, True).pow(3.0)
assert r2.requires_grad is True

# --- backward: scalar / scalar ---
a = Tensor(2.0, True)
b = Tensor(3.0, True)
out = a.pow(b)
assert out.data == 8.0 and out.requires_grad is True
out.backward()
# da = b*a^(b-1) = 3*4 ; db = y*log(a) = 8*log2
assert a.grad == 3.0 * 2.0 ** 2.0
assert b.grad == 8.0 * math.log(2.0)

# --- backward: grad validation ---
a = Tensor(2.0, True)
b = Tensor(3.0, True)
out = a.pow(b)
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward("x"))
expect(ValueError, lambda: out.backward(float("inf")))
expect(ValueError, lambda: out.backward([1.0]))
out.backward()
assert a.grad == 3.0 * 4.0

av = Tensor([2.0, 3.0], True)
bv = Tensor(2.0, True)
outv = av.pow(bv)
expect(ValueError, lambda: outv.backward())
expect(ValueError, lambda: outv.backward(1.0))
expect(ValueError, lambda: outv.backward([1.0]))
expect(TypeError, lambda: outv.backward([1.0, 2]))
expect(ValueError, lambda: outv.backward([1.0, float("nan")]))
outv.backward([1.0, 1.0])
assert av.grad == [2.0 * 2.0, 2.0 * 3.0]
assert bv.grad == 4.0 * math.log(2.0) + 9.0 * math.log(3.0)

# --- backward: vector base, scalar exponent broadcast ---
a = Tensor([2.0, 3.0], True)
b = Tensor(2.0, True)
a.pow(b).backward([1.0, 1.0])
assert a.grad == [2.0 * 2.0, 2.0 * 3.0]
assert b.grad == 4.0 * math.log(2.0) + 9.0 * math.log(3.0)

# --- backward: scalar base, vector exponent broadcast ---
a = Tensor(2.0, True)
b = Tensor([3.0, 4.0], True)
a.pow(b).backward([1.0, 1.0])
assert a.grad == 3.0 * 2.0 ** 2.0 + 4.0 * 2.0 ** 3.0
assert b.grad == [8.0 * math.log(2.0), 16.0 * math.log(2.0)]

# --- only requires-grad side gets grad ---
a = Tensor(2.0, True)
b = Tensor(3.0)
a.pow(b).backward()
assert a.grad == 3.0 * 4.0 and b.grad is None

a = Tensor(2.0)
b = Tensor(3.0, True)
a.pow(b).backward()
assert a.grad is None and b.grad == 8.0 * math.log(2.0)

# float exponent with requires-grad base
a = Tensor(2.0, True)
a.pow(3.0).backward()
assert a.grad == 12.0

# --- same tensor on both sides merges: d(a^a)/da = a^a*(log a + 1) ---
x = Tensor(2.0, True)
x.pow(x).backward()
assert x.grad == 4.0 * (math.log(2.0) + 1.0)

# --- snapshot: mutation/replacement after forward does not affect backward ---
a = Tensor([2.0, 3.0], True)
b = Tensor([2.0, 2.0], True)
out = a.pow(b)
a.data[0] = 100.0
b.data = [9.0, 9.0]
out.backward([1.0, 1.0])
assert a.grad == [2.0 * 2.0, 2.0 * 3.0]
assert b.grad == [4.0 * math.log(2.0), 9.0 * math.log(3.0)]

# replacing the input object's data wholesale after forward
a = Tensor(2.0, True)
b = Tensor(3.0, True)
out = a.pow(b)
a.data = 100.0
b.data = 9.0
out.backward()
assert a.grad == 12.0
assert b.grad == 8.0 * math.log(2.0)

# --- repeated backward accumulates ---
a = Tensor(2.0, True)
b = Tensor(3.0, True)
a.pow(b).backward()
a.pow(b).backward()
assert a.grad == 2.0 * 12.0
assert b.grad == 2.0 * 8.0 * math.log(2.0)

# --- shared path accumulates ---
x = Tensor(2.0, True)
y = x.pow(2.0).mul(x.pow(2.0))
y.backward()
# y = x^4, dy/dx = 4 x^3
assert x.grad == 4.0 * 8.0

# --- non-finite backward intermediate -> V, grads untouched ---
# da path: forward finite (1e100^2 = 1e200), but g*b*a^(b-1) overflows.
c = Tensor(1e100, True)
d = Tensor(2.0, True)
r = c.pow(d)
try:
    r.backward(1e308)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
assert c.grad is None and d.grad is None

# db path: with exponent 1, da = g is finite but g*y*log(a) overflows.
c = Tensor(10.0, True)
d = Tensor(1.0, True)
r = c.pow(d)
assert r.data == 10.0
try:
    r.backward(1e308)
    raise AssertionError("expected ValueError")
except ValueError:
    pass
assert c.grad is None and d.grad is None

# --- existing grad merge non-finite -> V, untouched ---
a = Tensor(2.0, True)
a.grad = float("inf")
expect(ValueError, lambda: a.pow(Tensor(3.0, True)).backward())
assert a.grad == float("inf")

# --- gradcheck numeric agreement ---
passed, err = gradcheck(lambda t: t.pow(2.3).sum(), [1.5, 2.5])
assert passed, err
passed, err = gradcheck(lambda t: t.pow(1.7), 2.0)
assert passed, err
# exponent-side gradient
passed, err = gradcheck(lambda e: Tensor(2.0).pow(e).sum(), [0.5, 1.5])
assert passed, err
# same tensor both sides
passed, err = gradcheck(lambda t: t.pow(t).sum(), [1.5, 2.5])
assert passed, err

print("all pow tests passed")
