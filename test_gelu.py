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


def gelu_ref(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_grad_ref(x):
    a = 1.0 + math.erf(x / math.sqrt(2.0))
    r = math.exp(-0.5 * x * x)
    return 0.5 * a + x * r / math.sqrt(2.0 * math.pi)


# --- requires_grad must be a bool at call time ---
t = Tensor(-2.0)
t.requires_grad = 1
expect(TypeError, t.gelu)
t.requires_grad = "x"
expect(TypeError, t.gelu)
t.requires_grad = False
assert t.gelu().data == gelu_ref(-2.0)
assert Tensor(3.0, True).gelu().backward() is None

# --- forward: data validation (mutated after construction) ---
bad = Tensor(1.0)
bad.data = 1
expect(TypeError, bad.gelu)
bad.data = True
expect(TypeError, bad.gelu)
bad.data = "1.0"
expect(TypeError, bad.gelu)
bad.data = None
expect(TypeError, bad.gelu)
bad.data = [[1.0]]
expect(TypeError, bad.gelu)  # nested list
bad.data = [1.0, True]
expect(TypeError, bad.gelu)
bad.data = [1.0, 2]
expect(TypeError, bad.gelu)
bad.data = []
expect(ValueError, bad.gelu)
bad.data = float("nan")
expect(ValueError, bad.gelu)
bad.data = float("inf")
expect(ValueError, bad.gelu)
bad.data = [1.0, float("nan")]
expect(ValueError, bad.gelu)
bad.data = 2.0
bad.requires_grad = 1
expect(TypeError, bad.gelu)
bad.requires_grad = False

# --- forward: exact values ---
assert Tensor(0.0).gelu().data == 0.0
assert Tensor(1.0).gelu().data == 0.8413447460685429
assert Tensor(-1.0).gelu().data == -0.15865525393145707
out = Tensor([0.0, 1.0, -1.0, 2.5]).gelu()
assert out.data == [gelu_ref(x) for x in [0.0, 1.0, -1.0, 2.5]]
# huge finite inputs stay finite and saturate
assert Tensor(1e200).gelu().data == 1e200
assert Tensor(-1e200).gelu().data == 0.0

# --- no graph without requires_grad; one-parent graph otherwise ---
r = Tensor(4.0).gelu()
assert r.requires_grad is False and r._parents == ()
expect(ValueError, r.backward)
r2 = Tensor(4.0, True).gelu()
assert r2.requires_grad is True and len(r2._parents) == 1

# --- backward: scalar ---
a = Tensor(0.0, True)
out = a.gelu()
assert out.data == 0.0
out.backward()
# d = 0.5*(1+erf(0)) + 0 = 0.5
assert a.grad == 0.5

a = Tensor(1.0, True)
a.gelu().backward()
assert a.grad == gelu_grad_ref(1.0)

# --- backward: grad validation ---
a = Tensor(1.0, True)
out = a.gelu()
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward("x"))
expect(ValueError, lambda: out.backward(float("inf")))
expect(ValueError, lambda: out.backward(float("nan")))
expect(ValueError, lambda: out.backward([1.0]))
out.backward(2.0)
assert a.grad == 2.0 * gelu_grad_ref(1.0)

av = Tensor([0.0, 1.0, -1.0], True)
outv = av.gelu()
expect(ValueError, outv.backward)  # vector needs explicit grad
expect(ValueError, lambda: outv.backward(1.0))
expect(ValueError, lambda: outv.backward([1.0, 1.0]))
expect(ValueError, lambda: outv.backward([1.0, 1.0, 1.0, 1.0]))
expect(TypeError, lambda: outv.backward([1.0, 2, 1.0]))
expect(TypeError, lambda: outv.backward([1.0, None, 1.0]))
expect(ValueError, lambda: outv.backward([1.0, float("nan"), 1.0]))
outv.backward([1.0, 2.0, 3.0])
assert av.grad == [
    1.0 * gelu_grad_ref(0.0),
    2.0 * gelu_grad_ref(1.0),
    3.0 * gelu_grad_ref(-1.0),
]

# --- backward: non-finite intermediate (q = x*x overflows), untouched ---
big = Tensor(1e200, True)
rb = big.gelu()
assert rb.data == 1e200
expect(ValueError, lambda: rb.backward(1.0))
assert big.grad is None
# failure at a later index still changes nothing
bv = Tensor([1.0, 1e200], True)
expect(ValueError, lambda: bv.gelu().backward([1.0, 1.0]))
assert bv.grad is None
# dx overflow: huge grad times a finite derivative
c = Tensor(1.0, True)
expect(ValueError, lambda: c.gelu().backward(1.7e308))
assert c.grad is None

# --- snapshot: mutation/replacement after forward does not affect backward ---
a = Tensor([0.0, 1.0], True)
out = a.gelu()
a.data[0] = 100.0
a.data = [100.0, 100.0]
out.backward([1.0, 1.0])
assert a.grad == [gelu_grad_ref(0.0), gelu_grad_ref(1.0)]

a = Tensor(1.0, True)
out = a.gelu()
a.data = 100.0
out.backward(2.0)
assert a.grad == 2.0 * gelu_grad_ref(1.0)

# --- repeated backward accumulates ---
a = Tensor(1.0, True)
out = a.gelu()
out.backward(1.0)
out.backward(1.0)
assert a.grad == 2.0 * gelu_grad_ref(1.0)
av = Tensor([0.0, 1.0], True)
outv = av.gelu()
outv.backward([1.0, 2.0])
outv.backward([1.0, 2.0])
assert av.grad == [2.0 * gelu_grad_ref(0.0), 4.0 * gelu_grad_ref(1.0)]

# --- shared path accumulates: y = gelu(x) + gelu(x), dy/dx = 2*d ---
x = Tensor(1.0, True)
y = x.gelu().add(x.gelu())
y.backward()
assert x.grad == 2.0 * gelu_grad_ref(1.0)

# --- alias: same child used twice through a shared node ---
x = Tensor(1.0, True)
g = x.gelu()
y = g.mul(g)
y.backward()
assert x.grad == 2.0 * gelu_ref(1.0) * gelu_grad_ref(1.0)

# --- existing grad merge non-finite -> ValueError, untouched ---
a = Tensor(1.0, True)
a.grad = float("inf")
expect(ValueError, lambda: a.gelu().backward(1.0))
assert a.grad == float("inf")

# --- gradcheck numeric agreement ---
passed, err = gradcheck(lambda t: t.gelu().sum(), [0.25, 1.0, -2.0, 4.0])
assert passed, err
passed, err = gradcheck(lambda t: t.gelu(), 2.0)
assert passed, err
passed, err = gradcheck(lambda t: t.gelu(), -0.5)
assert passed, err

print("all gelu tests passed")
