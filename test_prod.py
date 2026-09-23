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


# --- requires_grad must be a bool at call time ---
t = Tensor(2.0)
t.requires_grad = 1
expect(TypeError, t.prod)
t.requires_grad = "x"
expect(TypeError, t.prod)
t.requires_grad = False
assert t.prod().data == 2.0
assert Tensor(3.0, True).prod().backward() is None
assert Tensor([1.0, 2.0, 3.0]).prod().data == 6.0

# --- forward: data validation (mutated after construction) ---
bad = Tensor(1.0)
bad.data = 1
expect(TypeError, bad.prod)
bad.data = True
expect(TypeError, bad.prod)
bad.data = "1.0"
expect(TypeError, bad.prod)
bad.data = None
expect(TypeError, bad.prod)
bad.data = [[1.0]]
expect(TypeError, bad.prod)  # nested list
bad.data = (1.0, 2.0)
expect(TypeError, bad.prod)  # tuple, not list
bad.data = [1.0, True]
expect(TypeError, bad.prod)
bad.data = [1.0, 2]
expect(TypeError, bad.prod)
bad.data = []
expect(ValueError, bad.prod)
bad.data = float("nan")
expect(ValueError, bad.prod)
bad.data = float("inf")
expect(ValueError, bad.prod)
bad.data = [1.0, float("nan")]
expect(ValueError, bad.prod)
bad.data = 2.0
bad.requires_grad = 1
expect(TypeError, bad.prod)
bad.requires_grad = False

# --- forward: scalar result is x itself; vector multiplies from 1.0 ---
assert Tensor(0.0).prod().data == 0.0
assert Tensor(-3.5).prod().data == -3.5
assert Tensor([2.0, 3.0, 4.0]).prod().data == 24.0
assert Tensor([-2.0, 3.0, -4.0]).prod().data == 24.0
# single-element vector: empty-product-style forward gives x[0]
assert Tensor([7.0]).prod().data == 7.0
assert isinstance(Tensor([2.0, 3.0]).prod().data, float)
assert isinstance(Tensor(2.0).prod().data, float)
# zero element: the running product becomes 0.0 and stays there
assert Tensor([1.0, 2.0, 0.0, 9.0]).prod().data == 0.0

# --- forward overflow -> ValueError; failure leaves the input untouched ---
expect(ValueError, lambda: Tensor([1e300, 1e300]).prod())
x = Tensor([1e300, 1e300])
expect(ValueError, x.prod)
assert x.data == [1e300, 1e300] and x.grad is None
x = Tensor([1e300, 1e300], True)
expect(ValueError, x.prod)
assert x.data == [1e300, 1e300] and x.grad is None

# --- no graph without requires_grad; one-parent graph otherwise ---
r = Tensor([2.0, 3.0]).prod()
assert r.data == 6.0 and r.requires_grad is False and r._parents == ()
expect(ValueError, r.backward)
r2 = Tensor([2.0, 3.0], True).prod()
assert r2.requires_grad is True and len(r2._parents) == 1
rs = Tensor(2.0).prod()
assert rs.requires_grad is False and rs._parents == ()
expect(ValueError, rs.backward)

# --- backward: scalar contribution is g ---
a = Tensor(4.0, True)
out = a.prod()
assert out.data == 4.0
out.backward()
assert a.grad == 1.0
a = Tensor(4.0, True)
a.prod().backward(3.0)
assert a.grad == 3.0

# --- backward: vector, dx_i = g * prod of all other elements (ascending j) ---
v = Tensor([2.0, 3.0, 4.0], True)
out = v.prod()
assert out.data == 24.0
out.backward()
assert v.grad == [12.0, 8.0, 6.0]
v = Tensor([2.0, 3.0, 4.0], True)
v.prod().backward(2.0)
assert v.grad == [24.0, 16.0, 12.0]

# --- backward: single-element vector uses the empty product 1.0 ---
v = Tensor([7.0], True)
v.prod().backward()
assert v.grad == [1.0]
v = Tensor([7.0], True)
v.prod().backward(5.0)
assert v.grad == [5.0]

# --- backward: zero element still gets the natural product gradient ---
v = Tensor([0.0, 3.0, 4.0], True)
v.prod().backward()
# dx = [3*4, 0*4, 0*3] = [12, 0, 0]
assert v.grad == [12.0, 0.0, 0.0]
v = Tensor([2.0, 0.0, 4.0, 0.0], True)
v.prod().backward(3.0)
# dx_0 = 3*0*4*0 = 0; dx_1 = 3*2*4*0 = 0; dx_2 = 3*2*0*0 = 0;
# dx_3 = 3*2*0*4 = 0
assert v.grad == [0.0, 0.0, 0.0, 0.0]

# --- backward: grad validation (output is always a scalar) ---
out = Tensor([2.0, 3.0], True).prod()
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward("x"))
expect(ValueError, lambda: out.backward([1.0]))
expect(ValueError, lambda: out.backward(float("inf")))
expect(ValueError, lambda: out.backward(float("nan")))
out.backward(2.0)
assert out._parents[0].grad == [6.0, 4.0]

# --- backward: non-finite intermediate (overflow), grads untouched ---
# prod of all-but-i overflows for the smallest element's contribution.
v = Tensor([1e-200, 1e200, 1e200], True)
out = v.prod()
assert out.data == 1e200
expect(ValueError, lambda: out.backward(1.0))
assert v.grad is None
# g large enough that product * g overflows even when the product is finite
v = Tensor([2.0, 3.0, 4.0], True)
expect(ValueError, lambda: v.prod().backward(1e308))
assert v.grad is None

# --- snapshot: mutation/replacement after forward does not affect backward ---
v = Tensor([2.0, 3.0, 4.0], True)
out = v.prod()
v.data[0] = 100.0
v.data = [100.0, 100.0, 100.0]
out.backward()
assert v.grad == [12.0, 8.0, 6.0]
a = Tensor(4.0, True)
out = a.prod()
a.data = 100.0
out.backward(3.0)
assert a.grad == 3.0

# --- repeated backward accumulates ---
a = Tensor(4.0, True)
out = a.prod()
out.backward(1.0)
out.backward(1.0)
assert a.grad == 2.0
v = Tensor([2.0, 3.0, 4.0], True)
out = v.prod()
out.backward(1.0)
out.backward(1.0)
assert v.grad == [24.0, 16.0, 12.0]

# --- shared path accumulates: y = prod(x) + prod(x), dy/dx = 2 * others ---
v = Tensor([2.0, 3.0, 4.0], True)
p = v.prod()
y = p.add(p)
y.backward()
assert v.grad == [24.0, 16.0, 12.0]

# --- alias: same child used twice through a shared node ---
v = Tensor([2.0, 3.0], True)
p = v.prod()
y = p.mul(p)
# y = 36; dy/d(prod) = 2*6 = 12; dx = 12*[3, 2] = [36, 24]
y.backward()
assert v.grad == [36.0, 24.0]

# --- scalar through shared path: y = prod(x) * prod(x) = x^2, dy/dx = 2x ---
x = Tensor(4.0, True)
p = x.prod()
y = p.mul(p)
y.backward()
assert x.grad == 8.0

# --- existing grad merge non-finite -> ValueError, untouched ---
v = Tensor([2.0, 3.0], True)
v.grad = [float("inf"), 0.0]
expect(ValueError, lambda: v.prod().backward(1.0))
assert v.grad == [float("inf"), 0.0]

# --- failed backward changes no grad anywhere in the graph ---
v = Tensor([2.0, 3.0, 4.0], True)
w = Tensor([1e-200, 1e200, 1e200], True)
out = v.prod().add(w.prod())
expect(ValueError, lambda: out.backward())
assert v.grad is None and w.grad is None

# --- gradcheck numeric agreement ---
passed, err = gradcheck(lambda t: t.prod(), [0.5, 2.0, 3.0, -1.5])
assert passed, err
passed, err = gradcheck(lambda t: t.prod(), 2.0)
assert passed, err

print("all prod tests passed")
