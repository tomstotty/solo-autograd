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


# --- forward: data validation ---
t = Tensor([1.0])
t.data = 1
expect(ValueError, t.norm)  # non-list -> V
t.data = True
expect(ValueError, t.norm)
t.data = 1.0
expect(ValueError, t.norm)  # scalar -> V (requires non-empty vector)
t.data = []
expect(ValueError, t.norm)
t.data = None
expect(ValueError, t.norm)
t.data = [[1.0]]
expect(TypeError, t.norm)  # nested -> T
t.data = [1.0, True]
expect(TypeError, t.norm)
t.data = [1.0, 2]
expect(TypeError, t.norm)
t.data = [1.0, "x"]
expect(TypeError, t.norm)
t.data = [1.0, float("nan")]
expect(ValueError, t.norm)
t.data = [float("inf")]
expect(ValueError, t.norm)
t.data = [1.0]
t.requires_grad = 1
expect(TypeError, t.norm)
t.requires_grad = "x"
expect(TypeError, t.norm)
t.requires_grad = False

# --- forward: p validation ---
t = Tensor([3.0, 4.0])
expect(TypeError, lambda: t.norm(2))
expect(TypeError, lambda: t.norm(True))
expect(TypeError, lambda: t.norm("2"))
expect(TypeError, lambda: t.norm(None))
expect(ValueError, lambda: t.norm(float("nan")))
expect(ValueError, lambda: t.norm(float("inf")))
expect(ValueError, lambda: t.norm(0.0))
expect(ValueError, lambda: t.norm(-1.0))

# --- forward: values ---
assert Tensor([3.0, 4.0]).norm().data == 5.0
assert isinstance(Tensor([3.0]).norm().data, float)
assert Tensor([-3.0, -4.0]).norm().data == 5.0
assert Tensor([0.0]).norm().data == 0.0
assert Tensor([1.0, 2.0, 2.0]).norm(3.0) != 0.0
v = Tensor([1.0, -2.0, 2.0]).norm(3.0).data
assert abs(v - (1.0 + 8.0 + 8.0) ** (1.0 / 3.0)) < 1e-12
# p=1: sum of absolutes
assert Tensor([1.0, -2.0, 3.0]).norm(1.0).data == 6.0
# p=0.5
val = Tensor([4.0, 9.0]).norm(0.5).data
assert abs(val - (2.0 + 3.0) ** 2.0) < 1e-9
# large p / overflow -> V
expect(ValueError, lambda: Tensor([1e300, 1e300]).norm(2.0))
expect(ValueError, lambda: Tensor([1e20]).norm(100.0))
# failure leaves input untouched
x = Tensor([1.0, 2.0], True)
expect(ValueError, lambda: x.norm(100000.0))
assert x.data == [1.0, 2.0] and x.grad is None

# --- no graph without requires_grad; one-parent graph otherwise ---
r = Tensor([3.0, 4.0]).norm()
assert r.data == 5.0 and r.requires_grad is False and r._parents == ()
expect(ValueError, r.backward)
r2 = Tensor([3.0, 4.0], True).norm()
assert r2.requires_grad is True and len(r2._parents) == 1

# --- backward: scalar rules ---
# d_i = sign(x_i)*abs(x_i)**(p-1)/n**(p-1); p=2 -> x_i / n
a = Tensor([3.0, 4.0], True)
out = a.norm()
assert out.data == 5.0
out.backward()
assert a.grad == [0.6, 0.8]

# p=1: sign only
a = Tensor([1.0, -2.0, 3.0], True)
a.norm(1.0).backward()
assert a.grad == [1.0, -1.0, 1.0]

# zero with p >= 1 -> derivative 0
a = Tensor([0.0, 3.0, 0.0], True)
a.norm(2.0).backward()
assert a.grad == [0.0, 1.0, 0.0]
a = Tensor([0.0, 0.0], True)
a.norm(1.0).backward()
assert a.grad == [0.0, 0.0]

# n == 0 and p < 1 -> ValueError, grads untouched
a = Tensor([0.0, 0.0], True)
expect(ValueError, lambda: a.norm(0.5).backward())
assert a.grad is None

# zero entry but n != 0 with p < 1 -> infinite derivative -> ValueError
a = Tensor([0.0, 1.0], True)
expect(ValueError, lambda: a.norm(0.5).backward())
assert a.grad is None

# --- backward: grad validation ---
out = Tensor([3.0, 4.0], True).norm()
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward("x"))
expect(ValueError, lambda: out.backward(float("inf")))
expect(ValueError, lambda: out.backward(float("nan")))
expect(ValueError, lambda: out.backward([1.0]))
expect(ValueError, lambda: out.backward([1.0, 2]))
assert out.backward() is None

# --- snapshot: mutation/replacement after forward does not affect backward ---
a = Tensor([3.0, 4.0], True)
out = a.norm()
a.data[0] = 100.0
a.data = [100.0, 100.0]
out.backward()
assert a.grad == [0.6, 0.8]
a = Tensor([0.0, 4.0], True)
out = a.norm()
a.data = [3.0, 4.0]
out.backward()
assert a.grad == [0.0, 1.0]

# --- repeated backward accumulates ---
a = Tensor([3.0, 4.0], True)
out = a.norm()
out.backward()
out.backward()
assert a.grad == [1.2, 1.6]

# --- shared path accumulates: y = norm(x)*norm(x) = sum x_i^2; dy/dx = 2x ---
x = Tensor([3.0, 4.0], True)
y = x.norm().mul(x.norm())
y.backward()
assert x.grad == [6.0, 8.0]
s = Tensor([3.0, 4.0], True)
n = s.norm()
y = n.mul(n)
y.backward()
assert s.grad == [6.0, 8.0]

# --- existing grad merge non-finite -> ValueError, untouched ---
a = Tensor([3.0, 4.0], True)
a.grad = [float("inf"), 0.0]
expect(ValueError, lambda: a.norm().backward())
assert a.grad == [float("inf"), 0.0]

# --- p=3 numeric agreement, positive/negative/zero mix ---
passed, err = gradcheck(
    lambda t: t.norm(3.0), [0.5, -1.2, 2.3, -0.7]
)
assert passed, err
passed, err = gradcheck(lambda t: t.norm(1.5), [0.3, 1.1, -2.0])
assert passed, err
# p < 1 away from zero is smooth
passed, err = gradcheck(lambda t: t.norm(0.7), [0.9, 1.4, -1.1])
assert passed, err

print("all norm tests passed")
