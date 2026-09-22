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


def reference_group_norm(values, num_groups, eps):
    n = len(values)
    m = n // num_groups
    out = []
    stats = []
    for g in range(num_groups):
        xs = values[g * m:(g + 1) * m]
        mu = sum(xs, 0.0) / m
        c = [v - mu for v in xs]
        v = sum(z * z for z in c) / m
        r = 1.0 / math.sqrt(v + eps)
        out.extend(z * r for z in c)
        stats.append((mu, c, r))
    return out, stats


# --- forward basics: independent equal-length groups ---
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
expected, _ = reference_group_norm(x, 2, 1e-5)
got = Tensor(x).group_norm(2)
assert len(got.data) == 6
assert all(
    math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    for a, b in zip(got.data, expected)
)

# one group normalizes the whole vector
expected, _ = reference_group_norm(x, 1, 1e-5)
assert all(
    math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    for a, b in zip(Tensor(x).group_norm(1).data, expected)
)

# groups == length: each singleton group is centered to exactly zero
assert Tensor(x).group_norm(6).data == [0.0] * 6

# custom eps flows through
expected, _ = reference_group_norm([3.0, 4.0], 1, 1.0)
assert all(
    math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    for a, b in zip(Tensor([3.0, 4.0]).group_norm(1, 1.0).data, expected)
)

# groups really are independent: scaling one group must not change another
a = Tensor([1.0, 2.0]).group_norm(1).data
b = Tensor([1.0, 2.0, 100.0, -100.0]).group_norm(2).data[:2]
assert all(
    math.isclose(p, q, rel_tol=1e-10, abs_tol=1e-10) for p, q in zip(a, b)
)

# --- graph construction ---
assert Tensor(x, False).group_norm(2).requires_grad is False
assert Tensor(x, False).group_norm(2)._parents == ()
rg = Tensor(x, True).group_norm(2)
assert rg.requires_grad is True
assert len(rg._parents) == 1 and rg._parents[0] is not None

# --- data validation (mutated after construction where needed) ---
expect(ValueError, lambda: Tensor(1.0).group_norm(1))
bad = Tensor([1.0, 2.0])
bad.data = 1
expect(ValueError, lambda: bad.group_norm(1))
bad.data = None
expect(ValueError, lambda: bad.group_norm(1))
bad.data = "x"
expect(ValueError, lambda: bad.group_norm(1))
bad.data = []
expect(ValueError, lambda: bad.group_norm(1))
bad.data = [1.0, 2]
expect(TypeError, lambda: bad.group_norm(1))
bad.data = [1.0, True]
expect(TypeError, lambda: bad.group_norm(1))
bad.data = [1.0, "x"]
expect(TypeError, lambda: bad.group_norm(1))
bad.data = [1.0, float("inf")]
expect(ValueError, lambda: bad.group_norm(1))
bad.data = [float("nan"), 1.0]
expect(ValueError, lambda: bad.group_norm(1))
bad.data = [1.0, 2.0]
bad.requires_grad = 1
expect(TypeError, lambda: bad.group_norm(1))
bad.requires_grad = "x"
expect(TypeError, lambda: bad.group_norm(1))

# --- groups validation ---
t = Tensor([1.0, 2.0, 3.0, 4.0])
expect(TypeError, lambda: t.group_norm(True))
expect(TypeError, lambda: t.group_norm(False))
expect(TypeError, lambda: t.group_norm(2.0))
expect(TypeError, lambda: t.group_norm("2"))
expect(TypeError, lambda: t.group_norm(None))
expect(ValueError, lambda: t.group_norm(0))
expect(ValueError, lambda: t.group_norm(-1))
expect(ValueError, lambda: t.group_norm(3))

# --- eps validation ---
expect(TypeError, lambda: t.group_norm(2, True))
expect(TypeError, lambda: t.group_norm(2, 1))
expect(TypeError, lambda: t.group_norm(2, "1e-5"))
expect(TypeError, lambda: t.group_norm(2, None))
expect(ValueError, lambda: t.group_norm(2, 0.0))
expect(ValueError, lambda: t.group_norm(2, -0.0))
expect(ValueError, lambda: t.group_norm(2, -1e-5))
expect(ValueError, lambda: t.group_norm(2, float("inf")))
expect(ValueError, lambda: t.group_norm(2, float("-inf")))
expect(ValueError, lambda: t.group_norm(2, float("nan")))

# --- non-finite forward intermediate -> V, input untouched ---
huge = Tensor([1e308, -1e308], True)
expect(ValueError, lambda: huge.group_norm(1))
assert huge.data == [1e308, -1e308] and huge.grad is None

# --- backward: grad validation ---
out = Tensor(x, True).group_norm(2)
expect(ValueError, lambda: out.backward())
expect(ValueError, lambda: out.backward(1.0))
expect(ValueError, lambda: out.backward([1.0] * 5))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward("x"))
expect(TypeError, lambda: out.backward([1.0] * 5 + [2]))
expect(TypeError, lambda: out.backward([1.0] * 5 + [None]))
expect(ValueError, lambda: out.backward([1.0] * 5 + [float("inf")]))
expect(ValueError, lambda: out.backward([float("nan")] + [1.0] * 5))

# --- backward: closed form per group ---
x = [1.5, -2.0, 0.5, 3.0, 0.25, -1.0]
g = [0.7, -1.3, 0.4, 0.1, -0.6, 0.9]
_, stats = reference_group_norm(x, 2, 1e-5)
expected_dx = []
m = 3
for group_index in range(2):
    mu, c, r = stats[group_index]
    gs = g[group_index * m:(group_index + 1) * m]
    G = sum(gs, 0.0)
    H = sum(a * b for a, b in zip(gs, c))
    expected_dx.extend(
        (r / m) * (m * a - G - b * r * r * H)
        for a, b in zip(gs, c)
    )
xt = Tensor(x, True)
xt.group_norm(2).backward(list(g))
assert all(
    math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11)
    for a, b in zip(xt.grad, expected_dx)
)

# singleton groups pass a zero gradient through
zs = Tensor([1.0, 2.0, 3.0], True)
zs.group_norm(3).backward([2.0, -3.0, 5.0])
assert zs.grad == [0.0, 0.0, 0.0]

# --- snapshot: mutation after forward does not affect backward ---
xt = Tensor(x, True)
out = xt.group_norm(2)
xt.data[0] = 100.0
xt.data = [9.0] * 6
out.backward(list(g))
assert all(
    math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11)
    for a, b in zip(xt.grad, expected_dx)
)

# --- repeated backward accumulates ---
xt = Tensor(x, True)
out = xt.group_norm(2)
out.backward(list(g))
out.backward(list(g))
assert all(
    math.isclose(a, 2.0 * b, rel_tol=1e-11, abs_tol=1e-11)
    for a, b in zip(xt.grad, expected_dx)
)

# --- shared path accumulates ---
xs = Tensor([1.5, -2.0, 0.5, 0.25], True)
n = xs.group_norm(2)
n.sum().add(n.sum()).backward()
ref = Tensor([1.5, -2.0, 0.5, 0.25], True)
ref.group_norm(2).sum().add(ref.group_norm(2).sum()).backward()
assert all(
    math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    for a, b in zip(xs.grad, ref.grad)
)

# --- non-finite backward intermediate -> V, grads untouched ---
# Forward stays finite (c ~ 1e150, r ~ 1e-150), but H = sum(g*c) overflows
# while G cancels to zero, so the failure happens inside backward.
xb = Tensor([1e150, -1e150, 0.0, 0.0], True)
out = xb.group_norm(2)
assert all(math.isfinite(v) for v in out.data)
expect(ValueError, lambda: out.backward([1e308, -1e308, 1.0, 1.0]))
assert xb.grad is None

# --- existing grad merge non-finite -> V, untouched ---
xm = Tensor([1.0, 2.0, 3.0, 4.0], True)
xm.grad = [float("inf"), 0.0, 0.0, 0.0]
expect(ValueError, lambda: xm.group_norm(2).backward([1.0] * 4))
assert xm.grad == [float("inf"), 0.0, 0.0, 0.0]

# --- gradcheck numeric agreement ---
passed, err = gradcheck(lambda t: t.group_norm(2).sum(), [1.5, -2.0, 0.5, 0.25])
assert passed, err
passed, err = gradcheck(
    lambda t: t.group_norm(2, 0.5).sum(), [0.3, -1.2, 2.5, -0.7]
)
assert passed, err
passed, err = gradcheck(
    lambda t: t.group_norm(3)
    .mul(Tensor([2.0, -1.0, 0.5, 1.5, -2.5, 0.25]))
    .sum(),
    [1.5, -2.0, 0.5, 0.25, 1.0, -0.5],
)
assert passed, err
# near-constant input keeps eps meaningful
passed, err = gradcheck(lambda t: t.group_norm(2).sum(), [1e-6, -2e-6, 3e-6, 1e-6])
assert passed, err

print("all group_norm tests passed")
