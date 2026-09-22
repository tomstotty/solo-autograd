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


def group_ref(data, groups, eps=1e-5):
    """Closed-form per-group normalization used as the forward reference."""
    m = len(data) // groups
    out = []
    for g in range(groups):
        chunk = data[g * m:(g + 1) * m]
        mu = sum(chunk) / m
        var = sum((v - mu) ** 2 for v in chunk) / m
        r = 1.0 / math.sqrt(var + eps)
        out.extend((v - mu) * r for v in chunk)
    return out


def close(a, b, rel_tol=1e-12, abs_tol=1e-12):
    return all(
        math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
        for x, y in zip(a, b)
    )


# --- forward basics ---
r = Tensor([1.0, 2.0, 4.0, 8.0]).group_norm(2)
assert len(r.data) == 4
assert close(r.data, group_ref([1.0, 2.0, 4.0, 8.0], 2))
assert r.requires_grad is False and r._parents == ()
# each group is normalized independently of the other
first = group_ref([1.0, 2.0], 1)
second = group_ref([4.0, 8.0], 1)
assert close(r.data[:2], first) and close(r.data[2:], second)

# groups == 1 normalizes the whole vector as a single group
r = Tensor([1.0, 2.0, 3.0]).group_norm(1)
assert close(r.data, group_ref([1.0, 2.0, 3.0], 1))

# groups == n: every element is its own zero-variance group
r = Tensor([3.0, -1.0, 7.5]).group_norm(3)
assert r.data == [0.0, 0.0, 0.0]

# custom eps
r = Tensor([3.0, 4.0]).group_norm(1, 1.0)
inv = 1.0 / math.sqrt(0.25 + 1.0)
assert r.data == [-0.5 * inv, 0.5 * inv]

# a constant group only works thanks to eps
z = Tensor([2.0, 2.0, 2.0, 2.0]).group_norm(2)
assert z.data == [0.0, 0.0, 0.0, 0.0]
z = Tensor([2.0, 2.0]).group_norm(1, 1e-6)
assert z.data == [0.0, 0.0]

# --- data validation (mutated after construction where needed) ---
expect(ValueError, lambda: Tensor(1.0).group_norm(1))
bad = Tensor(1.0)
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
x = Tensor([1.0, 2.0, 3.0, 4.0])
expect(TypeError, lambda: x.group_norm(True))
expect(TypeError, lambda: x.group_norm(2.0))
expect(TypeError, lambda: x.group_norm("2"))
expect(TypeError, lambda: x.group_norm(None))
expect(ValueError, lambda: x.group_norm(0))
expect(ValueError, lambda: x.group_norm(-1))
expect(ValueError, lambda: x.group_norm(3))
expect(ValueError, lambda: x.group_norm(5))

# --- eps validation ---
expect(TypeError, lambda: x.group_norm(2, 1))
expect(TypeError, lambda: x.group_norm(2, True))
expect(TypeError, lambda: x.group_norm(2, "1e-5"))
expect(TypeError, lambda: x.group_norm(2, None))
expect(ValueError, lambda: x.group_norm(2, 0.0))
expect(ValueError, lambda: x.group_norm(2, -1.0))
expect(ValueError, lambda: x.group_norm(2, float("inf")))
expect(ValueError, lambda: x.group_norm(2, float("nan")))

# --- non-finite forward intermediate -> V, input untouched ---
huge = Tensor([1e308, 1e308], True)
expect(ValueError, lambda: huge.group_norm(1))
assert huge.data == [1e308, 1e308] and huge.grad is None

# --- graph construction ---
assert Tensor([1.0, 2.0], False).group_norm(1).requires_grad is False
rg = Tensor([1.0, 2.0], True).group_norm(1)
assert rg.requires_grad is True
assert len(rg._parents) == 1

# --- backward: grad validation ---
out = Tensor([1.0, 2.0, 4.0, 8.0], True).group_norm(2)
expect(ValueError, lambda: out.backward())
expect(ValueError, lambda: out.backward(1.0))
expect(ValueError, lambda: out.backward([1.0]))
expect(ValueError, lambda: out.backward([1.0, 2.0, 3.0]))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward("x"))
expect(TypeError, lambda: out.backward([1.0, 2.0, 3.0, 4]))
expect(TypeError, lambda: out.backward([1.0, None, 3.0, 4.0]))
expect(ValueError, lambda: out.backward([1.0, 2.0, 3.0, float("inf")]))
expect(ValueError, lambda: out.backward([float("nan"), 2.0, 3.0, 4.0]))

# --- backward: value check against the closed form ---
x = Tensor([1.0, 2.0, 4.0, 8.0], True)
out = x.group_norm(2)
g = [0.5, -1.0, 2.0, 0.25]
out.backward(list(g))
expected = []
for base, mu, r in ((0, 1.5, 1.0 / math.sqrt(0.25 + 1e-5)),
                    (2, 6.0, 1.0 / math.sqrt(4.0 + 1e-5))):
    chunk_x = x.data[base:base + 2]
    chunk_g = g[base:base + 2]
    G = sum(chunk_g)
    H = sum(chunk_g[k] * (chunk_x[k] - mu) for k in range(2))
    for k in range(2):
        c = chunk_x[k] - mu
        expected.append((r / 2) * (2 * chunk_g[k] - G - c * r * r * H))
assert close(x.grad, expected)

# --- snapshot: mutation after forward does not affect backward ---
x = Tensor([1.0, 2.0, 4.0, 8.0], True)
out = x.group_norm(2)
x.data[0] = 100.0
x.data = [9.0, 9.0, 9.0, 9.0]
out.backward(list(g))
assert close(x.grad, expected)

# --- repeated backward accumulates ---
x = Tensor([1.0, 2.0, 4.0, 8.0], True)
out = x.group_norm(2)
out.backward(list(g))
out.backward(list(g))
assert close(x.grad, [2.0 * e for e in expected])

# --- shared path accumulates ---
x = Tensor([1.5, -2.0, 0.5, 3.0], True)
n = x.group_norm(2)
n.sum().add(n.sum()).backward()
ref = Tensor([1.5, -2.0, 0.5, 3.0], True)
ref.group_norm(2).sum().add(ref.group_norm(2).sum()).backward()
assert close(x.grad, ref.grad, rel_tol=1e-10, abs_tol=1e-10)

# --- non-finite backward intermediate -> V, grads untouched ---
# G = 1e308 + 1e308 overflows inside the first group.
x = Tensor([1.0, 2.0], True)
out = x.group_norm(1)
expect(ValueError, lambda: out.backward([1e308, 1e308]))
assert x.grad is None

# --- existing grad merge non-finite -> V, untouched ---
x = Tensor([1.0, 2.0], True)
x.grad = [float("inf"), 0.0]
expect(ValueError, lambda: x.group_norm(1).backward([1.0, 1.0]))
assert x.grad == [float("inf"), 0.0]

# --- gradcheck numeric agreement ---
passed, err = gradcheck(lambda t: t.group_norm(1).sum(), [1.5, -2.0, 0.5])
assert passed, err
passed, err = gradcheck(
    lambda t: t.group_norm(2).sum(), [1.5, -2.0, 0.5, 3.0]
)
assert passed, err
passed, err = gradcheck(lambda t: t.group_norm(2, 0.5).sum(), [0.3, -1.2])
assert passed, err
# a non-sum scalar reduction of the normalized output
passed, err = gradcheck(
    lambda t: t.group_norm(2).mul(Tensor([2.0, -1.0, 0.5, 1.5])).sum(),
    [1.5, -2.0, 0.5, 3.0],
)
assert passed, err
# near-zero input
passed, err = gradcheck(
    lambda t: t.group_norm(1, 1e-4).sum(), [1e-6, -2e-6]
)
assert passed, err

print("all group_norm tests passed")
