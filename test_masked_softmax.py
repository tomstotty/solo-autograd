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
base = Tensor([1.0, 2.0, 1.0, 2.0])
# non-list data
bad = Tensor(1.0)
expect(ValueError, lambda: bad.masked_softmax([True], 1, 1))
bad.data = "x"
expect(ValueError, lambda: bad.masked_softmax([True], 1, 1))
# empty
bad.data = []
expect(ValueError, lambda: bad.masked_softmax([True], 1, 1))
# wrong length
expect(ValueError, lambda: base.masked_softmax([True] * 3, 2, 2))
# non-float elements
bad.data = [1.0, True]
expect(TypeError, lambda: bad.masked_softmax([True, True], 1, 2))
bad.data = [1, 2]
expect(TypeError, lambda: bad.masked_softmax([True, True], 1, 2))
# non-finite
bad.data = [1.0, float("inf")]
expect(ValueError, lambda: bad.masked_softmax([True, True], 1, 2))
bad.data = [1.0, float("nan")]
expect(ValueError, lambda: bad.masked_softmax([True, True], 1, 2))
# requires_grad non-bool
bad.data = [1.0, 2.0]
bad.requires_grad = 1
expect(TypeError, lambda: bad.masked_softmax([True, True], 1, 2))
bad.requires_grad = False

# --- rows / cols validation ---
for name, kw in (
    ("rows bool", dict(rows=True, cols=2)),
    ("rows float", dict(rows=1.0, cols=2)),
    ("rows str", dict(rows="1", cols=2)),
    ("cols bool", dict(rows=1, cols=False)),
    ("cols float", dict(rows=1, cols=2.0)),
):
    expect(
        TypeError,
        lambda kw=kw: base.masked_softmax([True] * 4, **kw),
    )
expect(ValueError, lambda: base.masked_softmax([True] * 4, 0, 2))
expect(ValueError, lambda: base.masked_softmax([True] * 4, 2, 0))
expect(ValueError, lambda: base.masked_softmax([True] * 4, -1, 2))

# --- mask validation ---
expect(TypeError, lambda: base.masked_softmax((True,) * 4, 2, 2))
expect(TypeError, lambda: base.masked_softmax(True, 2, 2))
expect(TypeError, lambda: base.masked_softmax(None, 2, 2))
expect(TypeError, lambda: base.masked_softmax("xxxx", 2, 2))
expect(TypeError, lambda: base.masked_softmax([1, 0, 1, 0], 2, 2))
expect(TypeError, lambda: base.masked_softmax([True, 0, True, False], 2, 2))
expect(TypeError, lambda: base.masked_softmax([1.0] * 4, 2, 2))
expect(ValueError, lambda: base.masked_softmax([], 2, 2))
expect(ValueError, lambda: base.masked_softmax([True] * 3, 2, 2))
expect(ValueError, lambda: base.masked_softmax([True] * 5, 2, 2))
# each row needs a true entry
expect(ValueError, lambda: base.masked_softmax([False] * 4, 2, 2))
expect(ValueError, lambda: base.masked_softmax([True, True, False, False], 2, 2))
expect(ValueError, lambda: base.masked_softmax([False, False, True, True], 2, 2))

# --- forward values ---
# single row, full mask matches softmax
x = [1.0, 2.0, 3.0]
full = Tensor(x).masked_softmax([True, True, True], 1, 3).data
ref = Tensor(x).softmax().data
for a, b in zip(full, ref):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (full, ref)

# masking out a position: softmax over [1,3] at cols 0 and 2, col1 = 0
m = [True, False, True]
y = Tensor([1.0, 99.0, 3.0]).masked_softmax(m, 1, 3).data
assert y[1] == 0.0
e2 = math.exp(2.0 - 2.0)  # max of kept is 3.0 -> diffs 1-3=-2, 3-3=0
e0 = math.exp(1.0 - 3.0)
s = e0 + e2
assert math.isclose(y[0], e0 / s)
assert math.isclose(y[2], e2 / s)

# masked value ignored even if huge
y = Tensor([1000.0, 1.0, 1.0]).masked_softmax(
    [False, True, True], 1, 3
).data
assert y[0] == 0.0
assert math.isclose(y[1], 0.5)
assert math.isclose(y[2], 0.5)

# two rows row-major
data = [0.0, 0.0, 0.0, 0.0]
y = Tensor(data).masked_softmax(
    [True, True, False, True], 2, 2
).data
assert y == [0.5, 0.5, 0.0, 1.0], y

# stability: huge finite kept values must stay finite (shifted max)
y = Tensor([1e300, 1e300]).masked_softmax([True, True], 1, 2).data
assert all(math.isfinite(v) for v in y)

# non-finite intermediate: underflow to 0 across the row is fine-ish; check
# extreme negative diffs still give finite 0.0
y = Tensor([-1e300, 0.0]).masked_softmax([True, True], 1, 2).data
assert all(math.isfinite(v) for v in y)

# --- no graph when requires_grad False ---
r = Tensor([1.0, 2.0]).masked_softmax([True, True], 1, 2)
assert r.requires_grad is False and r._parents == ()
expect(ValueError, lambda: r.backward([1.0, 1.0]))

# --- graph when requires_grad True ---
t = Tensor([1.0, 2.0, 3.0, 4.0], True)
r = t.masked_softmax([True, True, True, False], 2, 2)
assert r.requires_grad is True and r._parents == (t,)

# --- backward grad validation ---
expect(ValueError, lambda: r.backward())
expect(ValueError, lambda: r.backward(1.0))
expect(ValueError, lambda: r.backward([1.0, 2.0, 3.0]))
expect(ValueError, lambda: r.backward([1.0, 2.0, 3.0, 4.0, 5.0]))
expect(TypeError, lambda: r.backward(None))
expect(TypeError, lambda: r.backward(True))
expect(TypeError, lambda: r.backward(1))
expect(TypeError, lambda: r.backward("x"))
expect(TypeError, lambda: r.backward([1.0, 2.0, 3.0, 4]))
expect(ValueError, lambda: r.backward([1.0, 2.0, 3.0, float("nan")]))
assert t.grad is None

# --- backward values vs finite differences (full mask) ---
passed, err = gradcheck(
    lambda t: t.masked_softmax(
        [True] * 6, 2, 3
    ).sum(),
    [0.5, -1.0, 2.0, 1.5, 0.0, -0.5],
)
assert passed, err

# partial mask: masked inputs get zero grad, output independent of them
passed, err = gradcheck(
    lambda t: t.masked_softmax(
        [True, False, True, True, False, False], 2, 3
    ).sum(),
    [0.5, 100.0, 2.0, 1.5, -50.0, 9.0],
)
assert passed, err

# explicit backward check against standard softmax Jacobian on kept cols
t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], True)
mask = [True, False, True, False, True, True]
r = t.masked_softmax(mask, 2, 3)
g = [0.3, 0.9, -0.2, 0.4, 0.1, -0.7]
r.backward(list(g))
y = r.data
dx = t.grad
for row in range(2):
    base = row * 3
    kept = [c for c in range(3) if mask[base + c]]
    d = sum(g[base + c] * y[base + c] for c in kept)
    for c in range(3):
        expected = y[base + c] * (g[base + c] - d) if c in kept else 0.0
        assert math.isclose(dx[base + c], expected, rel_tol=1e-12, abs_tol=1e-12)

# --- snapshots: mutating mask/data after forward does not affect backward ---
mask = [True, True, False]
t = Tensor([1.0, 2.0, 7.0], True)
r = t.masked_softmax(mask, 1, 3)
saved_y = list(r.data)
mask[0] = False
mask[2] = True
t.data = [100.0, 100.0, 100.0]
assert r.data == saved_y
r.backward([1.0, 2.0, 3.0])
d = 1.0 * saved_y[0] + 2.0 * saved_y[1]
assert math.isclose(t.grad[0], saved_y[0] * (1.0 - d))
assert math.isclose(t.grad[1], saved_y[1] * (2.0 - d))
assert t.grad[2] == 0.0

# --- repeated backward accumulates ---
t = Tensor([1.0, 2.0], True)
r = t.masked_softmax([True, True], 1, 2)
r.backward([1.0, 0.0])
first = list(t.grad)
r.backward([1.0, 0.0])
for a, b in zip(t.grad, [2 * v for v in first]):
    assert math.isclose(a, b, rel_tol=1e-12)

# --- failed backward changes nothing ---
t = Tensor([1.0, 2.0], True)
r = t.masked_softmax([True, True], 1, 2)
r.backward([1.0, 1.0])
before = list(t.grad)
expect(ValueError, lambda: r.backward([float("inf"), 1.0]))
assert t.grad == before

# --- shared path accumulation ---
a = Tensor([1.0, 2.0, 3.0, 4.0], True)
r1 = a.masked_softmax([True] * 4, 2, 2)
r2 = a.masked_softmax([True] * 4, 2, 2)
r1.backward([1.0, 0.0, 0.0, 0.0])
r2.backward([0.0, 0.0, 1.0, 0.0])
assert all(v != 0.0 for v in a.grad), a.grad

print("all masked_softmax tests passed")
