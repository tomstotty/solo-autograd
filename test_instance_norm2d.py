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


def ref(data, B, C, H, W, eps=1e-5):
    """Closed-form per-(n, c) normalization used as the forward reference."""
    M = H * W
    out = [0.0] * len(data)
    for n in range(B):
        for c in range(C):
            base = (n * C + c) * M
            chunk = data[base:base + M]
            mu = sum(chunk) / M
            var = sum((v - mu) ** 2 for v in chunk) / M
            r = 1.0 / math.sqrt(var + eps)
            for i in range(M):
                out[base + i] = (chunk[i] - mu) * r
    return out


def close(a, b, rel_tol=1e-12, abs_tol=1e-12):
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
        for x, y in zip(a, b)
    )


DATA = [
    1.0, 2.0, 4.0, 8.0,    # n0 c0
    3.0, -1.0, 0.5, 2.0,   # n0 c1
    0.0, 1.5, -3.0, 2.5,   # n1 c0
    7.0, 7.0, 7.0, 7.0,    # n1 c1 (constant instance)
]
B = C = 2
H = W = 2
EPS = 1e-5

# --- forward basics ---
r = Tensor(DATA).instance_norm2d(B, C, H, W)
assert len(r.data) == B * C * H * W
assert close(r.data, ref(DATA, B, C, H, W))
assert r.requires_grad is False and r._parents == ()
# every (n, c) instance is normalized independently
for n in range(B):
    for c in range(C):
        base = (n * C + c) * 4
        chunk = r.data[base:base + 4]
        assert math.isclose(sum(chunk), 0.0, abs_tol=1e-12)
# a constant instance normalizes to zeros thanks to eps
assert r.data[12:16] == [0.0, 0.0, 0.0, 0.0]

# H = W = 1: every spatial plane is a single zero-variance element
ones = Tensor([3.0, -2.0]).instance_norm2d(1, 2, 1, 1)
assert ones.data == [0.0, 0.0]

# custom eps
z = Tensor([3.0, 4.0]).instance_norm2d(1, 1, 1, 2, 1.0)
inv = 1.0 / math.sqrt(0.25 + 1.0)
assert z.data == [-0.5 * inv, 0.5 * inv]

# different B/C/H/W mix; flat layout really is n, c, r, s row-major
data = [float(i) for i in range(24)]
out = Tensor(data).instance_norm2d(2, 3, 2, 2)
assert close(out.data, ref(data, 2, 3, 2, 2))

# --- dim validation: type errors ---
x = Tensor(DATA)
for kwargs in (
    dict(batch=True), dict(channels=True), dict(height=True), dict(width=True),
    dict(batch=2.0), dict(channels=2.0), dict(height=2.0), dict(width=2.0),
    dict(batch="2"), dict(channels=None), dict(height=[2]), dict(width=2.0),
):
    kw = dict(batch=B, channels=C, height=H, width=W)
    kw.update(kwargs)
    expect(TypeError, lambda kw=kw: x.instance_norm2d(**kw))
# --- dim validation: value errors ---
for kwargs in (
    dict(batch=0), dict(channels=-1), dict(height=0), dict(width=-3),
):
    kw = dict(batch=B, channels=C, height=H, width=W)
    kw.update(kwargs)
    expect(ValueError, lambda kw=kw: x.instance_norm2d(**kw))

# --- data validation ---
expect(ValueError, lambda: Tensor(1.0).instance_norm2d(1, 1, 1, 1))
bad = Tensor(DATA)
bad.data = 1
expect(ValueError, lambda: bad.instance_norm2d(B, C, H, W))
bad.data = None
expect(ValueError, lambda: bad.instance_norm2d(B, C, H, W))
bad.data = "x"
expect(ValueError, lambda: bad.instance_norm2d(B, C, H, W))
bad.data = []
expect(ValueError, lambda: bad.instance_norm2d(B, C, H, W))
# wrong length
expect(ValueError, lambda: Tensor(DATA[:-1]).instance_norm2d(B, C, H, W))
expect(ValueError, lambda: Tensor(DATA + [1.0]).instance_norm2d(B, C, H, W))
# element type errors
bad.data = list(DATA)
bad.data[0] = 1
expect(TypeError, lambda: bad.instance_norm2d(B, C, H, W))
bad.data[0] = True
expect(TypeError, lambda: bad.instance_norm2d(B, C, H, W))
bad.data[0] = "x"
expect(TypeError, lambda: bad.instance_norm2d(B, C, H, W))
# non-finite elements
bad.data[0] = float("inf")
expect(ValueError, lambda: bad.instance_norm2d(B, C, H, W))
bad.data[0] = float("nan")
expect(ValueError, lambda: bad.instance_norm2d(B, C, H, W))
# requires_grad non-bool
bad.data = list(DATA)
bad.requires_grad = 1
expect(TypeError, lambda: bad.instance_norm2d(B, C, H, W))
bad.requires_grad = "x"
expect(TypeError, lambda: bad.instance_norm2d(B, C, H, W))

# --- eps validation ---
t = Tensor(DATA, True)
expect(TypeError, lambda: t.instance_norm2d(B, C, H, W, 1))
expect(TypeError, lambda: t.instance_norm2d(B, C, H, W, True))
expect(TypeError, lambda: t.instance_norm2d(B, C, H, W, "1e-5"))
expect(TypeError, lambda: t.instance_norm2d(B, C, H, W, None))
expect(ValueError, lambda: t.instance_norm2d(B, C, H, W, 0.0))
expect(ValueError, lambda: t.instance_norm2d(B, C, H, W, -1.0))
expect(ValueError, lambda: t.instance_norm2d(B, C, H, W, float("inf")))
expect(ValueError, lambda: t.instance_norm2d(B, C, H, W, float("nan")))

# --- non-finite forward intermediate -> V, input untouched ---
huge = Tensor([1e308] * 4, True)
expect(ValueError, lambda: huge.instance_norm2d(1, 1, 2, 2))
assert huge.data == [1e308] * 4 and huge.grad is None

# --- graph construction ---
assert Tensor(DATA, False).instance_norm2d(B, C, H, W).requires_grad is False
parent_src = Tensor(DATA, True)
rg = parent_src.instance_norm2d(B, C, H, W)
assert rg.requires_grad is True
assert rg._parents == (parent_src,)

# --- no graph: any grad, valid or not, -> ValueError ---
out0 = Tensor(DATA, False).instance_norm2d(B, C, H, W)
expect(ValueError, lambda: out0.backward([1.0] * 16))
expect(ValueError, lambda: out0.backward())
expect(ValueError, lambda: out0.backward(True))
expect(ValueError, lambda: out0.backward(1.0))
expect(ValueError, lambda: out0.backward(None))
expect(ValueError, lambda: out0.backward("x"))

# --- backward: grad validation (bool/int/float/scalars -> V) ---
out = Tensor(DATA, True).instance_norm2d(B, C, H, W)
expect(ValueError, lambda: out.backward())
expect(ValueError, lambda: out.backward(1.0))
expect(ValueError, lambda: out.backward(1))
expect(ValueError, lambda: out.backward(True))
expect(ValueError, lambda: out.backward(False))
expect(ValueError, lambda: out.backward([1.0]))
expect(ValueError, lambda: out.backward([1.0] * 15))
expect(ValueError, lambda: out.backward([1.0] * 17))
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward("x"))
expect(TypeError, lambda: out.backward([1] * 16))
expect(TypeError, lambda: out.backward([True] * 16))
expect(TypeError, lambda: out.backward([None] * 16))
expect(ValueError, lambda: out.backward([float("inf")] * 16))
expect(ValueError, lambda: out.backward([float("nan")] * 16))

# --- backward: value check against the closed form ---
x = Tensor(DATA, True)
out = x.instance_norm2d(B, C, H, W)
g = [
    0.5, -1.0, 2.0, 0.25,
    1.0, -0.5, 0.25, 3.0,
    -2.0, 1.5, 0.75, -0.25,
    1.0, 1.0, -1.0, -1.0,
]
out.backward(list(g))
expected = [0.0] * 16
for n in range(B):
    for c in range(C):
        base = (n * C + c) * 4
        chunk_x = DATA[base:base + 4]
        chunk_g = g[base:base + 4]
        mu = sum(chunk_x) / 4
        var = sum((v - mu) ** 2 for v in chunk_x) / 4
        R = 1.0 / math.sqrt(var + EPS)
        G = sum(chunk_g)
        Q = sum(chunk_g[k] * (chunk_x[k] - mu) for k in range(4))
        for k in range(4):
            z = chunk_x[k] - mu
            expected[base + k] = (R / 4) * (
                4 * chunk_g[k] - G - z * R * R * Q
            )
assert close(x.grad, expected)

# constant instance: every upstream contributes -G/M scaled by 1/sqrt(eps)
xc = Tensor([7.0, 7.0, 7.0, 7.0], True)
oc = xc.instance_norm2d(1, 1, 2, 2)
oc.backward([1.0, 2.0, 3.0, 4.0])
R = 1.0 / math.sqrt(EPS)
Gc = 10.0
assert close(
    xc.grad,
    [(R / 4) * (4 * gv - Gc) for gv in (1.0, 2.0, 3.0, 4.0)],
)

# --- snapshot: mutation after forward does not affect backward ---
x = Tensor(DATA, True)
out = x.instance_norm2d(B, C, H, W)
x.data[0] = 100.0
x.data = [9.0] * 16
out.backward(list(g))
assert close(x.grad, expected)

# --- repeated backward accumulates ---
x = Tensor(DATA, True)
out = x.instance_norm2d(B, C, H, W)
out.backward(list(g))
out.backward(list(g))
assert close(x.grad, [2.0 * e for e in expected])

# --- shared path accumulates ---
x = Tensor(DATA, True)
n = x.instance_norm2d(B, C, H, W)
n.sum().add(n.sum()).backward()
ref_t = Tensor(DATA, True)
a = ref_t.instance_norm2d(B, C, H, W).sum()
b = ref_t.instance_norm2d(B, C, H, W).sum()
a.add(b).backward()
assert close(x.grad, ref_t.grad, rel_tol=1e-10, abs_tol=1e-10)

# --- non-finite backward intermediate -> V, grads untouched ---
xh = Tensor([1.0, 2.0, 3.0, 4.0], True)
oh = xh.instance_norm2d(1, 1, 2, 2)
expect(ValueError, lambda: oh.backward([1e308, 1e308, 1e308, 1e308]))
assert xh.grad is None

# --- existing grad merge non-finite -> V, untouched ---
x = Tensor([1.0, 2.0, 3.0, 4.0], True)
x.grad = [float("inf"), 0.0, 0.0, 0.0]
expect(
    ValueError,
    lambda: x.instance_norm2d(1, 1, 2, 2)
    .backward([1.0, 1.0, 1.0, 1.0]),
)
assert x.grad == [float("inf"), 0.0, 0.0, 0.0]

# --- failed backward changes neither input data nor any graph grad ---
x = Tensor([1.0, 2.0, 3.0, 4.0], True)
out = x.instance_norm2d(1, 1, 2, 2)
snapshot_data = list(x.data)
expect(ValueError, lambda: out.backward([1e308, -1e308, 1.0, 1.0]))
assert x.data == snapshot_data
assert x.grad is None
# the graph still works after the failed attempt
out.backward([1.0, 1.0, 1.0, 1.0])
assert x.grad is not None and all(math.isfinite(v) for v in x.grad)

# --- gradcheck numeric agreement ---
passed, err = gradcheck(
    lambda t: t.instance_norm2d(2, 2, 2, 2).sum(),
    [1.5, -2.0, 0.5, 3.0, 0.0, 1.0, -1.0, 2.0,
     2.5, -0.5, 0.25, 1.25, -3.0, 3.0, -2.0, 0.1],
)
assert passed, err
passed, err = gradcheck(
    lambda t: t.instance_norm2d(1, 2, 2, 2, 0.5).sum(),
    [0.3, -1.2, 2.1, 0.4, -0.7, 1.8, 0.0, 0.9],
)
assert passed, err
# non-sum scalar reduction of the normalized output
weights = Tensor(
    [2.0, -1.0, 0.5, 1.5, 1.0, 1.0, -2.0, 0.25,
     0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 0.2, -0.2]
)
passed, err = gradcheck(
    lambda t: t.instance_norm2d(2, 2, 2, 2).mul(weights).sum(),
    [1.5, -2.0, 0.5, 3.0, 0.0, 1.0, -1.0, 2.0,
     2.5, -0.5, 0.25, 1.25, -3.0, 3.0, -2.0, 0.1],
)
assert passed, err

print("all instance_norm2d tests passed")
