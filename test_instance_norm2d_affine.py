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


def ref(data, w, b, B, C, H, W, eps=1e-5):
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
                out[base + i] = (chunk[i] - mu) * r * w[c] + b[c]
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
    7.0, 7.0, 7.0, 7.0,    # n1 c1
]
WV = [2.0, -1.5]
BV = [0.25, 1.0]
B = C = 2
H = W = 2
EPS = 1e-5

# --- forward basics ---
r = Tensor(DATA).instance_norm2d_affine(Tensor(WV), Tensor(BV), B, C, H, W)
assert len(r.data) == B * C * H * W
assert close(r.data, ref(DATA, WV, BV, B, C, H, W))
assert r.requires_grad is False and r._parents == ()

# affine result equals plain instance norm * w + b
plain = Tensor(DATA).instance_norm2d(B, C, H, W).data
assert close(r.data, [plain[i] * WV[(i // 4) % 2] + BV[(i // 4) % 2]
                      for i in range(16)])

# H = W = 1: output is just bias
ones = Tensor([3.0, -2.0]).instance_norm2d_affine(
    Tensor([5.0, 7.0]), Tensor([-1.0, 2.0]), 1, 2, 1, 1)
assert ones.data == [-1.0, 2.0]

# custom eps
z = Tensor([3.0, 4.0]).instance_norm2d_affine(
    Tensor([2.0]), Tensor([1.0]), 1, 1, 1, 2, 1.0)
inv = 1.0 / math.sqrt(0.25 + 1.0)
assert close(z.data, [-1.0 * inv + 1.0, 1.0 * inv + 1.0])

# different B/C/H/W mix, flat row-major
data = [float(i) for i in range(24)]
w3 = [1.5, -2.0, 0.5]
b3 = [0.1, 0.2, 0.3]
out = Tensor(data).instance_norm2d_affine(Tensor(w3), Tensor(b3), 2, 3, 2, 2)
assert close(out.data, ref(data, w3, b3, 2, 3, 2, 2))

# --- weight/bias type ---
x = Tensor(DATA)
expect(TypeError, lambda: x.instance_norm2d_affine(WV, Tensor(BV), B, C, H, W))
expect(TypeError, lambda: x.instance_norm2d_affine(Tensor(WV), BV, B, C, H, W))
expect(TypeError, lambda: x.instance_norm2d_affine(None, Tensor(BV), B, C, H, W))

# --- dim validation: type errors ---
for kwargs in (
    dict(batch=True), dict(channels=True), dict(height=True), dict(width=True),
    dict(batch=2.0), dict(channels=2.0), dict(height=2.0), dict(width=2.0),
    dict(batch="2"), dict(channels=None), dict(height=[2]), dict(width=2.0),
):
    kw = dict(batch=B, channels=C, height=H, width=W)
    kw.update(kwargs)
    expect(TypeError,
           lambda kw=kw: x.instance_norm2d_affine(Tensor(WV), Tensor(BV),
                                                  **kw))
# --- dim validation: value errors ---
for kwargs in (
    dict(batch=0), dict(channels=-1), dict(height=0), dict(width=-3),
):
    kw = dict(batch=B, channels=C, height=H, width=W)
    kw.update(kwargs)
    expect(ValueError,
           lambda kw=kw: x.instance_norm2d_affine(Tensor(WV), Tensor(BV),
                                                  **kw))

# --- data validation ---
expect(ValueError, lambda: Tensor(1.0).instance_norm2d_affine(
    Tensor([1.0]), Tensor([1.0]), 1, 1, 1, 1))
# wrong lengths
expect(ValueError, lambda: Tensor(DATA[:-1]).instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W))
expect(ValueError, lambda: Tensor(DATA).instance_norm2d_affine(
    Tensor(WV + [1.0]), Tensor(BV), B, C, H, W))
expect(ValueError, lambda: Tensor(DATA).instance_norm2d_affine(
    Tensor(WV), Tensor(BV[:-1]), B, C, H, W))
# element type errors on each operand
bad = Tensor(DATA)
bad.data[0] = 1
expect(TypeError, lambda: bad.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W))
badw = Tensor(WV)
badw.data[0] = True
expect(TypeError, lambda: Tensor(DATA).instance_norm2d_affine(
    badw, Tensor(BV), B, C, H, W))
badb = Tensor(BV)
badb.data[0] = "x"
expect(TypeError, lambda: Tensor(DATA).instance_norm2d_affine(
    Tensor(WV), badb, B, C, H, W))
# empty / non-list / non-finite
badw.data = []
expect(ValueError, lambda: Tensor(DATA).instance_norm2d_affine(
    badw, Tensor(BV), B, C, H, W))
badw.data = None
expect(ValueError, lambda: Tensor(DATA).instance_norm2d_affine(
    badw, Tensor(BV), B, C, H, W))
badw.data = list(WV)
badw.data[0] = float("nan")
expect(ValueError, lambda: Tensor(DATA).instance_norm2d_affine(
    badw, Tensor(BV), B, C, H, W))
# requires_grad non-bool on any of the three
bad.data = list(DATA)
bad.requires_grad = 1
expect(TypeError, lambda: bad.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W))
bad.requires_grad = False
badw.data = list(WV)
badw.requires_grad = "x"
expect(TypeError, lambda: Tensor(DATA).instance_norm2d_affine(
    badw, Tensor(BV), B, C, H, W))

# --- eps validation ---
t = Tensor(DATA, True)
expect(TypeError, lambda: t.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W, 1))
expect(TypeError, lambda: t.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W, True))
expect(TypeError, lambda: t.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W, "1e-5"))
expect(ValueError, lambda: t.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W, 0.0))
expect(ValueError, lambda: t.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W, -1.0))
expect(ValueError, lambda: t.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W, float("inf")))
expect(ValueError, lambda: t.instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W, float("nan")))

# --- non-finite forward intermediate -> V, inputs untouched ---
huge = Tensor([1e308] * 4, True)
expect(ValueError, lambda: huge.instance_norm2d_affine(
    Tensor([1.0], True), Tensor([0.0], True), 1, 1, 2, 2))
assert huge.data == [1e308] * 4 and huge.grad is None
# non-finite from the affine scale itself
# non-finite from the affine scale+bias itself (h = +/-1, sum overflows)
bigw = Tensor([1e308], True)
expect(ValueError, lambda: Tensor([1.0, -1.0, 1.0, -1.0], True)
       .instance_norm2d_affine(bigw, Tensor([1e308]), 1, 1, 2, 2))

# --- graph construction ---
assert Tensor(DATA, False).instance_norm2d_affine(
    Tensor(WV, False), Tensor(BV, False), B, C, H, W).requires_grad is False
xs = Tensor(DATA, True)
ws = Tensor(WV, True)
bs = Tensor(BV, True)
rg = xs.instance_norm2d_affine(ws, bs, B, C, H, W)
assert rg.requires_grad is True
assert rg._parents == (xs, ws, bs)
# only one parent requiring grad
xo = Tensor(DATA, False)
rg2 = xo.instance_norm2d_affine(Tensor(WV, True), Tensor(BV, False),
                               B, C, H, W)
assert rg2._parents == (xo, rg2._parents[1], rg2._parents[2])
assert rg2.requires_grad is True

# --- no graph: any grad, valid or not, -> ValueError ---
out0 = Tensor(DATA, False).instance_norm2d_affine(
    Tensor(WV), Tensor(BV), B, C, H, W)
expect(ValueError, lambda: out0.backward([1.0] * 16))
expect(ValueError, lambda: out0.backward())
expect(ValueError, lambda: out0.backward(True))
expect(ValueError, lambda: out0.backward(1.0))
expect(ValueError, lambda: out0.backward(None))
expect(ValueError, lambda: out0.backward("x"))

# --- backward: grad validation ---
out = Tensor(DATA, True).instance_norm2d_affine(
    Tensor(WV, True), Tensor(BV, True), B, C, H, W)
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
expect(ValueError, lambda: out.backward([float("inf")] * 16))
expect(ValueError, lambda: out.backward([float("nan")] * 16))

# --- backward: closed-form value check ---
x = Tensor(DATA, True)
w = Tensor(WV, True)
b = Tensor(BV, True)
out = x.instance_norm2d_affine(w, b, B, C, H, W)
g = [
    0.5, -1.0, 2.0, 0.25,
    1.0, -0.5, 0.25, 3.0,
    -2.0, 1.5, 0.75, -0.25,
    1.0, 1.0, -1.0, -1.0,
]
out.backward(list(g))
exp_dx = [0.0] * 16
exp_dw = [0.0, 0.0]
exp_db = [0.0, 0.0]
for n in range(B):
    for c in range(C):
        base = (n * C + c) * 4
        chunk_x = DATA[base:base + 4]
        chunk_g = g[base:base + 4]
        mu = sum(chunk_x) / 4
        var = sum((v - mu) ** 2 for v in chunk_x) / 4
        R = 1.0 / math.sqrt(var + EPS)
        u = [gv * WV[c] for gv in chunk_g]
        U = sum(u)
        Q = sum(u[k] * (chunk_x[k] - mu) for k in range(4))
        for k in range(4):
            z = chunk_x[k] - mu
            exp_dx[base + k] = (R / 4) * (4 * u[k] - U - z * R * R * Q)
            h = z * R
            exp_dw[c] += chunk_g[k] * h
            exp_db[c] += chunk_g[k]
assert close(x.grad, exp_dx)
assert close(w.grad, exp_dw)
assert close(b.grad, exp_db)

# parents not requiring grad receive nothing
x = Tensor(DATA, False)
w = Tensor(WV, True)
b = Tensor(BV, False)
out = x.instance_norm2d_affine(w, b, B, C, H, W)
out.backward(list(g))
assert x.grad is None and b.grad is None
assert w.grad is not None
assert close(w.grad, exp_dw)

# constant instance
xc = Tensor([7.0, 7.0, 7.0, 7.0], True)
wc = Tensor([3.0], True)
bc = Tensor([9.0], True)
oc = xc.instance_norm2d_affine(wc, bc, 1, 1, 2, 2)
oc.backward([1.0, 2.0, 3.0, 4.0])
R = 1.0 / math.sqrt(EPS)
u = [3.0, 6.0, 9.0, 12.0]
U = 30.0
assert close(xc.grad, [(R / 4) * (4 * uv - U) for uv in u])
assert close(wc.grad, [0.0])
assert close(bc.grad, [10.0])

# --- snapshot: mutation after forward does not affect backward ---
x = Tensor(DATA, True)
w = Tensor(WV, True)
b = Tensor(BV, True)
out = x.instance_norm2d_affine(w, b, B, C, H, W)
x.data[0] = 100.0
x.data = [9.0] * 16
w.data[0] = 500.0
b.data = [8.0, 8.0]
out.backward(list(g))
assert close(x.grad, exp_dx)
assert close(w.grad, exp_dw)
assert close(b.grad, exp_db)

# --- repeated backward accumulates ---
x = Tensor(DATA, True)
w = Tensor(WV, True)
b = Tensor(BV, True)
out = x.instance_norm2d_affine(w, b, B, C, H, W)
out.backward(list(g))
out.backward(list(g))
assert close(x.grad, [2.0 * e for e in exp_dx])
assert close(w.grad, [2.0 * e for e in exp_dw])
assert close(b.grad, [2.0 * e for e in exp_db])

# --- shared path accumulates ---
x = Tensor(DATA, True)
w = Tensor(WV, True)
b = Tensor(BV, True)
n = x.instance_norm2d_affine(w, b, B, C, H, W)
n.sum().add(n.sum()).backward()
xr = Tensor(DATA, True)
wr = Tensor(WV, True)
br = Tensor(BV, True)
a = xr.instance_norm2d_affine(wr, br, B, C, H, W).sum()
bb = xr.instance_norm2d_affine(wr, br, B, C, H, W).sum()
a.add(bb).backward()
assert close(x.grad, xr.grad, rel_tol=1e-10, abs_tol=1e-10)
assert close(w.grad, wr.grad, rel_tol=1e-10, abs_tol=10)
assert close(b.grad, br.grad, rel_tol=1e-10, abs_tol=1e-10)

# --- same tensor playing multiple roles: x is also weight (lengths match C) ---
# B*C*H*W must equal C for object sharing; use 1x4x1x1 with x as weight too.
d4 = [1.0, -2.0, 3.0, -4.0]
shared = Tensor(d4, True)       # also serves as weight and bias
o = shared.instance_norm2d_affine(shared, shared, 1, 4, 1, 1)
o.backward([1.0, 1.0, 1.0, 1.0])
# numeric reference: h == 0 everywhere (M=1), so y = b; db = g, dw = 0,
# and per instance U = u = g*w, hence dx = (R)*(u - U) = 0.
expected = [1.0, 1.0, 1.0, 1.0]
assert close(shared.grad, expected, rel_tol=1e-6, abs_tol=1e-6)

# weight and bias are the same tensor; its grad merges dw and db
d8 = [1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -1.5, 2.5]
p = Tensor([1.5, -2.0], True)
x8 = Tensor(d8, True)
o = x8.instance_norm2d_affine(p, p, 1, 2, 2, 2)
g8 = [1.0, 2.0, -1.0, 0.5, -2.0, 1.0, 1.0, -0.5]
o.backward(list(g8))
exp_p = [0.0, 0.0]
for c in range(2):
    base = c * 4
    chunk = d8[base:base + 4]
    mu = sum(chunk) / 4
    var = sum((v - mu) ** 2 for v in chunk) / 4
    R = 1.0 / math.sqrt(var + EPS)
    for k in range(4):
        exp_p[c] += g8[base + k] * (chunk[k] - mu) * R
    exp_p[c] += sum(g8[base:base + 4])
assert close(p.grad, exp_p)

# --- non-finite backward intermediate -> V, grads untouched ---
xh = Tensor([1.0, 2.0, 3.0, 4.0], True)
wh = Tensor([1e308], True)
oh = xh.instance_norm2d_affine(wh, Tensor([0.0]), 1, 1, 2, 2)
expect(ValueError, lambda: oh.backward([1.0, 1.0, 1.0, 1.0]))
assert xh.grad is None and wh.grad is None

# --- existing grad merge non-finite -> V, untouched ---
x = Tensor([1.0, 2.0, 3.0, 4.0], True)
x.grad = [float("inf"), 0.0, 0.0, 0.0]
expect(ValueError, lambda: x.instance_norm2d_affine(
    Tensor([1.0]), Tensor([0.0]), 1, 1, 2, 2)
    .backward([1.0, 1.0, 1.0, 1.0]))
assert x.grad == [float("inf"), 0.0, 0.0, 0.0]

# --- failed backward changes nothing; graph still works ---
x = Tensor([1.0, 2.0, 3.0, 4.0], True)
w = Tensor([1.0], True)
out = x.instance_norm2d_affine(w, Tensor([0.0]), 1, 1, 2, 2)
snapshot_data = list(x.data)
expect(ValueError, lambda: out.backward([1e308, -1e308, 1.0, 1.0]))
assert x.data == snapshot_data
assert x.grad is None and w.grad is None
out.backward([1.0, 1.0, 1.0, 1.0])
assert x.grad is not None and all(math.isfinite(v) for v in x.grad)

# --- gradcheck numeric agreement ---
data16 = [
    1.5, -2.0, 0.5, 3.0, 0.0, 1.0, -1.0, 2.0,
    2.5, -0.5, 0.25, 1.25, -3.0, 3.0, -2.0, 0.1,
]
wv = Tensor([2.0, -1.5], True)
bv = Tensor([0.25, -0.5], True)
passed, err = gradcheck(
    lambda t: t.instance_norm2d_affine(wv, bv, 2, 2, 2, 2).sum(),
    data16,
)
assert passed, err
# grads w.r.t. weight and bias via perturbing their values (finite diff)
def fd_param(param, idx):
    eps = 1e-6

    def loss(pvals):
        wt = Tensor([pvals[0], pvals[1]])
        bt = Tensor([pvals[2], pvals[3]])
        return sum(Tensor(data16, True)
                   .instance_norm2d_affine(wt, bt, 2, 2, 2, 2).data
                   [i] for i in range(16))

    base = [WV[0], WV[1], BV[0], BV[1]]
    p1 = list(base)
    p1[idx] += eps
    p0 = list(base)
    p0[idx] -= eps
    return (loss(p1) - loss(p0)) / (2 * eps)

wt = Tensor(WV, True)
bt = Tensor(BV, True)
o = Tensor(data16, True).instance_norm2d_affine(wt, bt, 2, 2, 2, 2)
o.backward([1.0] * 16)
for c in range(2):
    assert math.isclose(wt.grad[c], fd_param(None, c),
                        rel_tol=1e-6, abs_tol=1e-6), (c, wt.grad)
    assert math.isclose(bt.grad[c], fd_param(None, c + 2),
                        rel_tol=1e-6, abs_tol=1e-6)

# non-sum weighted reduction
weights = Tensor(
    [2.0, -1.0, 0.5, 1.5, 1.0, 1.0, -2.0, 0.25,
     0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 0.2, -0.2]
)
passed, err = gradcheck(
    lambda t: t.instance_norm2d_affine(
        Tensor([0.5, 2.0]), Tensor([1.0, -1.0]), 2, 2, 2, 2)
    .mul(weights).sum(),
    data16,
)
assert passed, err

print("all instance_norm2d_affine tests passed")
