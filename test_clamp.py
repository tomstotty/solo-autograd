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


t = Tensor(1.0)

# --- bound type validation: Tensor or finite float only ---
expect(TypeError, lambda: t.clamp(True, 2.0))
expect(TypeError, lambda: t.clamp(0, 2.0))
expect(TypeError, lambda: t.clamp("0", 2.0))
expect(TypeError, lambda: t.clamp(None, 2.0))
expect(TypeError, lambda: t.clamp(0.0, True))
expect(TypeError, lambda: t.clamp(0.0, 2))
expect(TypeError, lambda: t.clamp(0.0, "2"))
expect(TypeError, lambda: t.clamp(0.0, None))
expect(ValueError, lambda: t.clamp(float("inf"), 2.0))
expect(ValueError, lambda: t.clamp(float("nan"), 2.0))
expect(ValueError, lambda: t.clamp(0.0, float("inf")))
expect(ValueError, lambda: t.clamp(0.0, float("nan")))

# --- data validation (mutated after construction) ---
bad = Tensor(1.0)
bad.data = 1
expect(TypeError, lambda: bad.clamp(0.0, 2.0))
bad.data = [1.0, True]
expect(TypeError, lambda: bad.clamp(0.0, 2.0))
bad.data = []
expect(ValueError, lambda: bad.clamp(0.0, 2.0))
bad.data = float("inf")
expect(ValueError, lambda: bad.clamp(0.0, 2.0))
bad.data = 1.0
bad.requires_grad = 1
expect(TypeError, lambda: bad.clamp(0.0, 2.0))

lb = Tensor(0.0)
lb.data = 1
expect(TypeError, lambda: t.clamp(lb, 2.0))
lb.data = []
expect(ValueError, lambda: t.clamp(lb, 2.0))
lb.data = float("nan")
expect(ValueError, lambda: t.clamp(lb, 2.0))
lb.data = 0.0
lb.requires_grad = "x"
expect(TypeError, lambda: t.clamp(lb, 2.0))

ub = Tensor(2.0)
ub.data = 1
expect(TypeError, lambda: t.clamp(0.0, ub))
ub.data = []
expect(ValueError, lambda: t.clamp(0.0, ub))
ub.data = float("nan")
expect(ValueError, lambda: t.clamp(0.0, ub))
ub.data = 2.0
ub.requires_grad = []
expect(TypeError, lambda: t.clamp(0.0, ub))

# --- forward: scalar and vector selection with broadcasting ---
assert Tensor(1.0).clamp(0.0, 2.0).data == 1.0
assert Tensor(-1.0).clamp(0.0, 2.0).data == 0.0
assert Tensor(3.0).clamp(0.0, 2.0).data == 2.0
assert Tensor([-1.0, 1.0, 3.0]).clamp(0.0, 2.0).data == [0.0, 1.0, 2.0]
# scalar self broadcasts across vector bounds
assert Tensor(1.0).clamp(
    Tensor([0.0, 2.0, 0.0]), Tensor([2.0, 3.0, 0.5])
).data == [
    1.0,
    2.0,
    0.5,
]
# scalar bounds broadcast across vector self
assert Tensor([-5.0, 0.0, 5.0]).clamp(Tensor(0.0), Tensor(2.0)).data == [
    0.0,
    0.0,
    2.0,
]
# mixed scalar/vector
assert Tensor([1.0, 4.0]).clamp(Tensor([2.0, 0.0]), 3.0).data == [2.0, 3.0]

# raw lists are not Tensors/floats -> TypeError
expect(TypeError, lambda: Tensor([1.0, 2.0]).clamp([0.0, 0.0], 2.0))
expect(TypeError, lambda: Tensor([1.0, 2.0]).clamp(0.0, [2.0, 2.0]))

# --- vector length mismatch -> ValueError ---
expect(ValueError, lambda: Tensor([1.0, 2.0]).clamp(Tensor([0.0]), 2.0))
expect(
    ValueError,
    lambda: Tensor([1.0, 2.0]).clamp(0.0, Tensor([1.0, 2.0, 3.0])),
)
expect(
    ValueError,
    lambda: Tensor([1.0, 2.0]).clamp(Tensor([0.0, 0.0]), Tensor([1.0])),
)

# --- per-position lower < upper required ---
expect(ValueError, lambda: Tensor(1.0).clamp(2.0, 2.0))
expect(ValueError, lambda: Tensor(1.0).clamp(3.0, 2.0))
expect(
    ValueError,
    lambda: Tensor([1.0, 1.0]).clamp(
        Tensor([0.0, 5.0]), Tensor([2.0, 4.0])
    ),
)
# equality at one inverted position still rejects even if output finite
expect(
    ValueError,
    lambda: Tensor([-5.0, 1.0]).clamp(
        Tensor([0.0, 2.0]), Tensor([2.0, 2.0])
    ),
)

# --- no graph when no parent requires grad ---
r = Tensor(1.0).clamp(0.0, 2.0)
assert r.requires_grad is False and r._parents == ()
expect(ValueError, lambda: r.backward())
rv = Tensor([-1.0, 5.0]).clamp(0.0, 2.0)
assert rv.requires_grad is False and rv._parents == ()
expect(ValueError, lambda: rv.backward([1.0, 1.0]))

# --- scalar backward: interior / lower / upper ---
a = Tensor(1.0, True)
lo = Tensor(0.0, True)
hi = Tensor(2.0, True)
a.clamp(lo, hi).backward()
assert a.grad == 1.0 and lo.grad == 0.0 and hi.grad == 0.0
for p in (a, lo, hi):
    p.zero_grad()

a = Tensor(-1.0, True)
lo = Tensor(0.0, True)
hi = Tensor(2.0, True)
a.clamp(lo, hi).backward()
assert a.grad == 0.0 and lo.grad == 1.0 and hi.grad == 0.0
for p in (a, lo, hi):
    p.zero_grad()

a = Tensor(3.0, True)
lo = Tensor(0.0, True)
hi = Tensor(2.0, True)
a.clamp(lo, hi).backward(2.0)
assert a.grad == 0.0 and lo.grad == 0.0 and hi.grad == 2.0

# --- exact ties split 0.5 / 0.5 ---
a = Tensor(0.0, True)
lo = Tensor(0.0, True)
hi = Tensor(2.0, True)
a.clamp(lo, hi).backward()
assert a.grad == 0.5 and lo.grad == 0.5 and hi.grad == 0.0

a = Tensor(2.0, True)
lo = Tensor(0.0, True)
hi = Tensor(2.0, True)
a.clamp(lo, hi).backward(4.0)
assert a.grad == 2.0 and lo.grad == 0.0 and hi.grad == 2.0

# --- vector backward ---
a = Tensor([-2.0, 1.0, 9.0], True)
lo = Tensor([0.0, -1.0, 5.0], True)
hi = Tensor([2.0, 4.0, 8.0], True)
a.clamp(lo, hi).backward([1.0, 2.0, 4.0])
# index0 x<lower -> lower gets 1; index1 interior -> self gets 2;
# index2 x>upper -> upper gets 4
assert a.grad == [0.0, 2.0, 0.0]
assert lo.grad == [1.0, 0.0, 0.0]
assert hi.grad == [0.0, 0.0, 4.0]

# vector ties split elementwise
a = Tensor([0.0, 2.0, 1.0], True)
lo = Tensor([0.0, 0.0, 0.0], True)
hi = Tensor([2.0, 2.0, 2.0], True)
a.clamp(lo, hi).backward([2.0, 2.0, 2.0])
assert a.grad == [1.0, 1.0, 2.0]
assert lo.grad == [1.0, 0.0, 0.0]
assert hi.grad == [0.0, 1.0, 0.0]

# --- broadcast scalar parents reduce from 0.0 in ascending index order ---
# scalar self
a = Tensor(1.0, True)
a.clamp(
    Tensor([0.0, 5.0, 0.0]), Tensor([2.0, 6.0, 2.0])
).backward([1.0, 2.0, 4.0])
# index0 interior self +=1; index1 x<lower self 0; index2 interior self +=4
assert a.grad == 5.0

# scalar lower binds at multiple indices -> reduced
a = Tensor([-1.0, -2.0, 10.0], True)
lo = Tensor(0.0, True)
hi = Tensor(5.0, True)
a.clamp(lo, hi).backward([1.0, 2.0, 4.0])
assert a.grad == [0.0, 0.0, 0.0]
assert lo.grad == 3.0
assert hi.grad == 4.0

# scalar upper only
a = Tensor([-9.0, 9.0], True)
a.clamp(0.0, 2.0).backward([3.0, 5.0])
assert a.grad == [0.0, 0.0]

# self scalar split tie contributes 0.5 reductions
a = Tensor(0.0, True)
lo = Tensor([0.0, 0.0], True)
hi = Tensor([2.0, 2.0], True)
a.clamp(lo, hi).backward([2.0, 4.0])
assert a.grad == 3.0
assert lo.grad == [1.0, 2.0]
assert hi.grad == [0.0, 0.0]

# --- only some parents require grad ---
a = Tensor([-1.0, 1.0, 3.0], True)
r = a.clamp(0.0, 2.0)
assert len(r._parents) == 3
r.backward([1.0, 1.0, 1.0])
assert a.grad == [0.0, 1.0, 0.0]

a = Tensor([-1.0, 3.0], True)
lo = Tensor(0.0, True)
r = a.clamp(lo, 2.0)
r.backward([2.0, 3.0])
assert a.grad == [0.0, 0.0]
assert lo.grad == 2.0

# --- backward grad validation against output shape ---
a = Tensor([-1.0, 1.0], True)
r = a.clamp(0.0, 2.0)
expect(ValueError, lambda: r.backward())
expect(ValueError, lambda: r.backward(1.0))
expect(ValueError, lambda: r.backward([1.0]))
expect(TypeError, lambda: r.backward(None))
expect(TypeError, lambda: r.backward(True))
expect(TypeError, lambda: r.backward(1))
expect(TypeError, lambda: r.backward("x"))
expect(TypeError, lambda: r.backward([1.0, 2]))
expect(ValueError, lambda: r.backward([1.0, float("nan")]))
s = Tensor(-1.0, True).clamp(0.0, 2.0)
expect(ValueError, lambda: s.backward([1.0]))
expect(TypeError, lambda: s.backward(None))
expect(ValueError, lambda: s.backward(float("inf")))
assert a.grad is None

# --- forward snapshots operands, shapes and branches ---
a = Tensor([-1.0, 1.0, 3.0], True)
lo = Tensor(0.0, True)
hi = Tensor(2.0, True)
r = a.clamp(lo, hi)
a.data = [9.0, 9.0, 9.0]
lo.data = 100.0
hi.data = -100.0
assert r.data == [0.0, 1.0, 2.0]
r.backward([1.0, 1.0, 1.0])
assert a.grad == [0.0, 1.0, 0.0]
assert lo.grad == 1.0
assert hi.grad == 1.0

# --- same tensor in multiple roles merges contributions ---
# self == lower: each index at the tie splits 0.5 to each role and merges to 1
x = Tensor([1.0, 2.0], True)
hi = Tensor(5.0, True)
x.clamp(x, hi).backward([2.0, 4.0])
# each x is at its own lower tie: self 0.5 + lower 0.5 == 1.0 * g
assert x.grad == [2.0, 4.0]
assert hi.grad == 0.0

# self == upper tie
x = Tensor([1.0, 2.0], True)
lo = Tensor(-5.0, True)
x.clamp(lo, x).backward([2.0, 4.0])
assert x.grad == [2.0, 4.0]
assert lo.grad == 0.0

# --- shared path and repeated backward accumulate ---
a = Tensor([-1.0, 1.0], True)
r = a.clamp(0.0, 2.0)
r.backward([1.0, 1.0])
r.backward([1.0, 1.0])
assert a.grad == [0.0, 2.0]

# --- non-finite backward intermediate / existing grad merge -> V, untouched ---
a = Tensor(2.0, True)
a.grad = float("inf")
expect(ValueError, lambda: a.clamp(Tensor(0.0, True), 2.0).backward())
assert a.grad == float("inf")

# failure leaves whole graph grad unchanged
a = Tensor(-1.0, True)
lo = Tensor(0.0, True)
r = a.clamp(lo, 2.0)
expect(ValueError, lambda: r.backward(float("nan")))
assert a.grad is None and lo.grad is None

# --- gradcheck numeric agreement (interior region differentiable in self) ---
passed, err = gradcheck(
    lambda t: t.clamp(-10.0, 10.0).sum(), [1.0, 2.0, 3.0]
)
assert passed, err

# differentiable in a scalar lower when it strictly binds
passed, err = gradcheck(
    lambda l: Tensor([-5.0, -4.0]).clamp(l, 10.0).sum(), 0.0
)
assert passed, err

# differentiable in a scalar upper when it strictly binds
passed, err = gradcheck(
    lambda u: Tensor([15.0, 16.0]).clamp(0.0, u).sum(), 10.0
)
assert passed, err

print("all clamp tests passed")
