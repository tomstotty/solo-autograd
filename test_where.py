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


# --- forward: condition type validation ---
t = Tensor(1.0)
expect(TypeError, lambda: t.where(1, 2.0))
expect(TypeError, lambda: t.where("x", 2.0))
expect(TypeError, lambda: t.where(None, 2.0))
expect(TypeError, lambda: t.where((True, False), 2.0))
expect(TypeError, lambda: t.where([True, 0], 2.0))
expect(TypeError, lambda: t.where([1.0, 0.0], 2.0))
expect(TypeError, lambda: t.where([True, "x"], 2.0))
expect(ValueError, lambda: t.where([], 2.0))

# --- forward: other type validation ---
expect(TypeError, lambda: t.where(True, True))
expect(TypeError, lambda: t.where(True, 2))
expect(TypeError, lambda: t.where(True, "2.0"))
expect(TypeError, lambda: t.where(True, None))
expect(ValueError, lambda: t.where(True, float("inf")))
expect(ValueError, lambda: t.where(True, float("nan")))

# --- forward: data validation (mutated after construction) ---
bad = Tensor(1.0)
bad.data = 1
expect(TypeError, lambda: bad.where(True, 2.0))
bad.data = [1.0, True]
expect(TypeError, lambda: bad.where([True, False], 2.0))
bad.data = []
expect(ValueError, lambda: bad.where([True], 2.0))
bad.data = float("inf")
expect(ValueError, lambda: bad.where(True, 2.0))
bad.data = 1.0
bad.requires_grad = 1
expect(TypeError, lambda: bad.where(True, 2.0))
ob = Tensor(2.0)
ob.data = 1
expect(TypeError, lambda: Tensor(1.0).where(True, ob))
ob.data = []
expect(ValueError, lambda: Tensor([1.0]).where([True], ob))
ob.data = float("nan")
expect(ValueError, lambda: Tensor(1.0).where(True, ob))
ob.data = 2.0
ob.requires_grad = "x"
expect(TypeError, lambda: Tensor(1.0).where(True, ob))

# --- forward: selection and broadcasting ---
assert Tensor(1.0).where(True, 2.0).data == 1.0
assert Tensor(1.0).where(False, 2.0).data == 2.0
assert Tensor(1.0).where(False, Tensor(2.0)).data == 2.0
assert Tensor([1.0, 2.0, 3.0]).where(
    [True, False, True], 10.0
).data == [1.0, 10.0, 3.0]
assert Tensor(1.0).where(
    [True, False, False], Tensor([4.0, 5.0, 6.0])
).data == [1.0, 5.0, 6.0]
assert Tensor([1.0, 2.0]).where(False, Tensor([4.0, 5.0])).data == [4.0, 5.0]
assert Tensor([1.0, 2.0]).where(True, 5.0).data == [1.0, 2.0]
expect(ValueError, lambda: Tensor([1.0, 2.0]).where([True], 3.0))
expect(ValueError, lambda: Tensor([1.0, 2.0]).where([True, False, False], 3.0))
expect(
    ValueError,
    lambda: Tensor([1.0, 2.0]).where(True, Tensor([1.0, 2.0, 3.0])),
)

# --- no graph when neither side requires grad ---
r = Tensor(1.0).where(True, 2.0)
assert r.requires_grad is False and r._parents == ()
expect(ValueError, lambda: r.backward())
rv = Tensor([1.0, 2.0]).where([True, False], 3.0)
assert rv.requires_grad is False and rv._parents == ()
expect(ValueError, lambda: rv.backward([1.0, 1.0]))

# --- scalar backward ---
a = Tensor(3.0, True)
b = Tensor(7.0, True)
Tensor.where  # attribute sanity
a.where(True, b).backward()
assert a.grad == 1.0 and b.grad == 0.0
a.zero_grad()
b.zero_grad()
a.where(False, b).backward()
assert a.grad == 0.0 and b.grad == 1.0
a.zero_grad()
b.zero_grad()
a.where(False, b).backward(2.5)
assert a.grad == 0.0 and b.grad == 2.5

# --- vector backward ---
a = Tensor([1.0, 2.0, 3.0], True)
b = Tensor([4.0, 5.0, 6.0], True)
a.where([True, False, True], b).backward([1.0, 2.0, 4.0])
assert a.grad == [1.0, 0.0, 4.0]
assert b.grad == [0.0, 2.0, 0.0]

# --- broadcast scalar parents reduce from 0.0 by index ---
a = Tensor(5.0, True)
a.where([True, False, True], 9.0).backward([1.0, 2.0, 4.0])
assert a.grad == 5.0
a.zero_grad()
a.where([False, False, True], 9.0).backward([1.0, 2.0, 4.0])
assert a.grad == 4.0
b = Tensor(8.0, True)
Tensor(1.0, True).where([True, False, False], b).backward([3.0, 1.0, 1.0])
assert b.grad == 2.0

# --- only one side requiring grad ---
a = Tensor([1.0, 2.0], True)
r = a.where([True, False], 5.0)
assert len(r._parents) == 2
r.backward([2.0, 3.0])
assert a.grad == [2.0, 0.0]

# --- backward grad validation ---
a = Tensor([1.0, 2.0], True)
r = a.where([True, False], 5.0)
expect(ValueError, lambda: r.backward())
expect(ValueError, lambda: r.backward(2.0))
expect(ValueError, lambda: r.backward([1.0]))
expect(TypeError, lambda: r.backward(None))
expect(TypeError, lambda: r.backward(True))
expect(TypeError, lambda: r.backward(1))
expect(TypeError, lambda: r.backward("x"))
expect(TypeError, lambda: r.backward([1.0, 2]))
expect(ValueError, lambda: r.backward([1.0, float("nan")]))
s = Tensor(1.0, True).where(True, 2.0)
expect(ValueError, lambda: s.backward([1.0]))
expect(TypeError, lambda: s.backward(None))
expect(ValueError, lambda: s.backward(float("inf")))
assert a.grad is None

# --- forward snapshots condition and both shapes ---
cond = [True, False]
a = Tensor([1.0, 2.0], True)
b = Tensor([3.0, 4.0], True)
r = a.where(cond, b)
cond[0] = False
a.data = [9.0, 9.0]
b.data = [9.0, 9.0]
assert r.data == [1.0, 4.0]
r.backward([1.0, 1.0])
assert a.grad == [1.0, 0.0]
assert b.grad == [0.0, 1.0]

# --- same tensor on both sides merges ---
x = Tensor([1.0, 2.0], True)
x.where([True, False], x).backward([2.0, 3.0])
assert x.grad == [2.0, 3.0]
x = Tensor(5.0, True)
x.where([True, False, True], x).backward([2.0, 3.0, 4.0])
assert x.grad == 9.0

# --- repeated backward accumulates ---
a = Tensor([1.0, 2.0], True)
r = a.where([True, False], 8.0)
r.backward([1.0, 1.0])
r.backward([1.0, 1.0])
assert a.grad == [2.0, 0.0]

# --- non-finite existing grad merge -> V, untouched ---
a = Tensor(2.0, True)
a.grad = float("inf")
expect(ValueError, lambda: a.where(True, Tensor(3.0, True)).backward())
assert a.grad == float("inf")

# --- gradcheck numeric agreement ---
passed, err = gradcheck(
    lambda t: t.where([True, False, True], 0.0).sum(), [1.5, 2.5, 3.5]
)
assert passed, err
passed, err = gradcheck(
    lambda u: Tensor(1.0).where([False, True, False], u).sum(),
    [1.5, 2.5, 3.5],
)
assert passed, err
passed, err = gradcheck(lambda t: t.where([True, False], t).sum(), [1.5, 2.5])
assert passed, err

print("all where tests passed")
