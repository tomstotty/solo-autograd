import math

from autograd import Tensor


def expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no error raised")


# --- forward: condition type validation ---
t = Tensor(1.0)
expect(TypeError, lambda: t.where(1, 2.0))            # int -> T
expect(TypeError, lambda: t.where(1.0, 2.0))          # float -> T
expect(TypeError, lambda: t.where("x", 2.0))
expect(TypeError, lambda: t.where(None, 2.0))
expect(TypeError, lambda: t.where((True,), 2.0))      # tuple -> T
expect(TypeError, lambda: t.where([True, 0], 2.0))    # int element -> T
expect(TypeError, lambda: t.where([True, 1.0], 2.0))  # float element -> T
expect(ValueError, lambda: t.where([], 2.0))          # empty -> V

# --- forward: other type validation ---
expect(TypeError, lambda: t.where(True, True))        # bool -> T
expect(TypeError, lambda: t.where(True, 2))           # int -> T
expect(TypeError, lambda: t.where(True, "2.0"))
expect(TypeError, lambda: t.where(True, None))
expect(ValueError, lambda: t.where(True, float("inf")))
expect(ValueError, lambda: t.where(True, float("nan")))

# --- forward: data validation (mutated after construction) ---
bad = Tensor(1.0)
bad.data = 1                      # int scalar -> T
expect(TypeError, lambda: bad.where(True, 2.0))
bad.data = [1.0, True]            # bad element -> T
expect(TypeError, lambda: bad.where([True, False], 2.0))
bad.data = []                     # empty -> V
expect(ValueError, lambda: bad.where(True, 2.0))
bad.data = float("nan")           # non-finite -> V
expect(ValueError, lambda: bad.where(True, 2.0))
bad.data = 2.0
bad.requires_grad = 1             # non-bool -> T
expect(TypeError, lambda: bad.where(True, 2.0))
other_bad = Tensor(2.0)
other_bad.requires_grad = "x"
expect(TypeError, lambda: Tensor(1.0).where(True, other_bad))

# --- forward: selection / broadcasting / shapes ---
assert Tensor(5.0).where(True, 9.0).data == 5.0
assert Tensor(5.0).where(False, 9.0).data == 9.0
assert Tensor([1.0, 2.0, 3.0]).where([True, False, True], 0.0).data == [
    1.0, 0.0, 3.0
]
assert Tensor(1.0).where(
    [True, False, False], Tensor([4.0, 5.0, 6.0])
).data == [1.0, 5.0, 6.0]
assert Tensor([1.0, 2.0]).where(True, Tensor([7.0, 8.0])).data == [1.0, 2.0]
assert Tensor([1.0, 2.0]).where(False, Tensor([7.0, 8.0])).data == [7.0, 8.0]
# any list side makes the output a vector; all lists must match in length
expect(
    ValueError,
    lambda: Tensor([1.0, 2.0]).where([True, False, False], 0.0),
)
expect(
    ValueError,
    lambda: Tensor([1.0, 2.0]).where(
        [True, False], Tensor([1.0, 2.0, 3.0])
    ),
)

# --- no graph when neither side requires grad ---
r = Tensor(1.0).where(True, 2.0)
assert r.requires_grad is False and r._parents == ()
expect(ValueError, lambda: r.backward())

# --- backward: scalar / scalar ---
a = Tensor(5.0, True)
b = Tensor(9.0, True)
out = a.where(False, b)
assert out.data == 9.0 and out.requires_grad is True
out.backward()
assert a.grad == 0.0 and b.grad == 1.0
a = Tensor(5.0, True)
b = Tensor(9.0, True)
a.where(True, b).backward()
assert a.grad == 1.0 and b.grad == 0.0

# --- backward: grad validation ---
a = Tensor(5.0, True)
b = Tensor(9.0, True)
out = a.where(True, b)
expect(TypeError, lambda: out.backward(None))
expect(TypeError, lambda: out.backward(True))
expect(TypeError, lambda: out.backward(1))
expect(TypeError, lambda: out.backward("x"))
expect(ValueError, lambda: out.backward(float("inf")))
expect(ValueError, lambda: out.backward([1.0]))   # list for scalar -> V
out.backward()                                     # omitted -> 1.0
assert a.grad == 1.0

av = Tensor([1.0, 2.0, 3.0], True)
bv = Tensor(0.0, True)
outv = av.where([True, False, True], bv)
expect(ValueError, lambda: outv.backward())           # vector omitted -> V
expect(ValueError, lambda: outv.backward(1.0))        # scalar for vector -> V
expect(ValueError, lambda: outv.backward([1.0, 1.0]))  # wrong length -> V
expect(TypeError, lambda: outv.backward([1.0, 1.0, 2]))    # int element -> T
expect(TypeError, lambda: outv.backward([1.0, 1.0, None]))
expect(ValueError, lambda: outv.backward([1.0, float("nan"), 1.0]))
outv.backward([1.0, 2.0, 4.0])
assert av.grad == [1.0, 0.0, 4.0]
assert bv.grad == 2.0

# --- backward: scalar self broadcast over vector other ---
a = Tensor(7.0, True)
b = Tensor([1.0, 2.0, 3.0], True)
a.where([False, True, False], b).backward([10.0, 10.0, 10.0])
assert a.grad == 10.0
assert b.grad == [10.0, 0.0, 10.0]

# --- backward: vector self, scalar other broadcast ---
a = Tensor([1.0, 2.0, 3.0], True)
b = Tensor(9.0, True)
a.where([True, False, True], b).backward([1.0, 1.0, 1.0])
assert a.grad == [1.0, 0.0, 1.0]
assert b.grad == 1.0

# --- only requires-grad side behavior; other still gets 0.0 when tracked ---
a = Tensor(5.0, True)
b = Tensor(9.0)
a.where(False, b).backward()
assert a.grad == 0.0 and b.grad is None

# --- same tensor on both sides merges ---
x = Tensor(4.0, True)
x.where(True, x).backward()
assert x.grad == 1.0
x = Tensor([1.0, 2.0], True)
x.where([False, True], x).backward([1.0, 1.0])
assert x.grad == [1.0, 1.0]

# --- snapshot: mutation/replacement after forward does not affect backward ---
a = Tensor([1.0, 2.0], True)
b = Tensor([3.0, 4.0], True)
condition = [True, False]
out = a.where(condition, b)
condition[0] = False
a.data[0] = 99.0
b.data = [0.0, 0.0]
out.backward([1.0, 1.0])
assert a.grad == [1.0, 0.0]
assert b.grad == [0.0, 1.0]

# --- repeated backward accumulates ---
a = Tensor(5.0, True)
a.where(True, Tensor(2.0)).backward()
a.where(True, Tensor(2.0)).backward()
assert a.grad == 2.0

# --- shared path accumulates ---
x = Tensor([1.0, 2.0], True)
w = Tensor([3.0, 4.0], True)
y = x.where([True, True], Tensor(0.0)).mul(w)
y.backward([1.0, 1.0])
assert x.grad == [3.0, 4.0]
assert w.grad == [1.0, 2.0]

# --- non-finite backward grad -> V, grads untouched ---
a = Tensor(1.0, True)
out = a.where(True, Tensor(2.0, True))
expect(ValueError, lambda: out.backward(float("inf")))
assert a.grad is None

# --- existing grad merge non-finite -> V, untouched ---
a = Tensor(1.0, True)
a.grad = float("inf")
expect(ValueError, lambda: a.where(True, Tensor(2.0, True)).backward())
assert a.grad == float("inf")

print("all where tests passed")
