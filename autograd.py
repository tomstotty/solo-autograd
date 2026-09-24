#!/usr/bin/env python3
"""Minimal autograd framework: finite scalar / 1D float tensors."""

import hashlib
import json
import math
import os
import re
import sys
import tempfile


_MISSING = object()
"""Module-private unique sentinel marking an omitted backward() grad."""


def _as_finite_float(value):
    """Validate a scalar float; TypeError on bad type, ValueError if non-finite."""
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError("expected a finite float")
    if not math.isfinite(value):
        raise ValueError("expected a finite float")
    return value


def _validate_data(data):
    """Return data normalized to a float scalar or a non-empty list of floats."""
    if isinstance(data, bool):
        raise TypeError(
            "data must be a finite float scalar or a non-empty 1D float list"
        )
    if isinstance(data, float):
        if not math.isfinite(data):
            raise ValueError("data must be finite")
        return data
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("data list must be non-empty")
        return [_as_finite_float(x) for x in data]
    raise TypeError(
        "data must be a finite float scalar or a non-empty 1D float list"
    )


def _ensure_finite_data(data):
    """ValueError if a forward result contains a non-finite value."""
    values = data if isinstance(data, list) else [data]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("forward result must be finite")


def _ensure_finite_grad(value):
    """ValueError if a gradient to accumulate contains a non-finite value."""
    values = value if isinstance(value, list) else [value]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("gradient must be finite")


def _finite_power(base, exponent, message="power result must be finite"):
    """Raise base to exponent, mapping overflow/non-finite to ValueError."""
    try:
        result = base ** exponent
    except OverflowError:
        raise ValueError(message)
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def _finite_exp(x, message="exp result must be finite"):
    """Evaluate math.exp, mapping overflow/non-finite to ValueError."""
    try:
        result = math.exp(x)
    except OverflowError:
        raise ValueError(message)
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def _broadcast_apply(a, b, op):
    """Apply op over scalar/vector operands with scalar broadcasting."""
    a_vector = isinstance(a, list)
    b_vector = isinstance(b, list)
    if not a_vector and not b_vector:
        return op(a, b)
    if a_vector and not b_vector:
        return [op(x, b) for x in a]
    if not a_vector and b_vector:
        return [op(a, y) for y in b]
    if len(a) != len(b):
        raise ValueError("vector lengths must match")
    return [op(x, y) for x, y in zip(a, b)]


def _map_unary(data, op):
    """Apply op elementwise over a scalar or vector."""
    if isinstance(data, list):
        return [op(x) for x in data]
    return op(data)


def _reduce_like(parent_data, value):
    """Sum a broadcast gradient back down to the parent's shape."""
    if not isinstance(parent_data, list) and isinstance(value, list):
        return sum(value)
    return value


def _merge_grad(a, b):
    """Combine two gradient contributions of identical shape."""
    if isinstance(a, list):
        return [x + y for x, y in zip(a, b)]
    return a + b


def _sub_grad(a, b):
    """Subtract two gradient values of identical shape."""
    if isinstance(a, list):
        return [x - y for x, y in zip(a, b)]
    return a - b


def _same_grad_shape(a, b):
    """True when two gradient values have matching scalar/vector shape."""
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
        )
    return True


def _validate_grad(grad, like):
    """Validate an externally supplied outgoing gradient against tensor data."""
    if isinstance(grad, bool):
        raise TypeError("grad must be a float or a list of floats")
    if isinstance(like, float):
        if isinstance(grad, list):
            raise ValueError("grad shape must match tensor shape")
        if not isinstance(grad, float):
            raise TypeError("grad must be a float for a scalar tensor")
        if not math.isfinite(grad):
            raise ValueError("grad must be finite")
        return grad
    if isinstance(grad, list):
        if len(grad) != len(like):
            raise ValueError("grad shape must match tensor shape")
        return [_as_finite_float(x) for x in grad]
    if isinstance(grad, float):
        raise ValueError("grad shape must match tensor shape")
    raise TypeError("grad must be a float or a list of floats")


def _require_nonempty_float_vector(tensor, op):
    """Validate the shared log_softmax/cross_entropy preconditions."""
    data = tensor.data
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(op + " requires a non-empty 1D float list")
    for value in data:
        if isinstance(value, bool) or not isinstance(value, float):
            raise TypeError(op + " data elements must be floats")
    if not isinstance(tensor.requires_grad, bool):
        raise TypeError("requires_grad must be a bool")
    for value in data:
        if not math.isfinite(value):
            raise ValueError(op + " data elements must be finite")
    return data


def _stable_sigmoid_value(x):
    """Evaluate sigmoid(x) stably, never exponentiating a large positive.

    For x >= 0: e = exp(-x), d = 1 + e, y = 1/d.
    For x < 0:  e = exp(x),  d = 1 + e, y = e/d.
    ValueError if any of e, d, y is non-finite.
    """
    if x >= 0.0:
        e = math.exp(-x)
        if not math.isfinite(e):
            raise ValueError("sigmoid intermediate must be finite")
        d = 1.0 + e
        if not math.isfinite(d):
            raise ValueError("sigmoid intermediate must be finite")
        y = 1.0 / d
        if not math.isfinite(y):
            raise ValueError("sigmoid intermediate must be finite")
        return y
    e = math.exp(x)
    if not math.isfinite(e):
        raise ValueError("sigmoid intermediate must be finite")
    d = 1.0 + e
    if not math.isfinite(d):
        raise ValueError("sigmoid intermediate must be finite")
    y = e / d
    if not math.isfinite(y):
        raise ValueError("sigmoid intermediate must be finite")
    return y


def _log_softmax_values(data):
    """Compute log-softmax values stably.

    Shift by the maximum so exp is only ever evaluated on values in
    (-inf, 0]; exp(x_i) directly is never computed.
    """
    m = max(data)
    diffs = []
    z = []
    for value in data:
        diff = value - m
        if not math.isfinite(diff):
            raise ValueError("log_softmax intermediate must be finite")
        diffs.append(diff)
        z_i = math.exp(diff)
        if not math.isfinite(z_i):
            raise ValueError("log_softmax intermediate must be finite")
        z.append(z_i)
    s = sum(z)
    if not math.isfinite(s):
        raise ValueError("log_softmax intermediate must be finite")
    log_s = math.log(s)
    if not math.isfinite(log_s):
        raise ValueError("log_softmax intermediate must be finite")
    out = []
    for diff in diffs:
        l_i = diff - log_s
        if not math.isfinite(l_i):
            raise ValueError("log_softmax intermediate must be finite")
        out.append(l_i)
    return out


class Tensor:
    def __init__(self, data, requires_grad=False):
        normalized = _validate_data(data)
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        self.data = normalized
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = ()
        self._backward_fn = None
        self._backward_record = None

    @classmethod
    def _make(cls, data, requires_grad, parents, backward_fn):
        obj = cls.__new__(cls)
        obj.data = data
        obj.requires_grad = requires_grad
        obj.grad = None
        obj._parents = parents
        obj._backward_fn = backward_fn
        obj._backward_record = None
        return obj

    def _coerce(self, other):
        if isinstance(other, Tensor):
            return other
        if isinstance(other, bool) or not isinstance(other, float):
            raise TypeError("other must be a Tensor or a finite float")
        if not math.isfinite(other):
            raise ValueError("other must be finite")
        return Tensor(other)

    def add(self, other):
        other = self._coerce(other)
        out_data = _broadcast_apply(
            self.data, other.data, lambda x, y: x + y
        )
        _ensure_finite_data(out_data)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other

        def backward_fn(grad):
            return [
                (parent_self, _reduce_like(parent_self.data, grad)),
                (parent_other, _reduce_like(parent_other.data, grad)),
            ]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def sub(self, other):
        other = self._coerce(other)
        a = _validate_data(self.data)
        b = _validate_data(other.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(other.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        a_vector = isinstance(a, list)
        b_vector = isinstance(b, list)
        if a_vector and b_vector and len(a) != len(b):
            raise ValueError("vector lengths must match")
        # Subtract element by element in ascending output-index order; a
        # non-finite difference aborts before a result tensor exists, so no
        # state can change on failure.
        if a_vector and b_vector:
            out_data = []
            for i in range(len(a)):
                difference = a[i] - b[i]
                if not math.isfinite(difference):
                    raise ValueError("sub result must be finite")
                out_data.append(difference)
        elif a_vector:
            out_data = []
            for i in range(len(a)):
                difference = a[i] - b
                if not math.isfinite(difference):
                    raise ValueError("sub result must be finite")
                out_data.append(difference)
        elif b_vector:
            out_data = []
            for i in range(len(b)):
                difference = a - b[i]
                if not math.isfinite(difference):
                    raise ValueError("sub result must be finite")
                out_data.append(difference)
        else:
            out_data = a - b
            if not math.isfinite(out_data):
                raise ValueError("sub result must be finite")
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        # Snapshot both operands so later caller-side mutation or
        # replacement of either input cannot change what a pending
        # backward pass uses.
        snapshot_a = list(a) if a_vector else a
        snapshot_b = list(b) if b_vector else b

        def backward_fn(grad):
            grad_values = grad if isinstance(grad, list) else [grad]
            contributions = []
            if parent_self.requires_grad:
                if a_vector:
                    da = []
                    for i in range(len(snapshot_a)):
                        value = grad_values[i]
                        if not math.isfinite(value):
                            raise ValueError(
                                "sub backward intermediate must be finite"
                            )
                        da.append(value)
                    contributions.append((parent_self, da))
                else:
                    # A broadcast scalar parent reduces by accumulating
                    # from 0.0 in ascending output-index order.
                    total = 0.0
                    for i in range(len(grad_values)):
                        total += grad_values[i]
                        if not math.isfinite(total):
                            raise ValueError(
                                "sub backward intermediate must be finite"
                            )
                    contributions.append((parent_self, total))
            if parent_other.requires_grad:
                if b_vector:
                    db = []
                    for i in range(len(snapshot_b)):
                        value = -grad_values[i]
                        if not math.isfinite(value):
                            raise ValueError(
                                "sub backward intermediate must be finite"
                            )
                        db.append(value)
                    contributions.append((parent_other, db))
                else:
                    total = 0.0
                    for i in range(len(grad_values)):
                        total += -grad_values[i]
                        if not math.isfinite(total):
                            raise ValueError(
                                "sub backward intermediate must be finite"
                            )
                    contributions.append((parent_other, total))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def mul(self, other):
        other = self._coerce(other)
        out_data = _broadcast_apply(
            self.data, other.data, lambda x, y: x * y
        )
        _ensure_finite_data(out_data)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        a, b = parent_self.data, parent_other.data

        def backward_fn(grad):
            grad_self = _broadcast_apply(grad, b, lambda x, y: x * y)
            grad_other = _broadcast_apply(grad, a, lambda x, y: x * y)
            return [
                (parent_self, _reduce_like(a, grad_self)),
                (parent_other, _reduce_like(b, grad_other)),
            ]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def div(self, other):
        other = self._coerce(other)
        a = _validate_data(self.data)
        b = _validate_data(other.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(other.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        a_vector = isinstance(a, list)
        b_vector = isinstance(b, list)
        if a_vector and b_vector and len(a) != len(b):
            raise ValueError("vector lengths must match")
        denominators = b if b_vector else [b]
        for denominator in denominators:
            if denominator == 0.0:
                raise ValueError("divisor must be non-zero")
        # Divide element by element in ascending output-index order; a
        # non-finite quotient aborts before a result tensor exists, so no
        # state can change on failure.
        if a_vector and b_vector:
            out_data = []
            for i in range(len(a)):
                quotient = a[i] / b[i]
                if not math.isfinite(quotient):
                    raise ValueError("div result must be finite")
                out_data.append(quotient)
        elif a_vector:
            out_data = []
            for i in range(len(a)):
                quotient = a[i] / b
                if not math.isfinite(quotient):
                    raise ValueError("div result must be finite")
                out_data.append(quotient)
        elif b_vector:
            out_data = []
            for i in range(len(b)):
                quotient = a / b[i]
                if not math.isfinite(quotient):
                    raise ValueError("div result must be finite")
                out_data.append(quotient)
        else:
            out_data = a / b
            if not math.isfinite(out_data):
                raise ValueError("div result must be finite")
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        # Snapshot both operands so later caller-side mutation or
        # replacement of either input cannot change what a pending
        # backward pass uses.
        snapshot_a = list(a) if a_vector else a
        snapshot_b = list(b) if b_vector else b

        def backward_fn(grad):
            grad_values = grad if isinstance(grad, list) else [grad]
            contributions = []
            if parent_self.requires_grad:
                if a_vector:
                    da = []
                    for i in range(len(snapshot_a)):
                        denominator = (
                            snapshot_b[i] if b_vector else snapshot_b
                        )
                        value = grad_values[i] / denominator
                        if not math.isfinite(value):
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                        da.append(value)
                    contributions.append((parent_self, da))
                else:
                    # A broadcast scalar parent reduces by accumulating
                    # from 0.0 in ascending output-index order.
                    total = 0.0
                    for i in range(len(grad_values)):
                        denominator = (
                            snapshot_b[i] if b_vector else snapshot_b
                        )
                        value = grad_values[i] / denominator
                        if not math.isfinite(value):
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                    contributions.append((parent_self, total))
            if parent_other.requires_grad:
                if b_vector:
                    db = []
                    for i in range(len(snapshot_b)):
                        numerator = (
                            snapshot_a[i] if a_vector else snapshot_a
                        )
                        product = grad_values[i] * numerator
                        if not math.isfinite(product):
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                        square = snapshot_b[i] * snapshot_b[i]
                        if not math.isfinite(square) or square == 0.0:
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                        quotient = product / square
                        if not math.isfinite(quotient):
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                        db.append(-quotient)
                    contributions.append((parent_other, db))
                else:
                    square = snapshot_b * snapshot_b
                    if not math.isfinite(square) or square == 0.0:
                        raise ValueError(
                            "div backward intermediate must be finite"
                        )
                    total = 0.0
                    for i in range(len(grad_values)):
                        numerator = (
                            snapshot_a[i] if a_vector else snapshot_a
                        )
                        product = grad_values[i] * numerator
                        if not math.isfinite(product):
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                        quotient = product / square
                        if not math.isfinite(quotient):
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                        total += -quotient
                        if not math.isfinite(total):
                            raise ValueError(
                                "div backward intermediate must be finite"
                            )
                    contributions.append((parent_other, total))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def maximum(self, other):
        # other is accepted as a Tensor or a finite float only; bools, ints
        # and anything else are a TypeError, and a non-finite float is a
        # ValueError.
        other = self._coerce(other)
        a = _validate_data(self.data)
        b = _validate_data(other.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(other.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        a_vector = isinstance(a, list)
        b_vector = isinstance(b, list)
        if a_vector and b_vector and len(a) != len(b):
            raise ValueError("vector lengths must match")
        # Select the larger value at every output index in ascending order;
        # a scalar operand broadcasts across a vector operand. winners
        # records, per output index, which side was larger: 1 for self, -1
        # for other and 0 for an exact tie. Inputs are already finite, so a
        # non-finite selection cannot occur; the explicit finite check still
        # aborts before a result tensor exists, leaving every input
        # untouched on failure.
        n_outputs = len(a) if a_vector else (len(b) if b_vector else 1)
        out_values = []
        winners = []
        for i in range(n_outputs):
            x = a[i] if a_vector else a
            y = b[i] if b_vector else b
            if x > y:
                out_values.append(x)
                winners.append(1)
            elif x < y:
                out_values.append(y)
                winners.append(-1)
            else:
                out_values.append(x)
                winners.append(0)
        out_data = out_values if (a_vector or b_vector) else out_values[0]
        _ensure_finite_data(out_data)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        # The winner list snapshots the per-index forward decision so later
        # caller-side mutation or replacement of either input cannot change
        # which side a pending backward pass credits.

        def backward_fn(grad):
            grad_values = grad if isinstance(grad, list) else [grad]
            da_values = [0.0] * n_outputs
            db_values = [0.0] * n_outputs
            # Credit the larger side with the whole outgoing element and the
            # smaller side with 0.0; on an exact tie each side gets 0.5*g.
            for i in range(n_outputs):
                winner = winners[i]
                g = grad_values[i]
                if winner >= 0:
                    value = g if winner == 1 else 0.5 * g
                    if not math.isfinite(value):
                        raise ValueError(
                            "maximum backward intermediate must be finite"
                        )
                    da_values[i] = value
                if winner <= 0:
                    value = g if winner == -1 else 0.5 * g
                    if not math.isfinite(value):
                        raise ValueError(
                            "maximum backward intermediate must be finite"
                        )
                    db_values[i] = value
            roles = []
            if parent_self.requires_grad:
                if a_vector:
                    roles.append((parent_self, da_values))
                else:
                    # A broadcast scalar parent reduces by accumulating
                    # from 0.0 in ascending output-index order.
                    total = 0.0
                    for value in da_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "maximum backward intermediate must be finite"
                            )
                    roles.append((parent_self, total))
            if parent_other.requires_grad:
                if b_vector:
                    roles.append((parent_other, db_values))
                else:
                    total = 0.0
                    for value in db_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "maximum backward intermediate must be finite"
                            )
                    roles.append((parent_other, total))
            # The same object may play both sides; merge such roles into one
            # contribution per tensor, submitting only parents that require
            # grad.
            merged = {}
            for parent, value in roles:
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def minimum(self, other):
        # other is accepted as a Tensor or a finite float only; bools, ints
        # and anything else are a TypeError, and a non-finite float is a
        # ValueError.
        other = self._coerce(other)
        a = _validate_data(self.data)
        b = _validate_data(other.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(other.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        a_vector = isinstance(a, list)
        b_vector = isinstance(b, list)
        if a_vector and b_vector and len(a) != len(b):
            raise ValueError("vector lengths must match")
        # Select the smaller value at every output index in ascending order;
        # a scalar operand broadcasts across a vector operand. winners
        # records, per output index, which side was smaller: 1 for self, -1
        # for other and 0 for an exact tie. Inputs are already finite, so a
        # non-finite selection cannot occur; the explicit finite check still
        # aborts before a result tensor exists, leaving every input
        # untouched on failure.
        n_outputs = len(a) if a_vector else (len(b) if b_vector else 1)
        out_values = []
        winners = []
        for i in range(n_outputs):
            x = a[i] if a_vector else a
            y = b[i] if b_vector else b
            if x < y:
                out_values.append(x)
                winners.append(1)
            elif x > y:
                out_values.append(y)
                winners.append(-1)
            else:
                out_values.append(x)
                winners.append(0)
        out_data = out_values if (a_vector or b_vector) else out_values[0]
        _ensure_finite_data(out_data)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        # The winner list snapshots the per-index forward decision so later
        # caller-side mutation or replacement of either input cannot change
        # which side a pending backward pass credits.

        def backward_fn(grad):
            grad_values = grad if isinstance(grad, list) else [grad]
            da_values = [0.0] * n_outputs
            db_values = [0.0] * n_outputs
            # Credit the smaller side with the whole outgoing element and
            # the larger side with 0.0; on an exact tie each side gets
            # 0.5*g.
            for i in range(n_outputs):
                winner = winners[i]
                g = grad_values[i]
                if winner >= 0:
                    value = g if winner == 1 else 0.5 * g
                    if not math.isfinite(value):
                        raise ValueError(
                            "minimum backward intermediate must be finite"
                        )
                    da_values[i] = value
                if winner <= 0:
                    value = g if winner == -1 else 0.5 * g
                    if not math.isfinite(value):
                        raise ValueError(
                            "minimum backward intermediate must be finite"
                        )
                    db_values[i] = value
            roles = []
            if parent_self.requires_grad:
                if a_vector:
                    roles.append((parent_self, da_values))
                else:
                    # A broadcast scalar parent reduces by accumulating
                    # from 0.0 in ascending output-index order.
                    total = 0.0
                    for value in da_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "minimum backward intermediate must be finite"
                            )
                    roles.append((parent_self, total))
            if parent_other.requires_grad:
                if b_vector:
                    roles.append((parent_other, db_values))
                else:
                    total = 0.0
                    for value in db_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "minimum backward intermediate must be finite"
                            )
                    roles.append((parent_other, total))
            # The same object may play both sides; merge such roles into one
            # contribution per tensor, submitting only parents that require
            # grad.
            merged = {}
            for parent, value in roles:
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def logaddexp(self, other):
        # other is accepted as a Tensor or a finite float only; bools, ints
        # and anything else are a TypeError, and a non-finite float is a
        # ValueError.
        other = self._coerce(other)
        a = _validate_data(self.data)
        b = _validate_data(other.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(other.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        a_vector = isinstance(a, list)
        b_vector = isinstance(b, list)
        if a_vector and b_vector and len(a) != len(b):
            raise ValueError("vector lengths must match")
        # Evaluate log(exp(x) + exp(y)) stably at every output index in
        # ascending order; a scalar operand broadcasts across a vector
        # operand. With m = max(x, y) the exponential is only ever
        # evaluated on values in (-inf, 0]: e is 1.0 on an exact tie,
        # 0.0 when d = min(x, y) - m is -inf, and exp(d) otherwise. d is
        # the one intermediate permitted to be non-finite; any other
        # non-finite intermediate or result aborts before a result tensor
        # exists, leaving every input untouched on failure.
        n_outputs = len(a) if a_vector else (len(b) if b_vector else 1)
        out_values = []
        weights_a = []
        weights_b = []
        for i in range(n_outputs):
            x = a[i] if a_vector else a
            y = b[i] if b_vector else b
            m = x if x >= y else y
            if x == y:
                e = 1.0
            else:
                d = (y if y < x else x) - m
                if d == float("-inf"):
                    e = 0.0
                elif math.isfinite(d):
                    e = math.exp(d)
                else:
                    raise ValueError(
                        "logaddexp intermediate must be finite"
                    )
            if not math.isfinite(e):
                raise ValueError("logaddexp intermediate must be finite")
            log1p_e = math.log1p(e)
            if not math.isfinite(log1p_e):
                raise ValueError("logaddexp intermediate must be finite")
            value = m + log1p_e
            if not math.isfinite(value):
                raise ValueError("logaddexp result must be finite")
            out_values.append(value)
            # The larger side keeps weight 1/(1+e) and the smaller side
            # e/(1+e); on an exact tie e is 1.0 and both get 0.5.
            denominator = 1.0 + e
            if not math.isfinite(denominator):
                raise ValueError("logaddexp intermediate must be finite")
            high = 1.0 / denominator
            low = e / denominator
            if not math.isfinite(high) or not math.isfinite(low):
                raise ValueError("logaddexp intermediate must be finite")
            if x >= y:
                weights_a.append(high)
                weights_b.append(low)
            else:
                weights_a.append(low)
                weights_b.append(high)
        out_data = out_values if (a_vector or b_vector) else out_values[0]
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        # The weight lists snapshot the per-index forward decision so
        # later caller-side mutation or replacement of either input
        # cannot change what a pending backward pass credits.

        def backward_fn(grad):
            grad_values = grad if isinstance(grad, list) else [grad]
            da_values = [0.0] * n_outputs
            db_values = [0.0] * n_outputs
            for i in range(n_outputs):
                g = grad_values[i]
                value = g * weights_a[i]
                if not math.isfinite(value):
                    raise ValueError(
                        "logaddexp backward intermediate must be finite"
                    )
                da_values[i] = value
                value = g * weights_b[i]
                if not math.isfinite(value):
                    raise ValueError(
                        "logaddexp backward intermediate must be finite"
                    )
                db_values[i] = value
            roles = []
            if parent_self.requires_grad:
                if a_vector:
                    roles.append((parent_self, da_values))
                else:
                    # A broadcast scalar parent reduces by accumulating
                    # from 0.0 in ascending output-index order.
                    total = 0.0
                    for value in da_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "logaddexp backward intermediate"
                                " must be finite"
                            )
                    roles.append((parent_self, total))
            if parent_other.requires_grad:
                if b_vector:
                    roles.append((parent_other, db_values))
                else:
                    total = 0.0
                    for value in db_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "logaddexp backward intermediate"
                                " must be finite"
                            )
                    roles.append((parent_other, total))
            # The same object may play both sides; merge such roles into
            # one contribution per tensor, submitting only parents that
            # require grad.
            merged = {}
            for parent, value in roles:
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def where(self, condition, other):
        # condition is a bool or a non-empty list of bools only; a bad
        # container or element type is a TypeError and an empty list is a
        # ValueError.
        if isinstance(condition, bool):
            cond_values = [condition]
            cond_vector = False
        elif isinstance(condition, list):
            if len(condition) == 0:
                raise ValueError("condition list must be non-empty")
            for flag in condition:
                if not isinstance(flag, bool):
                    raise TypeError("condition elements must be bools")
            # Snapshot the condition so later caller-side mutation cannot
            # change a pending backward pass.
            cond_values = condition[:]
            cond_vector = True
        else:
            raise TypeError(
                "condition must be a bool or a non-empty list of bools"
            )
        # other is a Tensor or a finite float only; bools, ints and anything
        # else are a TypeError, and a non-finite float is a ValueError.
        other = self._coerce(other)
        a = _validate_data(self.data)
        b = _validate_data(other.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(other.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        a_vector = isinstance(a, list)
        b_vector = isinstance(b, list)
        # The output is a vector whenever the condition or either datum is a
        # list; every list must share one length, while scalars and a scalar
        # condition broadcast. All scalar inputs produce a scalar output.
        lengths = []
        if cond_vector:
            lengths.append(len(cond_values))
        if a_vector:
            lengths.append(len(a))
        if b_vector:
            lengths.append(len(b))
        if lengths and any(length != lengths[0] for length in lengths):
            raise ValueError("vector lengths must match")
        n_outputs = lengths[0] if lengths else 1
        # Select self where the condition is true and other otherwise, in
        # ascending output-index order. selections snapshots the per-index
        # decision and a/b snapshot both sides' shapes, so later mutation of
        # the condition or of either datum leaves this graph unchanged.
        out_values = []
        for i in range(n_outputs):
            chosen = cond_values[i] if cond_vector else cond_values[0]
            if chosen:
                out_values.append(a[i] if a_vector else a)
            else:
                out_values.append(b[i] if b_vector else b)
        out_data = out_values if lengths else out_values[0]
        _ensure_finite_data(out_data)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        selections = (
            cond_values[:]
            if cond_vector
            else [cond_values[0]] * n_outputs
        )

        def backward_fn(grad):
            grad_values = grad if isinstance(grad, list) else [grad]
            da_values = [0.0] * n_outputs
            db_values = [0.0] * n_outputs
            # Hand each outgoing element to the side the condition selected
            # and give the other side 0.0.
            for i in range(n_outputs):
                g = grad_values[i]
                if selections[i]:
                    da_values[i] = g
                else:
                    db_values[i] = g
            roles = []
            if parent_self.requires_grad:
                if a_vector:
                    roles.append((parent_self, da_values))
                else:
                    # A broadcast scalar parent reduces by accumulating
                    # from 0.0 in ascending output-index order.
                    total = 0.0
                    for value in da_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "where backward intermediate must be finite"
                            )
                    roles.append((parent_self, total))
            if parent_other.requires_grad:
                if b_vector:
                    roles.append((parent_other, db_values))
                else:
                    total = 0.0
                    for value in db_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "where backward intermediate must be finite"
                            )
                    roles.append((parent_other, total))
            # The same object may play both sides; merge such roles into one
            # contribution per tensor, submitting only parents that require
            # grad.
            merged = {}
            for parent, value in roles:
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def clamp(self, lower, upper):
        # Each bound is accepted as a Tensor or a finite float only; bools,
        # ints and anything else are a TypeError, and a non-finite float is a
        # ValueError.
        lower = self._coerce(lower)
        upper = self._coerce(upper)
        x = _validate_data(self.data)
        lo = _validate_data(lower.data)
        hi = _validate_data(upper.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(lower.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(upper.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        x_vector = isinstance(x, list)
        lo_vector = isinstance(lo, list)
        hi_vector = isinstance(hi, list)
        # The output is a vector whenever any operand is a list; every list
        # must share one length, while scalars broadcast. All scalars produce
        # a scalar output.
        lengths = []
        if x_vector:
            lengths.append(len(x))
        if lo_vector:
            lengths.append(len(lo))
        if hi_vector:
            lengths.append(len(hi))
        if lengths and any(length != lengths[0] for length in lengths):
            raise ValueError("vector lengths must match")
        n_outputs = lengths[0] if lengths else 1
        # Clamp in ascending output-index order, recording the branch taken
        # per index: -1 for x < lower (the lower bound binds), 0 for
        # lower <= x <= upper (x passes through) and 1 for x > upper (the
        # upper bound binds). Each position must satisfy lower < upper; a
        # non-finite result or an inverted/empty range aborts before a result
        # tensor exists, leaving every input untouched on failure.
        out_values = []
        branches = []
        for i in range(n_outputs):
            xv = x[i] if x_vector else x
            lv = lo[i] if lo_vector else lo
            hv = hi[i] if hi_vector else hi
            if not lv < hv:
                raise ValueError("clamp requires lower < upper at every position")
            if xv < lv:
                y = lv
                branch = -1
            elif xv > hv:
                y = hv
                branch = 1
            else:
                y = xv
                branch = 0
            if not math.isfinite(y):
                raise ValueError("clamp result must be finite")
            out_values.append(y)
            branches.append(branch)
        out_data = out_values if lengths else out_values[0]
        if not (self.requires_grad or lower.requires_grad or upper.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_lower, parent_upper = self, lower, upper
        # Snapshot every operand and the per-index branch decision so later
        # caller-side mutation or replacement of any input cannot change what
        # a pending backward pass uses; x == lower and x == upper are resolved
        # against these snapshots.
        snap_x = [x[i] if x_vector else x for i in range(n_outputs)]
        snap_lo = [lo[i] if lo_vector else lo for i in range(n_outputs)]
        snap_hi = [hi[i] if hi_vector else hi for i in range(n_outputs)]

        def backward_fn(grad):
            grad_values = grad if isinstance(grad, list) else [grad]
            dx_values = [0.0] * n_outputs
            dl_values = [0.0] * n_outputs
            du_values = [0.0] * n_outputs
            # Inside the range self gets g; outside, the binding bound gets g.
            # On an exact x == lower self and lower split 0.5*g each; on an
            # exact x == upper self and upper split 0.5*g each. Every other
            # role at that index stays 0.0.
            for i in range(n_outputs):
                g = grad_values[i]
                xv = snap_x[i]
                lv = snap_lo[i]
                hv = snap_hi[i]
                branch = branches[i]
                if branch == -1:
                    # Strictly below the range: the lower bound binds and
                    # takes the whole element (x == lower is impossible here,
                    # since that exact tie belongs to the interior branch).
                    if not math.isfinite(g):
                        raise ValueError(
                            "clamp backward intermediate must be finite"
                        )
                    dl_values[i] = g
                elif branch == 1:
                    # Strictly above the range: the upper bound binds.
                    if not math.isfinite(g):
                        raise ValueError(
                            "clamp backward intermediate must be finite"
                        )
                    du_values[i] = g
                else:
                    # Interior branch (lower <= x <= upper). An exact tie with
                    # either bound splits the element 0.5/0.5 between self and
                    # that bound; lower < upper makes both ties impossible at
                    # once. Otherwise self keeps the whole element.
                    if xv == lv:
                        half = 0.5 * g
                        if not math.isfinite(half):
                            raise ValueError(
                                "clamp backward intermediate must be finite"
                            )
                        dx_values[i] = half
                        dl_values[i] = half
                    elif xv == hv:
                        half = 0.5 * g
                        if not math.isfinite(half):
                            raise ValueError(
                                "clamp backward intermediate must be finite"
                            )
                        dx_values[i] = half
                        du_values[i] = half
                    else:
                        if not math.isfinite(g):
                            raise ValueError(
                                "clamp backward intermediate must be finite"
                            )
                        dx_values[i] = g
            roles = []
            if parent_self.requires_grad:
                if x_vector:
                    roles.append((parent_self, dx_values))
                else:
                    # A broadcast scalar parent reduces by accumulating
                    # from 0.0 in ascending output-index order.
                    total = 0.0
                    for value in dx_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "clamp backward intermediate must be finite"
                            )
                    roles.append((parent_self, total))
            if parent_lower.requires_grad:
                if lo_vector:
                    roles.append((parent_lower, dl_values))
                else:
                    total = 0.0
                    for value in dl_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "clamp backward intermediate must be finite"
                            )
                    roles.append((parent_lower, total))
            if parent_upper.requires_grad:
                if hi_vector:
                    roles.append((parent_upper, du_values))
                else:
                    total = 0.0
                    for value in du_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "clamp backward intermediate must be finite"
                            )
                    roles.append((parent_upper, total))
            # The same object may play more than one role (self/lower/upper);
            # merge such roles into one contribution per tensor, submitting
            # only parents that require grad.
            merged = {}
            for parent, value in roles:
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data,
            True,
            (parent_self, parent_lower, parent_upper),
            backward_fn,
        )

    def pow(self, exponent):
        # The exponent is accepted as a Tensor or a finite float only;
        # bools, ints and anything else are a TypeError, and a non-finite
        # float is a ValueError.
        if isinstance(exponent, bool) or not isinstance(
            exponent, (Tensor, float)
        ):
            raise TypeError("exponent must be a Tensor or a finite float")
        if isinstance(exponent, float) and not math.isfinite(exponent):
            raise ValueError("exponent must be finite")
        other = exponent if isinstance(exponent, Tensor) else Tensor(exponent)
        a = _validate_data(self.data)
        b = _validate_data(other.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(other.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        a_vector = isinstance(a, list)
        b_vector = isinstance(b, list)
        if a_vector and b_vector and len(a) != len(b):
            raise ValueError("vector lengths must match")
        # log(a) requires a strictly positive, finite base at every output
        # position, including broadcast scalar bases.
        bases = a if a_vector else [a]
        for base in bases:
            if base <= 0.0:
                raise ValueError("pow base must be positive")
        # Raise base to exponent element by element in ascending
        # output-index order; a non-finite power (overflow included) aborts
        # before a result tensor exists, so no state can change on failure.
        if a_vector and b_vector:
            out_data = [
                _finite_power(a[i], b[i], "pow result must be finite")
                for i in range(len(a))
            ]
        elif a_vector:
            out_data = [
                _finite_power(a[i], b, "pow result must be finite")
                for i in range(len(a))
            ]
        elif b_vector:
            out_data = [
                _finite_power(a, b[i], "pow result must be finite")
                for i in range(len(b))
            ]
        else:
            out_data = _finite_power(a, b, "pow result must be finite")
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        # Snapshot both operands and the forward result so later
        # caller-side mutation or replacement of either input cannot change
        # what a pending backward pass uses.
        snapshot_a = list(a) if a_vector else a
        snapshot_b = list(b) if b_vector else b
        snapshot_y = list(out_data) if isinstance(out_data, list) else out_data
        n_outputs = len(out_data) if (a_vector or b_vector) else 1

        def backward_fn(grad):
            grad_values = grad if isinstance(grad, list) else [grad]
            # Per-output partial contributions, index aligned with the
            # output, before any broadcast reduction.
            da_values = [0.0] * n_outputs
            db_values = [0.0] * n_outputs
            contributions = []
            if parent_self.requires_grad:
                for i in range(n_outputs):
                    base = snapshot_a[i] if a_vector else snapshot_a
                    exp_value = snapshot_b[i] if b_vector else snapshot_b
                    power = _finite_power(
                        base,
                        exp_value - 1.0,
                        "pow backward intermediate must be finite",
                    )
                    product = grad_values[i] * exp_value * power
                    if not math.isfinite(product):
                        raise ValueError(
                            "pow backward intermediate must be finite"
                        )
                    da_values[i] = product
                if a_vector:
                    contributions.append((parent_self, da_values))
                else:
                    # A broadcast scalar parent reduces by accumulating
                    # from 0.0 in ascending output-index order.
                    total = 0.0
                    for value in da_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "pow backward intermediate must be finite"
                            )
                    contributions.append((parent_self, total))
            if parent_other.requires_grad:
                for i in range(n_outputs):
                    base = snapshot_a[i] if a_vector else snapshot_a
                    y_value = snapshot_y[i] if (
                        a_vector or b_vector
                    ) else snapshot_y
                    log_base = math.log(base)
                    if not math.isfinite(log_base):
                        raise ValueError(
                            "pow backward intermediate must be finite"
                        )
                    product = grad_values[i] * y_value * log_base
                    if not math.isfinite(product):
                        raise ValueError(
                            "pow backward intermediate must be finite"
                        )
                    db_values[i] = product
                if b_vector:
                    contributions.append((parent_other, db_values))
                else:
                    total = 0.0
                    for value in db_values:
                        total += value
                        if not math.isfinite(total):
                            raise ValueError(
                                "pow backward intermediate must be finite"
                            )
                    contributions.append((parent_other, total))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def dot(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")
        a = _require_nonempty_float_vector(self, "dot")
        b = _require_nonempty_float_vector(other, "dot")
        if len(a) != len(b):
            raise ValueError("dot vector lengths must match")
        # Accumulate index by index starting from 0.0, multiplying before
        # adding at each index; a non-finite product or partial sum aborts
        # before a result tensor exists, so no state can change on failure.
        acc = 0.0
        for i in range(len(a)):
            product = a[i] * b[i]
            if not math.isfinite(product):
                raise ValueError("dot intermediate must be finite")
            acc += product
            if not math.isfinite(acc):
                raise ValueError("dot intermediate must be finite")
        out_data = acc
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        # Snapshot both operands so later caller-side mutation of either
        # input list cannot change what a pending backward pass uses.
        snapshot_a = list(parent_self.data)
        snapshot_b = list(parent_other.data)

        def backward_fn(grad):
            contributions = []
            if parent_self.requires_grad:
                dx = []
                for value in snapshot_b:
                    term = grad * value
                    if not math.isfinite(term):
                        raise ValueError(
                            "dot backward intermediate must be finite"
                        )
                    dx.append(term)
                contributions.append((parent_self, dx))
            if parent_other.requires_grad:
                db = []
                for value in snapshot_a:
                    term = grad * value
                    if not math.isfinite(term):
                        raise ValueError(
                            "dot backward intermediate must be finite"
                        )
                    db.append(term)
                contributions.append((parent_other, db))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def concat(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")
        a = _require_nonempty_float_vector(self, "concat")
        b = _require_nonempty_float_vector(other, "concat")
        # Copy both operands at call time in call order so later
        # caller-side mutation or replacement of either input can change
        # neither the forward result nor a pending backward pass.
        snapshot_a = list(a)
        snapshot_b = list(b)
        len_a = len(snapshot_a)
        len_b = len(snapshot_b)
        out_data = snapshot_a + snapshot_b
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other

        def backward_fn(grad):
            # Outputs 0..len_a-1 came from self and len_a..len_a+len_b-1
            # from other; split the upstream gradient at that boundary in
            # ascending output-index order. grad was already validated
            # finite, so each slice is finite; only merging the two roles
            # of one shared tensor can produce a non-finite value.
            roles = []
            if parent_self.requires_grad:
                roles.append((parent_self, list(grad[:len_a])))
            if parent_other.requires_grad:
                roles.append((parent_other, list(grad[len_a:])))
            # The same object may play both roles; merge such roles into
            # one contribution per tensor, and submit only parents that
            # require grad.
            merged = {}
            for parent, value in roles:
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def matmul(self, other, size):
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("size must be a positive int")
        if size <= 0:
            raise ValueError("size must be a positive int")
        # Both operands must hold non-empty 1D finite float lists; data may
        # have been mutated after construction, so re-validate at call time.
        a_data = _require_nonempty_float_vector(self, "matmul")
        b_data = _require_nonempty_float_vector(other, "matmul")
        # Each operand stores a size x size matrix in row-major order.
        if len(a_data) != size * size or len(b_data) != size * size:
            raise ValueError("matmul data length must equal size * size")
        # Snapshot both operands so later caller-side mutation of either
        # input list cannot change what a pending backward pass uses.
        snapshot_a = list(a_data)
        snapshot_b = list(b_data)
        # out[r*size+c] = sum_k a[r*size+k] * b[k*size+c], accumulated from
        # 0.0 in ascending (r, c, k) order; a non-finite product or partial
        # sum aborts before a result tensor exists, so no state changes.
        out_data = []
        for r in range(size):
            for c in range(size):
                acc = 0.0
                for k in range(size):
                    product = (
                        snapshot_a[r * size + k] * snapshot_b[k * size + c]
                    )
                    if not math.isfinite(product):
                        raise ValueError(
                            "matmul intermediate must be finite"
                        )
                    acc += product
                    if not math.isfinite(acc):
                        raise ValueError(
                            "matmul intermediate must be finite"
                        )
                out_data.append(acc)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other

        def backward_fn(grad):
            # da[r*size+k] += g[r*size+c] * b[k*size+c] and
            # db[k*size+c] += g[r*size+c] * a[r*size+k], accumulated from
            # 0.0 in ascending (r, c, k) order. A non-finite product or
            # partial sum aborts the whole pass before any grad is written.
            count = size * size
            da = [0.0] * count
            db = [0.0] * count
            for r in range(size):
                for c in range(size):
                    g_rc = grad[r * size + c]
                    for k in range(size):
                        term_a = g_rc * snapshot_b[k * size + c]
                        if not math.isfinite(term_a):
                            raise ValueError(
                                "matmul backward intermediate must be finite"
                            )
                        da[r * size + k] = da[r * size + k] + term_a
                        if not math.isfinite(da[r * size + k]):
                            raise ValueError(
                                "matmul backward intermediate must be finite"
                            )
                        term_b = g_rc * snapshot_a[r * size + k]
                        if not math.isfinite(term_b):
                            raise ValueError(
                                "matmul backward intermediate must be finite"
                            )
                        db[k * size + c] = db[k * size + c] + term_b
                        if not math.isfinite(db[k * size + c]):
                            raise ValueError(
                                "matmul backward intermediate must be finite"
                            )
            # The same object may play both roles; merge such roles into
            # one contribution per tensor, and submit only parents that
            # require grad.
            merged = {}
            roles = (
                (parent_self, da),
                (parent_other, db),
            )
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def matmul_rect(self, other, rows, inner, cols):
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")
        # The three dimensions must be non-bool positive ints.
        for name, value in (
            ("rows", rows),
            ("inner", inner),
            ("cols", cols),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        # Both operands must hold non-empty 1D finite float lists; data may
        # have been mutated after construction, so re-validate at call time.
        a_data = _require_nonempty_float_vector(self, "matmul_rect")
        b_data = _require_nonempty_float_vector(other, "matmul_rect")
        # self stores a rows x inner matrix and other an inner x cols
        # matrix, both in row-major order.
        if len(a_data) != rows * inner:
            raise ValueError("matmul_rect left length must equal rows * inner")
        if len(b_data) != inner * cols:
            raise ValueError(
                "matmul_rect right length must equal inner * cols"
            )
        # Snapshot both operands so later caller-side mutation of either
        # input list cannot change what a pending backward pass uses.
        snapshot_a = list(a_data)
        snapshot_b = list(b_data)
        # out[r*cols+c] = sum_k a[r*inner+k] * b[k*cols+c], accumulated
        # from 0.0 in ascending (r, c, k) order; a non-finite product or
        # partial sum aborts before a result tensor exists, so no state
        # changes.
        out_data = []
        for r in range(rows):
            for c in range(cols):
                acc = 0.0
                for k in range(inner):
                    product = (
                        snapshot_a[r * inner + k]
                        * snapshot_b[k * cols + c]
                    )
                    if not math.isfinite(product):
                        raise ValueError(
                            "matmul_rect intermediate must be finite"
                        )
                    acc += product
                    if not math.isfinite(acc):
                        raise ValueError(
                            "matmul_rect intermediate must be finite"
                        )
                out_data.append(acc)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other

        def backward_fn(grad):
            # da[r*inner+k] += g[r*cols+c] * b[k*cols+c] and
            # db[k*cols+c] += g[r*cols+c] * a[r*inner+k], accumulated from
            # 0.0 in ascending (r, c, k) order. A non-finite product or
            # partial sum aborts the whole pass before any grad is written.
            count_a = rows * inner
            count_b = inner * cols
            da = [0.0] * count_a
            db = [0.0] * count_b
            for r in range(rows):
                for c in range(cols):
                    g_rc = grad[r * cols + c]
                    for k in range(inner):
                        term_a = g_rc * snapshot_b[k * cols + c]
                        if not math.isfinite(term_a):
                            raise ValueError(
                                "matmul_rect backward intermediate"
                                " must be finite"
                            )
                        da[r * inner + k] = da[r * inner + k] + term_a
                        if not math.isfinite(da[r * inner + k]):
                            raise ValueError(
                                "matmul_rect backward intermediate"
                                " must be finite"
                            )
                        term_b = g_rc * snapshot_a[r * inner + k]
                        if not math.isfinite(term_b):
                            raise ValueError(
                                "matmul_rect backward intermediate"
                                " must be finite"
                            )
                        db[k * cols + c] = db[k * cols + c] + term_b
                        if not math.isfinite(db[k * cols + c]):
                            raise ValueError(
                                "matmul_rect backward intermediate"
                                " must be finite"
                            )
            # The same object may play both roles; merge such roles into
            # one contribution per tensor, and submit only parents that
            # require grad.
            merged = {}
            roles = (
                (parent_self, da),
                (parent_other, db),
            )
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def bmm(self, other, batch, rows, inner, cols):
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")
        # The four dimensions must be non-bool positive ints.
        for name, value in (
            ("batch", batch),
            ("rows", rows),
            ("inner", inner),
            ("cols", cols),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        # Both operands must hold non-empty 1D finite float lists; data may
        # have been mutated after construction, so re-validate at call time.
        a_data = _require_nonempty_float_vector(self, "bmm")
        b_data = _require_nonempty_float_vector(other, "bmm")
        B, R, I, C = batch, rows, inner, cols
        # self stores B matrices of rows x inner and other B matrices of
        # inner x cols, all in row-major order with batches stacked.
        if len(a_data) != B * R * I:
            raise ValueError("bmm left length must equal batch * rows * inner")
        if len(b_data) != B * I * C:
            raise ValueError(
                "bmm right length must equal batch * inner * cols"
            )
        # Snapshot both operands so later caller-side mutation of either
        # input list cannot change what a pending backward pass uses.
        snapshot_a = list(a_data)
        snapshot_b = list(b_data)
        # out[q*R*C+r*C+c] = sum_k a[q*R*I+r*I+k] * b[q*I*C+k*C+c],
        # accumulated from 0.0 in ascending (q, r, c, k) order; a non-finite
        # product or partial sum aborts before a result tensor exists, so no
        # state changes.
        out_data = []
        for q in range(B):
            a_base = q * R * I
            b_base = q * I * C
            for r in range(R):
                for c in range(C):
                    acc = 0.0
                    for k in range(I):
                        product = (
                            snapshot_a[a_base + r * I + k]
                            * snapshot_b[b_base + k * C + c]
                        )
                        if not math.isfinite(product):
                            raise ValueError(
                                "bmm intermediate must be finite"
                            )
                        acc += product
                        if not math.isfinite(acc):
                            raise ValueError(
                                "bmm intermediate must be finite"
                            )
                    out_data.append(acc)
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other

        def backward_fn(grad):
            # da[q*R*I+r*I+k] += g[q*R*C+r*C+c] * b[q*I*C+k*C+c] and
            # db[q*I*C+k*C+c] += g[q*R*C+r*C+c] * a[q*R*I+r*I+k],
            # accumulated from 0.0 in ascending (q, r, c, k) order. A
            # non-finite product or partial sum aborts the whole pass before
            # any grad is written.
            da = [0.0] * (B * R * I)
            db = [0.0] * (B * I * C)
            for q in range(B):
                a_base = q * R * I
                b_base = q * I * C
                g_base = q * R * C
                for r in range(R):
                    for c in range(C):
                        g_rc = grad[g_base + r * C + c]
                        for k in range(I):
                            term_a = g_rc * snapshot_b[b_base + k * C + c]
                            if not math.isfinite(term_a):
                                raise ValueError(
                                    "bmm backward intermediate must be finite"
                                )
                            da[a_base + r * I + k] = (
                                da[a_base + r * I + k] + term_a
                            )
                            if not math.isfinite(da[a_base + r * I + k]):
                                raise ValueError(
                                    "bmm backward intermediate must be finite"
                                )
                            term_b = g_rc * snapshot_a[a_base + r * I + k]
                            if not math.isfinite(term_b):
                                raise ValueError(
                                    "bmm backward intermediate must be finite"
                                )
                            db[b_base + k * C + c] = (
                                db[b_base + k * C + c] + term_b
                            )
                            if not math.isfinite(db[b_base + k * C + c]):
                                raise ValueError(
                                    "bmm backward intermediate must be finite"
                                )
            # The same object may play both roles; merge such roles into
            # one contribution per tensor, and submit only parents that
            # require grad.
            merged = {}
            roles = (
                (parent_self, da),
                (parent_other, db),
            )
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def attention(self, key, value, q, k, d, mask=None):
        if not isinstance(key, Tensor):
            raise TypeError("key must be a Tensor")
        if not isinstance(value, Tensor):
            raise TypeError("value must be a Tensor")
        # The three dimensions must be non-bool positive ints.
        for name, dim in (("q", q), ("k", k), ("d", d)):
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise TypeError(name + " must be a positive int")
            if dim <= 0:
                raise ValueError(name + " must be a positive int")
        # All three operands must hold non-empty 1D finite float lists;
        # data may have been mutated after construction, so re-validate at
        # call time. Query holds q rows of d, key and value k rows of d,
        # all in row-major order.
        q_data = _require_nonempty_float_vector(self, "attention")
        k_data = _require_nonempty_float_vector(key, "attention")
        v_data = _require_nonempty_float_vector(value, "attention")
        if len(q_data) != q * d:
            raise ValueError("attention query length must equal q * d")
        if len(k_data) != k * d:
            raise ValueError("attention key length must equal k * d")
        if len(v_data) != k * d:
            raise ValueError("attention value length must equal k * d")
        if mask is not None:
            if not isinstance(mask, list):
                raise TypeError("mask must be a list of bools")
            if len(mask) != q * k:
                raise ValueError("mask length must equal q * k")
            for flag in mask:
                if not isinstance(flag, bool):
                    raise TypeError("mask elements must be bools")
            for i in range(q):
                base = i * k
                if not any(mask[base:base + k]):
                    raise ValueError(
                        "each mask row must contain at least one true"
                    )
        # Snapshot the inputs at call time so later caller-side mutation
        # or replacement of data or mask can change neither the forward
        # result nor a pending backward pass.
        snapshot_q = list(q_data)
        snapshot_k = list(k_data)
        snapshot_v = list(v_data)
        snapshot_mask = None if mask is None else list(mask)
        scale = math.sqrt(d)
        # S = Q K^T / sqrt(d), one dot product per (i, j), accumulated
        # from 0.0 in ascending (i, j, r) order and then scaled; a
        # non-finite product, partial sum or scaled value aborts before a
        # result tensor exists, so no state can change on failure.
        scores = [0.0] * (q * k)
        for i in range(q):
            q_base = i * d
            s_base = i * k
            for j in range(k):
                k_base = j * d
                acc = 0.0
                for r in range(d):
                    product = snapshot_q[q_base + r] * snapshot_k[k_base + r]
                    if not math.isfinite(product):
                        raise ValueError("attention intermediate must be finite")
                    acc += product
                    if not math.isfinite(acc):
                        raise ValueError("attention intermediate must be finite")
                scaled = acc / scale
                if not math.isfinite(scaled):
                    raise ValueError("attention intermediate must be finite")
                scores[s_base + j] = scaled
        # P: per-row stable softmax over the allowed positions only, in
        # row-major and ascending column order: m = max of the row's
        # allowed scores, z_j = exp(s_j - m), denom = sum(z),
        # p_j = z_j / denom; masked positions stay 0.0. exp is only ever
        # evaluated on values in (-inf, 0]; exp(s_j) directly is never
        # computed.
        probs = [0.0] * (q * k)
        for i in range(q):
            base = i * k
            m = None
            for j in range(k):
                if snapshot_mask is not None and not snapshot_mask[base + j]:
                    continue
                score = scores[base + j]
                if m is None or score > m:
                    m = score
            denom = 0.0
            for j in range(k):
                if snapshot_mask is not None and not snapshot_mask[base + j]:
                    continue
                diff = scores[base + j] - m
                if not math.isfinite(diff):
                    raise ValueError("attention intermediate must be finite")
                z_j = math.exp(diff)
                if not math.isfinite(z_j):
                    raise ValueError("attention intermediate must be finite")
                denom += z_j
                if not math.isfinite(denom):
                    raise ValueError("attention intermediate must be finite")
            for j in range(k):
                if snapshot_mask is not None and not snapshot_mask[base + j]:
                    continue
                diff = scores[base + j] - m
                if not math.isfinite(diff):
                    raise ValueError("attention intermediate must be finite")
                z_j = math.exp(diff)
                if not math.isfinite(z_j):
                    raise ValueError("attention intermediate must be finite")
                p_j = z_j / denom
                if not math.isfinite(p_j):
                    raise ValueError("attention intermediate must be finite")
                probs[base + j] = p_j
        # Y = P V, accumulated from 0.0 in ascending (i, r, j) order;
        # masked positions carry P = 0 and contribute nothing.
        out_data = [0.0] * (q * d)
        for i in range(q):
            p_base = i * k
            y_base = i * d
            for r in range(d):
                acc = 0.0
                for j in range(k):
                    if snapshot_mask is not None and not snapshot_mask[
                        p_base + j
                    ]:
                        continue
                    product = probs[p_base + j] * snapshot_v[j * d + r]
                    if not math.isfinite(product):
                        raise ValueError("attention intermediate must be finite")
                    acc += product
                    if not math.isfinite(acc):
                        raise ValueError("attention intermediate must be finite")
                out_data[y_base + r] = acc
        if not (
            self.requires_grad or key.requires_grad or value.requires_grad
        ):
            return Tensor._make(out_data, False, (), None)
        parent_q, parent_k, parent_v = self, key, value
        # Save snapshots and P with the graph so a pending backward pass
        # is independent of any subsequent input or output mutation.
        saved_q = snapshot_q
        saved_k = snapshot_k
        saved_v = snapshot_v
        saved_p = probs

        def backward_fn(grad):
            # H = G V^T, accumulated from 0.0 in ascending (i, j, r)
            # order. Masked positions are excluded from every later
            # contribution; P is 0 there regardless.
            h = [0.0] * (q * k)
            for i in range(q):
                g_base = i * d
                h_base = i * k
                for j in range(k):
                    acc = 0.0
                    v_base = j * d
                    for r in range(d):
                        product = grad[g_base + r] * saved_v[v_base + r]
                        if not math.isfinite(product):
                            raise ValueError(
                                "attention backward intermediate must be finite"
                            )
                        acc += product
                        if not math.isfinite(acc):
                            raise ValueError(
                                "attention backward intermediate must be finite"
                            )
                    h[h_base + j] = acc
            # D = P * (H - row_sum(P * H)), with the row sum accumulated
            # from 0.0 in ascending (i, j) order over the allowed
            # positions, then D filled in ascending (i, j) order.
            dmat = [0.0] * (q * k)
            for i in range(q):
                base = i * k
                row_sum = 0.0
                for j in range(k):
                    if snapshot_mask is not None and not snapshot_mask[
                        base + j
                    ]:
                        continue
                    product = saved_p[base + j] * h[base + j]
                    if not math.isfinite(product):
                        raise ValueError(
                            "attention backward intermediate must be finite"
                        )
                    row_sum += product
                    if not math.isfinite(row_sum):
                        raise ValueError(
                            "attention backward intermediate must be finite"
                        )
                for j in range(k):
                    if snapshot_mask is not None and not snapshot_mask[
                        base + j
                    ]:
                        continue
                    difference = h[base + j] - row_sum
                    if not math.isfinite(difference):
                        raise ValueError(
                            "attention backward intermediate must be finite"
                        )
                    contribution = saved_p[base + j] * difference
                    if not math.isfinite(contribution):
                        raise ValueError(
                            "attention backward intermediate must be finite"
                        )
                    dmat[base + j] = contribution
            # dQ = D K / sqrt(d), accumulated from 0.0 in ascending
            # (i, r, j) order and then scaled.
            dq = [0.0] * (q * d)
            for i in range(q):
                d_base = i * k
                q_base = i * d
                for r in range(d):
                    acc = 0.0
                    for j in range(k):
                        if snapshot_mask is not None and not snapshot_mask[
                            d_base + j
                        ]:
                            continue
                        product = dmat[d_base + j] * saved_k[j * d + r]
                        if not math.isfinite(product):
                            raise ValueError(
                                "attention backward intermediate must be finite"
                            )
                        acc += product
                        if not math.isfinite(acc):
                            raise ValueError(
                                "attention backward intermediate must be finite"
                            )
                    scaled = acc / scale
                    if not math.isfinite(scaled):
                        raise ValueError(
                            "attention backward intermediate must be finite"
                        )
                    dq[q_base + r] = scaled
            # dK = D^T Q / sqrt(d), accumulated from 0.0 in ascending
            # (j, r, i) order and then scaled.
            dk = [0.0] * (k * d)
            for j in range(k):
                k_out_base = j * d
                for r in range(d):
                    acc = 0.0
                    for i in range(q):
                        if snapshot_mask is not None and not snapshot_mask[
                            i * k + j
                        ]:
                            continue
                        product = dmat[i * k + j] * saved_q[i * d + r]
                        if not math.isfinite(product):
                            raise ValueError(
                                "attention backward intermediate must be finite"
                            )
                        acc += product
                        if not math.isfinite(acc):
                            raise ValueError(
                                "attention backward intermediate must be finite"
                            )
                    scaled = acc / scale
                    if not math.isfinite(scaled):
                        raise ValueError(
                            "attention backward intermediate must be finite"
                        )
                    dk[k_out_base + r] = scaled
            # dV = P^T G, accumulated from 0.0 in ascending (j, r, i)
            # order; masked positions carry P = 0 and contribute nothing.
            dv = [0.0] * (k * d)
            for j in range(k):
                v_base = j * d
                for r in range(d):
                    acc = 0.0
                    for i in range(q):
                        if snapshot_mask is not None and not snapshot_mask[
                            i * k + j
                        ]:
                            continue
                        product = saved_p[i * k + j] * grad[i * d + r]
                        if not math.isfinite(product):
                            raise ValueError(
                                "attention backward intermediate must be finite"
                            )
                        acc += product
                        if not math.isfinite(acc):
                            raise ValueError(
                                "attention backward intermediate must be finite"
                            )
                    dv[v_base + r] = acc
            # The same object may play several of the query/key/value
            # roles; merge such roles into one contribution per tensor,
            # and submit only parents that require grad.
            merged = {}
            roles = (
                (parent_q, dq),
                (parent_k, dk),
                (parent_v, dv),
            )
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_q, parent_k, parent_v), backward_fn
        )

    def linear(self, weight, bias):
        if not isinstance(weight, Tensor):
            raise TypeError("weight must be a Tensor")
        if not isinstance(bias, Tensor):
            raise TypeError("bias must be a Tensor")
        # All three operands must hold non-empty 1D finite float lists;
        # data may have been mutated after construction, so re-validate at
        # call time. bias doubles as the output descriptor: m = len(bias).
        x_data = _require_nonempty_float_vector(self, "linear")
        w_data = _require_nonempty_float_vector(weight, "linear")
        b_data = _require_nonempty_float_vector(bias, "linear")
        n = len(x_data)
        m = len(b_data)
        # weight stores the m x n matrix in row-major order.
        if len(w_data) != m * n:
            raise ValueError("linear weight length must equal m * n")
        # Snapshot all three operands so later caller-side mutation or
        # replacement of any input can change neither the forward result
        # nor a pending backward pass.
        snapshot_x = list(x_data)
        snapshot_w = list(w_data)
        snapshot_b = list(b_data)
        # out[o] = b[o] + sum_i x[i] * w[o*n+i], accumulated starting from
        # b[o] in ascending (o, i) order; a non-finite product or partial
        # sum aborts before a result tensor exists, so no state changes.
        out_data = []
        for o in range(m):
            acc = snapshot_b[o]
            for i in range(n):
                product = snapshot_x[i] * snapshot_w[o * n + i]
                if not math.isfinite(product):
                    raise ValueError("linear intermediate must be finite")
                acc += product
                if not math.isfinite(acc):
                    raise ValueError("linear intermediate must be finite")
            out_data.append(acc)
        if not (
            self.requires_grad
            or weight.requires_grad
            or bias.requires_grad
        ):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_weight, parent_bias = self, weight, bias

        def backward_fn(grad):
            # dx[i] += g[o]*w[o*n+i], dw[o*n+i] = g[o]*x[i], db[o] = g[o],
            # evaluated in ascending (o, i) order. A non-finite product or
            # partial sum aborts the whole pass before any grad is written.
            dx = [0.0] * n
            dw = [0.0] * (m * n)
            db = [0.0] * m
            for o in range(m):
                go = grad[o]
                db[o] = go
                for i in range(n):
                    term_x = go * snapshot_w[o * n + i]
                    if not math.isfinite(term_x):
                        raise ValueError(
                            "linear backward intermediate must be finite"
                        )
                    dx[i] = dx[i] + term_x
                    if not math.isfinite(dx[i]):
                        raise ValueError(
                            "linear backward intermediate must be finite"
                        )
                    term_w = go * snapshot_x[i]
                    if not math.isfinite(term_w):
                        raise ValueError(
                            "linear backward intermediate must be finite"
                        )
                    dw[o * n + i] = term_w
            # One object may play several roles (e.g. weight is also bias);
            # merge such roles into one contribution per tensor, and submit
            # only parents that require grad. Role shapes agree whenever
            # the objects coincide, since len(w) = m*n then equals n or m.
            merged = {}
            roles = (
                (parent_self, dx),
                (parent_weight, dw),
                (parent_bias, db),
            )
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data,
            True,
            (parent_self, parent_weight, parent_bias),
            backward_fn,
        )

    def linear_batch(self, weight, bias, batch_size):
        if not isinstance(weight, Tensor):
            raise TypeError("weight must be a Tensor")
        if not isinstance(bias, Tensor):
            raise TypeError("bias must be a Tensor")
        # All three operands must hold non-empty 1D finite float lists;
        # data may have been mutated after construction, so re-validate at
        # call time. x is a B x n row-major batch, w the m x n row-major
        # matrix and b the length-m bias, so m = len(bias).
        x_data = _require_nonempty_float_vector(self, "linear_batch")
        w_data = _require_nonempty_float_vector(weight, "linear_batch")
        b_data = _require_nonempty_float_vector(bias, "linear_batch")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be a positive int")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive int")
        if len(x_data) % batch_size != 0:
            raise ValueError("batch_size must divide the input length")
        B = batch_size
        n = len(x_data) // B
        m = len(b_data)
        # weight stores the m x n matrix in row-major order.
        if len(w_data) != m * n:
            raise ValueError("linear_batch weight length must equal m * n")
        # Snapshot all three operands so later caller-side mutation or
        # replacement of any input can change neither the forward result
        # nor a pending backward pass.
        snapshot_x = list(x_data)
        snapshot_w = list(w_data)
        snapshot_b = list(b_data)
        # out[q*m+o] = b[o] + sum_i x[q*n+i] * w[o*n+i], accumulated
        # starting from b[o] in ascending (q, o, i) order; a non-finite
        # product or partial sum aborts before a result tensor exists, so
        # no state changes.
        out_data = []
        for q in range(B):
            for o in range(m):
                acc = snapshot_b[o]
                for i in range(n):
                    product = snapshot_x[q * n + i] * snapshot_w[o * n + i]
                    if not math.isfinite(product):
                        raise ValueError(
                            "linear_batch intermediate must be finite"
                        )
                    acc += product
                    if not math.isfinite(acc):
                        raise ValueError(
                            "linear_batch intermediate must be finite"
                        )
                out_data.append(acc)
        if not (
            self.requires_grad
            or weight.requires_grad
            or bias.requires_grad
        ):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_weight, parent_bias = self, weight, bias

        def backward_fn(grad):
            # h = grad[q*m+o]; db[o] += h, dx[q*n+i] += h*w[o*n+i] and
            # dw[o*n+i] += h*x[q*n+i], accumulated from 0.0 in ascending
            # (q, o, i) order. A non-finite product or partial sum aborts
            # the whole pass before any grad is written.
            dx = [0.0] * (B * n)
            dw = [0.0] * (m * n)
            db = [0.0] * m
            for q in range(B):
                for o in range(m):
                    h = grad[q * m + o]
                    db[o] = db[o] + h
                    if not math.isfinite(db[o]):
                        raise ValueError(
                            "linear_batch backward intermediate must be"
                            " finite"
                        )
                    for i in range(n):
                        j = q * n + i
                        t = o * n + i
                        term_x = h * snapshot_w[t]
                        if not math.isfinite(term_x):
                            raise ValueError(
                                "linear_batch backward intermediate must be"
                                " finite"
                            )
                        dx[j] = dx[j] + term_x
                        if not math.isfinite(dx[j]):
                            raise ValueError(
                                "linear_batch backward intermediate must be"
                                " finite"
                            )
                        term_w = h * snapshot_x[j]
                        if not math.isfinite(term_w):
                            raise ValueError(
                                "linear_batch backward intermediate must be"
                                " finite"
                            )
                        dw[t] = dw[t] + term_w
                        if not math.isfinite(dw[t]):
                            raise ValueError(
                                "linear_batch backward intermediate must be"
                                " finite"
                            )
            # One object may play several roles (e.g. weight is also
            # bias); merge such roles into one contribution per tensor,
            # and submit only parents that require grad.
            merged = {}
            roles = (
                (parent_self, dx),
                (parent_weight, dw),
                (parent_bias, db),
            )
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data,
            True,
            (parent_self, parent_weight, parent_bias),
            backward_fn,
        )

    def mse_loss(self, target):
        if not isinstance(target, Tensor):
            raise TypeError("target must be a Tensor")
        a = _require_nonempty_float_vector(self, "mse_loss")
        b = _require_nonempty_float_vector(target, "mse_loss")
        if len(a) != len(b):
            raise ValueError("mse_loss vector lengths must match")
        # Copy both operands at call time so later caller-side mutation or
        # replacement of either input can change neither the forward result
        # nor a pending backward pass.
        x = list(a)
        y = list(b)
        n = len(x)
        # Accumulate the squared residual index by index from 0.0 in
        # ascending order, subtracting, squaring and adding at each index;
        # a non-finite residual, square, partial sum or quotient aborts
        # before a result tensor exists, so no state can change on failure.
        residuals = []
        q = 0.0
        for i in range(n):
            r_i = x[i] - y[i]
            if not math.isfinite(r_i):
                raise ValueError("mse_loss intermediate must be finite")
            residuals.append(r_i)
            square = r_i * r_i
            if not math.isfinite(square):
                raise ValueError("mse_loss intermediate must be finite")
            q += square
            if not math.isfinite(q):
                raise ValueError("mse_loss intermediate must be finite")
        out_data = q / n
        if not math.isfinite(out_data):
            raise ValueError("mse_loss intermediate must be finite")
        if not (self.requires_grad or target.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, target
        # Save x, y and the residuals with the graph so a pending backward
        # pass is independent of any subsequent data mutation.
        snapshot_x = x
        snapshot_y = y
        snapshot_r = residuals

        def backward_fn(grad):
            # h_i = g * 2 * r_i / n, multiplying and dividing in that
            # order per index; a non-finite intermediate aborts the whole
            # pass before any grad is written.
            h = []
            for r_i in snapshot_r:
                scaled = grad * 2.0
                if not math.isfinite(scaled):
                    raise ValueError(
                        "mse_loss backward intermediate must be finite"
                    )
                product = scaled * r_i
                if not math.isfinite(product):
                    raise ValueError(
                        "mse_loss backward intermediate must be finite"
                    )
                value = product / n
                if not math.isfinite(value):
                    raise ValueError(
                        "mse_loss backward intermediate must be finite"
                    )
                h.append(value)
            contributions = []
            if parent_self.requires_grad:
                contributions.append((parent_self, h))
            if parent_other.requires_grad:
                negated = []
                for value in h:
                    neg = -value
                    if not math.isfinite(neg):
                        raise ValueError(
                            "mse_loss backward intermediate must be finite"
                        )
                    negated.append(neg)
                contributions.append((parent_other, negated))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def huber_loss(self, target, delta=1.0):
        if not isinstance(target, Tensor):
            raise TypeError("target must be a Tensor")
        a = _require_nonempty_float_vector(self, "huber_loss")
        b = _require_nonempty_float_vector(target, "huber_loss")
        if len(a) != len(b):
            raise ValueError("huber_loss vector lengths must match")
        if isinstance(delta, bool) or not isinstance(delta, float):
            raise TypeError("delta must be a positive finite float")
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be a positive finite float")
        # Copy both operands at call time so later caller-side mutation or
        # replacement of either input can change neither the forward result
        # nor a pending backward pass.
        x = list(a)
        y = list(b)
        n = len(x)
        # Accumulate the per-index Huber term from 0.0 in ascending index
        # order: r_i = x_i - y_i, a_i = |r_i|; the term is 0.5*r_i*r_i when
        # a_i <= delta and delta*(a_i - 0.5*delta) otherwise; the total is
        # divided by n at the end. A non-finite intermediate aborts before
        # a result tensor exists, so no state can change on failure.
        residuals = []
        q = 0.0
        for i in range(n):
            r_i = x[i] - y[i]
            if not math.isfinite(r_i):
                raise ValueError("huber_loss intermediate must be finite")
            residuals.append(r_i)
            a_i = abs(r_i)
            if not math.isfinite(a_i):
                raise ValueError("huber_loss intermediate must be finite")
            if a_i <= delta:
                square = r_i * r_i
                if not math.isfinite(square):
                    raise ValueError(
                        "huber_loss intermediate must be finite"
                    )
                term = 0.5 * square
            else:
                half_delta = 0.5 * delta
                if not math.isfinite(half_delta):
                    raise ValueError(
                        "huber_loss intermediate must be finite"
                    )
                shifted = a_i - half_delta
                if not math.isfinite(shifted):
                    raise ValueError(
                        "huber_loss intermediate must be finite"
                    )
                term = delta * shifted
            if not math.isfinite(term):
                raise ValueError("huber_loss intermediate must be finite")
            q += term
            if not math.isfinite(q):
                raise ValueError("huber_loss intermediate must be finite")
        out_data = q / n
        if not math.isfinite(out_data):
            raise ValueError("huber_loss intermediate must be finite")
        if not (self.requires_grad or target.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, target
        # Save the residuals with the graph so a pending backward pass is
        # independent of any subsequent data mutation.
        snapshot_r = residuals

        def backward_fn(grad):
            # d_i = r_i when |r_i| <= delta, else +/-delta by the sign of
            # r_i; h_i = g * d_i / n, multiplying and dividing in that
            # order per index. A non-finite intermediate aborts the whole
            # pass before any grad is written.
            h = []
            for r_i in snapshot_r:
                if abs(r_i) <= delta:
                    d_i = r_i
                elif r_i > 0.0:
                    d_i = delta
                else:
                    d_i = -delta
                if not math.isfinite(d_i):
                    raise ValueError(
                        "huber_loss backward intermediate must be finite"
                    )
                product = grad * d_i
                if not math.isfinite(product):
                    raise ValueError(
                        "huber_loss backward intermediate must be finite"
                    )
                value = product / n
                if not math.isfinite(value):
                    raise ValueError(
                        "huber_loss backward intermediate must be finite"
                    )
                h.append(value)
            contributions = []
            if parent_self.requires_grad:
                contributions.append((parent_self, h))
            if parent_other.requires_grad:
                negated = []
                for value in h:
                    neg = -value
                    if not math.isfinite(neg):
                        raise ValueError(
                            "huber_loss backward intermediate must be finite"
                        )
                    negated.append(neg)
                contributions.append((parent_other, negated))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def binary_cross_entropy_with_logits(self, target):
        if not isinstance(target, Tensor):
            raise TypeError("target must be a Tensor")
        a = _require_nonempty_float_vector(
            self, "binary_cross_entropy_with_logits"
        )
        b = _require_nonempty_float_vector(
            target, "binary_cross_entropy_with_logits"
        )
        if len(a) != len(b):
            raise ValueError(
                "binary_cross_entropy_with_logits vector lengths must match"
            )
        for y_i in b:
            if y_i < 0.0 or y_i > 1.0:
                raise ValueError(
                    "binary_cross_entropy_with_logits target values must lie"
                    " in [0.0, 1.0]"
                )
        # Copy both operands at call time so later caller-side mutation or
        # replacement of either input can change neither the forward result
        # nor a pending backward pass.
        x = list(a)
        y = list(b)
        n = len(x)
        # Accumulate the per-index BCE-with-logits term from 0.0 in ascending
        # index order: max(x_i, 0) - x_i*y_i + log1p(exp(-|x_i|)); the total
        # is divided by n at the end. A non-finite intermediate aborts before
        # a result tensor exists, so no state can change on failure.
        total = 0.0
        for i in range(n):
            x_i = x[i]
            y_i = y[i]
            relu = max(x_i, 0.0)
            if not math.isfinite(relu):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
            product = x_i * y_i
            if not math.isfinite(product):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
            difference = relu - product
            if not math.isfinite(difference):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
            abs_x = abs(x_i)
            if not math.isfinite(abs_x):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
            neg_abs = -abs_x
            if not math.isfinite(neg_abs):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
            exp_term = math.exp(neg_abs)
            if not math.isfinite(exp_term):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
            log_term = math.log1p(exp_term)
            if not math.isfinite(log_term):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
            term = difference + log_term
            if not math.isfinite(term):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
            total += term
            if not math.isfinite(total):
                raise ValueError(
                    "binary_cross_entropy_with_logits intermediate must be"
                    " finite"
                )
        out_data = total / n
        if not math.isfinite(out_data):
            raise ValueError(
                "binary_cross_entropy_with_logits intermediate must be finite"
            )
        if not (self.requires_grad or target.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, target
        # Save x and y with the graph so a pending backward pass is
        # independent of any subsequent data mutation.
        snapshot_x = x
        snapshot_y = y

        def backward_fn(grad):
            # s_i is the sigmoid of x_i, evaluated via the non-overflowing
            # branch for each sign; dx_i = g*(s_i - y_i)/n and
            # dy_i = -g*x_i/n. Only the sides that require grad are computed
            # and submitted; a non-finite intermediate aborts the whole pass
            # before any grad is written.
            contributions = []
            if parent_self.requires_grad:
                dx = []
                for i in range(n):
                    x_i = snapshot_x[i]
                    y_i = snapshot_y[i]
                    if x_i >= 0.0:
                        exp_neg = math.exp(-x_i)
                        if not math.isfinite(exp_neg):
                            raise ValueError(
                                "binary_cross_entropy_with_logits backward"
                                " intermediate must be finite"
                            )
                        denominator = 1.0 + exp_neg
                        if not math.isfinite(denominator):
                            raise ValueError(
                                "binary_cross_entropy_with_logits backward"
                                " intermediate must be finite"
                            )
                        s_i = 1.0 / denominator
                    else:
                        e = math.exp(x_i)
                        if not math.isfinite(e):
                            raise ValueError(
                                "binary_cross_entropy_with_logits backward"
                                " intermediate must be finite"
                            )
                        denominator = 1.0 + e
                        if not math.isfinite(denominator):
                            raise ValueError(
                                "binary_cross_entropy_with_logits backward"
                                " intermediate must be finite"
                            )
                        s_i = e / denominator
                    if not math.isfinite(s_i):
                        raise ValueError(
                            "binary_cross_entropy_with_logits backward"
                            " intermediate must be finite"
                        )
                    delta = s_i - y_i
                    if not math.isfinite(delta):
                        raise ValueError(
                            "binary_cross_entropy_with_logits backward"
                            " intermediate must be finite"
                        )
                    scaled_x = grad * delta
                    if not math.isfinite(scaled_x):
                        raise ValueError(
                            "binary_cross_entropy_with_logits backward"
                            " intermediate must be finite"
                        )
                    dx_i = scaled_x / n
                    if not math.isfinite(dx_i):
                        raise ValueError(
                            "binary_cross_entropy_with_logits backward"
                            " intermediate must be finite"
                        )
                    dx.append(dx_i)
                contributions.append((parent_self, dx))
            if parent_other.requires_grad:
                dy = []
                for i in range(n):
                    x_i = snapshot_x[i]
                    scaled_y = grad * x_i
                    if not math.isfinite(scaled_y):
                        raise ValueError(
                            "binary_cross_entropy_with_logits backward"
                            " intermediate must be finite"
                        )
                    dy_i = -scaled_y / n
                    if not math.isfinite(dy_i):
                        raise ValueError(
                            "binary_cross_entropy_with_logits backward"
                            " intermediate must be finite"
                        )
                    dy.append(dy_i)
                contributions.append((parent_other, dy))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def cosine_similarity(self, other, eps=1e-12):
        if not isinstance(other, Tensor):
            raise TypeError("other must be a Tensor")
        a = _require_nonempty_float_vector(self, "cosine_similarity")
        b = _require_nonempty_float_vector(other, "cosine_similarity")
        if len(a) != len(b):
            raise ValueError("cosine_similarity vector lengths must match")
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        # Accumulate the dot product and both squared norms index by index
        # from 0.0, multiplying before adding at each index; a non-finite
        # product or partial sum aborts before a result tensor exists, so no
        # state can change on failure.
        d = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(len(a)):
            product = a[i] * b[i]
            if not math.isfinite(product):
                raise ValueError(
                    "cosine_similarity intermediate must be finite"
                )
            d += product
            if not math.isfinite(d):
                raise ValueError(
                    "cosine_similarity intermediate must be finite"
                )
            square_a = a[i] * a[i]
            if not math.isfinite(square_a):
                raise ValueError(
                    "cosine_similarity intermediate must be finite"
                )
            norm_a += square_a
            if not math.isfinite(norm_a):
                raise ValueError(
                    "cosine_similarity intermediate must be finite"
                )
            square_b = b[i] * b[i]
            if not math.isfinite(square_b):
                raise ValueError(
                    "cosine_similarity intermediate must be finite"
                )
            norm_b += square_b
            if not math.isfinite(norm_b):
                raise ValueError(
                    "cosine_similarity intermediate must be finite"
                )
        denom_a = norm_a + eps
        if not math.isfinite(denom_a):
            raise ValueError("cosine_similarity intermediate must be finite")
        u = math.sqrt(denom_a)
        if not math.isfinite(u):
            raise ValueError("cosine_similarity intermediate must be finite")
        denom_b = norm_b + eps
        if not math.isfinite(denom_b):
            raise ValueError("cosine_similarity intermediate must be finite")
        v = math.sqrt(denom_b)
        if not math.isfinite(v):
            raise ValueError("cosine_similarity intermediate must be finite")
        q = u * v
        if not math.isfinite(q):
            raise ValueError("cosine_similarity intermediate must be finite")
        out_data = d / q
        if not math.isfinite(out_data):
            raise ValueError("cosine_similarity intermediate must be finite")
        if not (self.requires_grad or other.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, other
        # Snapshot both operands and the forward scalars so later
        # caller-side mutation of either input cannot change what a pending
        # backward pass uses.
        snapshot_a = list(parent_self.data)
        snapshot_b = list(parent_other.data)
        A = norm_a
        B = norm_b
        y = out_data

        def backward_fn(grad):
            contributions = []
            if parent_self.requires_grad:
                da = []
                total_a = A + eps
                if not math.isfinite(total_a):
                    raise ValueError(
                        "cosine_similarity backward intermediate must be"
                        " finite"
                    )
                for i in range(len(snapshot_a)):
                    over_q = snapshot_b[i] / q
                    if not math.isfinite(over_q):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    scaled = y * snapshot_a[i]
                    if not math.isfinite(scaled):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    normalized = scaled / total_a
                    if not math.isfinite(normalized):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    diff = over_q - normalized
                    if not math.isfinite(diff):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    value = grad * diff
                    if not math.isfinite(value):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    da.append(value)
                contributions.append((parent_self, da))
            if parent_other.requires_grad:
                db = []
                total_b = B + eps
                if not math.isfinite(total_b):
                    raise ValueError(
                        "cosine_similarity backward intermediate must be"
                        " finite"
                    )
                for i in range(len(snapshot_b)):
                    over_q = snapshot_a[i] / q
                    if not math.isfinite(over_q):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    scaled = y * snapshot_b[i]
                    if not math.isfinite(scaled):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    normalized = scaled / total_b
                    if not math.isfinite(normalized):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    diff = over_q - normalized
                    if not math.isfinite(diff):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    value = grad * diff
                    if not math.isfinite(value):
                        raise ValueError(
                            "cosine_similarity backward intermediate must be"
                            " finite"
                        )
                    db.append(value)
                contributions.append((parent_other, db))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def sigmoid(self):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints and other types are a TypeError; an
        # empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        # Elementwise stable evaluation in ascending index order via
        # _stable_sigmoid_value: exp is only ever evaluated on values in
        # (-inf, 0], never on a large positive argument. A non-finite e, d
        # or y aborts before a result tensor exists, so no state can
        # change on failure.
        if isinstance(data, list):
            out_data = [_stable_sigmoid_value(x) for x in data]
        else:
            out_data = _stable_sigmoid_value(data)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Snapshot the forward output so later caller-side mutation or
        # replacement of the input cannot change a pending backward pass.
        snapshot_y = list(out_data) if isinstance(out_data, list) else out_data

        def backward_fn(grad):
            # Elementwise g*y*(1.0-y); each intermediate is checked as it
            # is produced, and the generic engine validates the
            # contribution itself and every merge into an existing grad.
            if isinstance(grad, list):
                contribution = []
                for i in range(len(snapshot_y)):
                    one_minus = 1.0 - snapshot_y[i]
                    if not math.isfinite(one_minus):
                        raise ValueError(
                            "sigmoid backward intermediate must be finite"
                        )
                    product = grad[i] * snapshot_y[i]
                    if not math.isfinite(product):
                        raise ValueError(
                            "sigmoid backward intermediate must be finite"
                        )
                    value = product * one_minus
                    if not math.isfinite(value):
                        raise ValueError(
                            "sigmoid backward intermediate must be finite"
                        )
                    contribution.append(value)
            else:
                one_minus = 1.0 - snapshot_y
                if not math.isfinite(one_minus):
                    raise ValueError(
                        "sigmoid backward intermediate must be finite"
                    )
                product = grad * snapshot_y
                if not math.isfinite(product):
                    raise ValueError(
                        "sigmoid backward intermediate must be finite"
                    )
                value = product * one_minus
                if not math.isfinite(value):
                    raise ValueError(
                        "sigmoid backward intermediate must be finite"
                    )
                contribution = value
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def softplus(self, beta=1.0, threshold=20.0):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints and other types are a TypeError; an
        # empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        # beta and threshold are plain floats: bools, ints and anything
        # else are a TypeError; non-finite values are a ValueError, and a
        # non-positive beta is additionally a ValueError.
        if isinstance(beta, bool) or not isinstance(beta, float):
            raise TypeError("beta must be a positive finite float")
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be a positive finite float")
        if isinstance(threshold, bool) or not isinstance(threshold, float):
            raise TypeError("threshold must be a finite float")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")

        def forward_value(x):
            # z = beta*x; when z > threshold the linear branch y = x is
            # used, otherwise y = log1p(exp(z))/beta. Every intermediate
            # is checked as it is produced.
            z = beta * x
            if not math.isfinite(z):
                raise ValueError("softplus intermediate must be finite")
            if z > threshold:
                y = x
                if not math.isfinite(y):
                    raise ValueError("softplus intermediate must be finite")
                return y, z, True
            exp_z = _finite_exp(z, "softplus intermediate must be finite")
            log_term = math.log1p(exp_z)
            if not math.isfinite(log_term):
                raise ValueError("softplus intermediate must be finite")
            y = log_term / beta
            if not math.isfinite(y):
                raise ValueError("softplus intermediate must be finite")
            return y, z, False

        # Elementwise evaluation in ascending index order; a non-finite
        # intermediate aborts before a result tensor exists, so no state
        # can change on failure.
        if isinstance(data, list):
            out_data = []
            snapshot_z = []
            snapshot_linear = []
            for x in data:
                y, z, linear = forward_value(x)
                out_data.append(y)
                snapshot_z.append(z)
                snapshot_linear.append(linear)
        else:
            out_data, z_value, linear_value = forward_value(data)
            snapshot_z = z_value
            snapshot_linear = linear_value
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Snapshot z and the per-element branch choice so later
        # caller-side mutation or replacement of the input cannot change a
        # pending backward pass. beta itself never needs to enter the
        # backward formula: d/dx softplus(x; beta) = sigmoid(beta*x).
        z_values = snapshot_z
        linear_flags = snapshot_linear
        vector = isinstance(out_data, list)

        def backward_value(g, z, linear):
            if linear:
                h = g
                if not math.isfinite(h):
                    raise ValueError(
                        "softplus backward intermediate must be finite"
                    )
                return h
            if z >= 0.0:
                q = _finite_exp(
                    -z, "softplus backward intermediate must be finite"
                )
                d = 1.0 + q
                if not math.isfinite(d):
                    raise ValueError(
                        "softplus backward intermediate must be finite"
                    )
                h = g / d
            else:
                q = _finite_exp(
                    z, "softplus backward intermediate must be finite"
                )
                d = 1.0 + q
                if not math.isfinite(d):
                    raise ValueError(
                        "softplus backward intermediate must be finite"
                    )
                product = g * q
                if not math.isfinite(product):
                    raise ValueError(
                        "softplus backward intermediate must be finite"
                    )
                h = product / d
            if not math.isfinite(h):
                raise ValueError(
                    "softplus backward intermediate must be finite"
                )
            return h

        def backward_fn(grad):
            # The linear branch contributes g; the smooth branch uses the
            # numerically appropriate reciprocal sigmoid factor: g/(1+e^-z)
            # for z >= 0 and g*e^z/(1+e^z) for z < 0. Each intermediate is
            # checked as it is produced, and the generic engine validates
            # the contribution itself and every merge into an existing grad.
            if vector:
                contribution = [
                    backward_value(grad[i], z_values[i], linear_flags[i])
                    for i in range(len(z_values))
                ]
            else:
                contribution = backward_value(
                    grad, z_values, linear_flags
                )
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def leaky_relu(self, negative_slope=0.01):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints and other types are a TypeError; an
        # empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        # negative_slope is a plain non-negative float: bools, ints and
        # anything else are a TypeError; a non-finite or negative value is
        # a ValueError.
        if isinstance(negative_slope, bool) or not isinstance(
            negative_slope, float
        ):
            raise TypeError("negative_slope must be a non-negative finite float")
        if not math.isfinite(negative_slope) or negative_slope < 0.0:
            raise ValueError(
                "negative_slope must be a non-negative finite float"
            )

        def forward_value(x):
            # y = x when x >= 0, else y = negative_slope * x; the result
            # is checked as it is produced.
            if x >= 0.0:
                y = x
            else:
                y = negative_slope * x
            if not math.isfinite(y):
                raise ValueError("leaky_relu result must be finite")
            return y

        # Elementwise evaluation in ascending index order; a non-finite
        # result aborts before a result tensor exists, so no state can
        # change on failure.
        out_data = _map_unary(data, forward_value)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Snapshot the input so later caller-side mutation or replacement
        # of the input cannot change which branch a pending backward pass
        # applies. x >= 0 (x == 0 included) has derivative 1.0; x < 0 has
        # derivative negative_slope.
        snapshot_x = list(data) if isinstance(data, list) else data

        def backward_fn(grad):
            # Elementwise g for x >= 0 and g * negative_slope for x < 0,
            # in ascending index order; each intermediate is checked as it
            # is produced, and the generic engine validates the
            # contribution itself and every merge into an existing grad.
            if isinstance(grad, list):
                contribution = []
                for i in range(len(snapshot_x)):
                    if snapshot_x[i] >= 0.0:
                        value = grad[i]
                    else:
                        value = grad[i] * negative_slope
                    if not math.isfinite(value):
                        raise ValueError(
                            "leaky_relu backward intermediate must be finite"
                        )
                    contribution.append(value)
            else:
                if snapshot_x >= 0.0:
                    value = grad
                else:
                    value = grad * negative_slope
                if not math.isfinite(value):
                    raise ValueError(
                        "leaky_relu backward intermediate must be finite"
                    )
                contribution = value
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def relu(self):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints, nested lists and other types are a
        # TypeError; an empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")

        # Elementwise y = x for x >= 0.0 and 0.0 for x < 0.0, evaluated in
        # ascending output-index order. A non-finite result aborts before a
        # result tensor exists, so the input and every other state stays
        # untouched on failure.
        if isinstance(data, list):
            out_data = []
            for x in data:
                y = x if x >= 0.0 else 0.0
                if not math.isfinite(y):
                    raise ValueError("relu result must be finite")
                out_data.append(y)
        else:
            out_data = data if data >= 0.0 else 0.0
            if not math.isfinite(out_data):
                raise ValueError("relu result must be finite")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Snapshot the shape and the forward branch decision so later
        # caller-side mutation or replacement of the input cannot change
        # which branch a pending backward pass applies. x >= 0.0 (x == 0.0
        # included) passes the gradient through; x < 0.0 blocks it.
        if isinstance(data, list):
            snapshot_shape = len(data)
            snapshot_branch = [x >= 0.0 for x in data]
        else:
            snapshot_shape = ()
            snapshot_branch = data >= 0.0

        def backward_fn(grad):
            # Elementwise dx = g on the x >= 0.0 branch and 0.0 on the
            # x < 0.0 branch, in ascending index order; each intermediate
            # is checked as it is produced, and the generic engine validates
            # the contribution itself and every merge into an existing grad.
            if isinstance(grad, list):
                contribution = []
                for i in range(snapshot_shape):
                    value = grad[i] if snapshot_branch[i] else 0.0
                    if not math.isfinite(value):
                        raise ValueError(
                            "relu backward intermediate must be finite"
                        )
                    contribution.append(value)
            else:
                value = grad if snapshot_branch else 0.0
                if not math.isfinite(value):
                    raise ValueError(
                        "relu backward intermediate must be finite"
                    )
                contribution = value
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def tanh(self):
        out_data = _map_unary(self.data, math.tanh)
        _ensure_finite_data(out_data)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        out = out_data

        def backward_fn(grad):
            contribution = _broadcast_apply(
                grad, out, lambda g, y: g * (1.0 - y * y)
            )
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def exp(self):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints and other types are a TypeError; an
        # empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        # Elementwise math.exp in ascending index order; an overflow or
        # non-finite result aborts before a result tensor exists, so no
        # state can change on failure.
        out_data = _map_unary(data, _finite_exp)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Snapshot the forward output so later caller-side mutation or
        # replacement of the input cannot change a pending backward pass.
        snapshot_y = (
            list(out_data) if isinstance(out_data, list) else out_data
        )

        def backward_fn(grad):
            # Elementwise g*y in ascending index order; each intermediate
            # is checked as it is produced, and the generic engine
            # validates the contribution itself and every merge into an
            # existing grad.
            if isinstance(grad, list):
                contribution = []
                for i in range(len(snapshot_y)):
                    value = grad[i] * snapshot_y[i]
                    if not math.isfinite(value):
                        raise ValueError(
                            "exp backward intermediate must be finite"
                        )
                    contribution.append(value)
            else:
                value = grad * snapshot_y
                if not math.isfinite(value):
                    raise ValueError(
                        "exp backward intermediate must be finite"
                    )
                contribution = value
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def log(self):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints and other types are a TypeError; an
        # empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        # Elementwise math.log in ascending index order; every input must
        # be strictly positive, and a non-finite result aborts before a
        # result tensor exists, so no state can change on failure.
        if isinstance(data, list):
            out_data = []
            for x in data:
                if x <= 0.0:
                    raise ValueError("log input must be positive")
                y = math.log(x)
                if not math.isfinite(y):
                    raise ValueError("log result must be finite")
                out_data.append(y)
        else:
            if data <= 0.0:
                raise ValueError("log input must be positive")
            out_data = math.log(data)
            if not math.isfinite(out_data):
                raise ValueError("log result must be finite")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Snapshot the forward input so later caller-side mutation or
        # replacement of the input cannot change a pending backward pass.
        snapshot_x = list(data) if isinstance(data, list) else data

        def backward_fn(grad):
            # Elementwise g/x in ascending index order; each intermediate
            # is checked as it is produced, and the generic engine
            # validates the contribution itself and every merge into an
            # existing grad.
            if isinstance(grad, list):
                contribution = []
                for i in range(len(snapshot_x)):
                    value = grad[i] / snapshot_x[i]
                    if not math.isfinite(value):
                        raise ValueError(
                            "log backward intermediate must be finite"
                        )
                    contribution.append(value)
            else:
                value = grad / snapshot_x
                if not math.isfinite(value):
                    raise ValueError(
                        "log backward intermediate must be finite"
                    )
                contribution = value
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def sqrt(self):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints, nested lists and other types are a
        # TypeError; an empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        # Elementwise math.sqrt in ascending index order; every input must
        # be non-negative (x == 0.0 is allowed, y == 0.0 is finite), and a
        # non-finite result aborts before a result tensor exists, so the
        # input and every other state stays untouched on failure.
        if isinstance(data, list):
            out_data = []
            for x in data:
                if x < 0.0:
                    raise ValueError("sqrt input must be non-negative")
                y = math.sqrt(x)
                if not math.isfinite(y):
                    raise ValueError("sqrt result must be finite")
                out_data.append(y)
        else:
            if data < 0.0:
                raise ValueError("sqrt input must be non-negative")
            out_data = math.sqrt(data)
            if not math.isfinite(out_data):
                raise ValueError("sqrt result must be finite")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Snapshot both the forward input and output so later caller-side
        # mutation or replacement of the input cannot change a pending
        # backward pass; dx = g / (2*y) needs y, and x == 0.0 makes the
        # derivative undefined.
        snapshot_x = list(data) if isinstance(data, list) else data
        snapshot_y = (
            list(out_data) if isinstance(out_data, list) else out_data
        )

        def backward_fn(grad):
            # Elementwise g/(2.0*y) in ascending index order; an x == 0.0
            # entry has no finite derivative, each intermediate is checked
            # as it is produced, and the generic engine validates the
            # contribution itself and every merge into an existing grad.
            if isinstance(grad, list):
                contribution = []
                for i in range(len(snapshot_x)):
                    if snapshot_x[i] == 0.0:
                        raise ValueError(
                            "sqrt is not differentiable at zero"
                        )
                    denominator = 2.0 * snapshot_y[i]
                    if not math.isfinite(denominator):
                        raise ValueError(
                            "sqrt backward intermediate must be finite"
                        )
                    value = grad[i] / denominator
                    if not math.isfinite(value):
                        raise ValueError(
                            "sqrt backward intermediate must be finite"
                        )
                    contribution.append(value)
            else:
                if snapshot_x == 0.0:
                    raise ValueError("sqrt is not differentiable at zero")
                denominator = 2.0 * snapshot_y
                if not math.isfinite(denominator):
                    raise ValueError(
                        "sqrt backward intermediate must be finite"
                    )
                value = grad / denominator
                if not math.isfinite(value):
                    raise ValueError(
                        "sqrt backward intermediate must be finite"
                    )
                contribution = value
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def sum(self):
        if isinstance(self.data, list):
            out_data = sum(self.data)
        else:
            out_data = self.data
        _ensure_finite_data(out_data)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self

        def backward_fn(grad):
            if isinstance(parent.data, list):
                return [(parent, [grad] * len(parent.data))]
            return [(parent, grad)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def prod(self):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints and other types are a TypeError; an
        # empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if isinstance(data, list):
            # Accumulate from 1.0 in ascending index order; a non-finite
            # partial product aborts before a result tensor exists, so no
            # state can change on failure.
            acc = 1.0
            for value in data:
                acc *= value
                if not math.isfinite(acc):
                    raise ValueError("prod intermediate must be finite")
            out_data = acc
        else:
            # A scalar's product is the scalar itself.
            out_data = data
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # _validate_data already returns a fresh list for vector input,
        # so `data` is a snapshot: later caller-side mutation or
        # replacement of the input cannot change a pending backward pass.
        snapshot = data

        def backward_fn(grad):
            # Scalar input: the contribution is g itself. Vector input:
            # element i receives g times the product of every x[j] with
            # j != i, accumulated from 1.0 in ascending index order; a
            # single-element vector therefore gets the empty product 1.0
            # and zero elements contribute their mathematical product.
            if not isinstance(snapshot, list):
                return [(parent, grad)]
            contribution = []
            for i in range(len(snapshot)):
                cofactor = 1.0
                for j in range(len(snapshot)):
                    if j == i:
                        continue
                    cofactor *= snapshot[j]
                    if not math.isfinite(cofactor):
                        raise ValueError(
                            "prod backward intermediate must be finite"
                        )
                value = cofactor * grad
                if not math.isfinite(value):
                    raise ValueError(
                        "prod backward intermediate must be finite"
                    )
                contribution.append(value)
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def mean(self):
        # Re-validate at call time since the data may have been mutated
        # after construction: a finite float scalar or a non-empty 1D
        # float list. bools, ints and other types are a TypeError; an
        # empty list or any non-finite value is a ValueError.
        data = _validate_data(self.data)
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if isinstance(data, list):
            # Accumulate from 0.0 in ascending index order, then divide
            # by the length; a non-finite partial sum or quotient aborts
            # before a result tensor exists, so no state can change on
            # failure.
            total = 0.0
            for value in data:
                total += value
                if not math.isfinite(total):
                    raise ValueError("mean intermediate must be finite")
            out_data = total / len(data)
            if not math.isfinite(out_data):
                raise ValueError("mean intermediate must be finite")
        else:
            # A scalar's mean is the scalar itself.
            out_data = data
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Snapshot the input shape so later caller-side mutation or
        # replacement of the input cannot change a pending backward pass.
        snapshot_n = len(data) if isinstance(data, list) else None

        def backward_fn(grad):
            # Scalar input: the contribution is g itself; vector input:
            # every element receives g/n. The generic engine validates
            # the contribution itself and every merge into an existing
            # grad.
            if snapshot_n is None:
                return [(parent, grad)]
            contribution = []
            for _ in range(snapshot_n):
                value = grad / snapshot_n
                if not math.isfinite(value):
                    raise ValueError(
                        "mean backward intermediate must be finite"
                    )
                contribution.append(value)
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def variance(self, correction=0):
        data = _require_nonempty_float_vector(self, "variance")
        if isinstance(correction, bool) or not isinstance(correction, int):
            raise TypeError("correction must be a non-bool int")
        if correction < 0 or correction >= len(data):
            raise ValueError(
                "correction must satisfy 0 <= correction < len(data)"
            )
        # Snapshot the input so later caller-side mutation cannot change
        # either the forward result or a pending backward pass.
        x = list(data)
        n = len(x)
        # Two passes, each accumulating from 0.0 in ascending index order,
        # multiplying before adding at each index; a non-finite
        # intermediate aborts before a result tensor exists, so no state
        # can change on failure.
        total = 0.0
        for i in range(n):
            total += x[i]
            if not math.isfinite(total):
                raise ValueError("variance intermediate must be finite")
        mu = total / n
        if not math.isfinite(mu):
            raise ValueError("variance intermediate must be finite")
        squared_sum = 0.0
        for i in range(n):
            diff = x[i] - mu
            if not math.isfinite(diff):
                raise ValueError("variance intermediate must be finite")
            square = diff * diff
            if not math.isfinite(square):
                raise ValueError("variance intermediate must be finite")
            squared_sum += square
            if not math.isfinite(squared_sum):
                raise ValueError("variance intermediate must be finite")
        denominator = n - correction
        out_data = squared_sum / denominator
        if not math.isfinite(out_data):
            raise ValueError("variance intermediate must be finite")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        snapshot_x = x
        saved_mu = mu
        saved_denominator = denominator

        def backward_fn(grad):
            dx = [0.0] * n
            for i in range(n):
                scaled = grad * 2.0
                if not math.isfinite(scaled):
                    raise ValueError(
                        "variance backward intermediate must be finite"
                    )
                diff = snapshot_x[i] - saved_mu
                if not math.isfinite(diff):
                    raise ValueError(
                        "variance backward intermediate must be finite"
                    )
                product = scaled * diff
                if not math.isfinite(product):
                    raise ValueError(
                        "variance backward intermediate must be finite"
                    )
                dx_i = product / saved_denominator
                if not math.isfinite(dx_i):
                    raise ValueError(
                        "variance backward intermediate must be finite"
                    )
                dx[i] = dx_i
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def softmax(self):
        data = self.data
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("softmax requires a non-empty 1D float list")
        for value in data:
            if isinstance(value, bool) or not isinstance(value, float):
                raise TypeError("softmax data elements must be floats")
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        for value in data:
            if not math.isfinite(value):
                raise ValueError("softmax data elements must be finite")
        # Shift by the maximum so exp is only ever evaluated on values
        # in (-inf, 0]; exp(x_i) directly is never computed.
        m = max(data)
        z = []
        for value in data:
            diff = value - m
            if not math.isfinite(diff):
                raise ValueError("softmax intermediate must be finite")
            z_i = math.exp(diff)
            if not math.isfinite(z_i):
                raise ValueError("softmax intermediate must be finite")
            z.append(z_i)
        s = sum(z)
        if not math.isfinite(s):
            raise ValueError("softmax intermediate must be finite")
        out_data = []
        for z_i in z:
            y_i = z_i / s
            if not math.isfinite(y_i):
                raise ValueError("softmax intermediate must be finite")
            out_data.append(y_i)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        out = out_data

        def backward_fn(grad):
            dot = sum(g * y for g, y in zip(grad, out))
            contribution = [y * (g - dot) for g, y in zip(grad, out)]
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def masked_softmax(self, mask, rows, cols):
        data = _require_nonempty_float_vector(self, "masked_softmax")
        for name, value in (("rows", rows), ("cols", cols)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        total = len(data)
        if total != rows * cols:
            raise ValueError("data length must equal rows * cols")
        if not isinstance(mask, list):
            raise TypeError("mask must be a list of bools")
        if len(mask) != total:
            raise ValueError("mask length must equal data length")
        for value in mask:
            if not isinstance(value, bool):
                raise TypeError("mask elements must be bools")
        for q in range(rows):
            base = q * cols
            if not any(mask[base:base + cols]):
                raise ValueError("each row must contain at least one true")
        # Snapshot the inputs at call time so later caller-side mutation
        # or replacement of data or mask can change neither the forward
        # result nor a pending backward pass.
        x = list(data)
        snapshot_mask = list(mask)
        # Per-row stable softmax over the true positions only, in row-major
        # and ascending column order: m = max of the row's true positions,
        # z_i = exp(x_i - m), s = sum(z), y_i = z_i / s; false positions
        # stay 0.0. exp is only ever evaluated on values in (-inf, 0];
        # exp(x_i) directly is never computed. A non-finite intermediate
        # aborts before a result tensor exists, so no state can change on
        # failure.
        out_data = [0.0] * total
        for q in range(rows):
            base = q * cols
            m = None
            for j in range(cols):
                if snapshot_mask[base + j]:
                    value = x[base + j]
                    if m is None or value > m:
                        m = value
            z = []
            s = 0.0
            for j in range(cols):
                k = base + j
                if not snapshot_mask[k]:
                    continue
                diff = x[k] - m
                if not math.isfinite(diff):
                    raise ValueError(
                        "masked_softmax intermediate must be finite"
                    )
                z_i = math.exp(diff)
                if not math.isfinite(z_i):
                    raise ValueError(
                        "masked_softmax intermediate must be finite"
                    )
                z.append((k, z_i))
                s += z_i
                if not math.isfinite(s):
                    raise ValueError(
                        "masked_softmax intermediate must be finite"
                    )
            for k, z_i in z:
                y_i = z_i / s
                if not math.isfinite(y_i):
                    raise ValueError(
                        "masked_softmax intermediate must be finite"
                    )
                out_data[k] = y_i
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Save mask and y with the graph so a pending backward pass is
        # independent of any subsequent input or output mutation.
        saved_mask = snapshot_mask
        saved_y = list(out_data)

        def backward_fn(grad):
            # Per row, over the true positions in ascending column order:
            # d = sum(g_i * y_i), then dx_i = y_i * (g_i - d); false
            # positions contribute 0.0. Each intermediate is checked as
            # it is produced, and the generic engine validates the
            # contribution itself and every merge into an existing grad.
            dx = [0.0] * total
            for q in range(rows):
                base = q * cols
                d = 0.0
                for j in range(cols):
                    k = base + j
                    if not saved_mask[k]:
                        continue
                    product = grad[k] * saved_y[k]
                    if not math.isfinite(product):
                        raise ValueError(
                            "masked_softmax backward intermediate"
                            " must be finite"
                        )
                    d += product
                    if not math.isfinite(d):
                        raise ValueError(
                            "masked_softmax backward intermediate"
                            " must be finite"
                        )
                for j in range(cols):
                    k = base + j
                    if not saved_mask[k]:
                        continue
                    difference = grad[k] - d
                    if not math.isfinite(difference):
                        raise ValueError(
                            "masked_softmax backward intermediate"
                            " must be finite"
                        )
                    contribution = saved_y[k] * difference
                    if not math.isfinite(contribution):
                        raise ValueError(
                            "masked_softmax backward intermediate"
                            " must be finite"
                        )
                    dx[k] = contribution
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def log_softmax(self):
        data = _require_nonempty_float_vector(self, "log_softmax")
        out_data = _log_softmax_values(data)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        out = out_data

        def backward_fn(grad):
            total = sum(grad)
            contribution = [
                g - math.exp(l_i) * total for g, l_i in zip(grad, out)
            ]
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def log_softmax_batch(self, rows, cols):
        data = _require_nonempty_float_vector(self, "log_softmax_batch")
        for name, value in (("rows", rows), ("cols", cols)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        total = len(data)
        if total != rows * cols:
            raise ValueError("data length must equal rows * cols")
        # Snapshot the input at call time so later caller-side mutation or
        # replacement of data can change neither the forward result nor a
        # pending backward pass.
        x = list(data)
        # Per-row stable log-softmax in row-major, ascending column order:
        # m = max of the row, s accumulated from 0.0 as s += exp(x_i - m),
        # then l_i = x_i - m - log(s). exp is only ever evaluated on
        # values in (-inf, 0]; exp(x_i) directly is never computed. A
        # non-finite intermediate aborts before a result tensor exists, so
        # no state can change on failure.
        out_data = [0.0] * total
        for q in range(rows):
            base = q * cols
            m = x[base]
            for j in range(1, cols):
                if x[base + j] > m:
                    m = x[base + j]
            diffs = []
            s = 0.0
            for j in range(cols):
                diff = x[base + j] - m
                if not math.isfinite(diff):
                    raise ValueError(
                        "log_softmax_batch intermediate must be finite"
                    )
                diffs.append(diff)
                z_i = math.exp(diff)
                if not math.isfinite(z_i):
                    raise ValueError(
                        "log_softmax_batch intermediate must be finite"
                    )
                s += z_i
                if not math.isfinite(s):
                    raise ValueError(
                        "log_softmax_batch intermediate must be finite"
                    )
            log_s = math.log(s)
            if not math.isfinite(log_s):
                raise ValueError(
                    "log_softmax_batch intermediate must be finite"
                )
            for j in range(cols):
                l_i = diffs[j] - log_s
                if not math.isfinite(l_i):
                    raise ValueError(
                        "log_softmax_batch intermediate must be finite"
                    )
                out_data[base + j] = l_i
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Save a private copy of l with the graph so later mutation of the
        # input or output data lists cannot change a pending backward pass.
        saved_l = list(out_data)

        def backward_fn(grad):
            # Per row, in ascending column order: G accumulated from 0.0
            # as G += g_i, then dx_i = g_i - exp(l_i) * G. Each
            # intermediate is checked as it is produced, and the generic
            # engine validates the contribution itself and every merge
            # into an existing grad.
            dx = [0.0] * total
            for q in range(rows):
                base = q * cols
                G = 0.0
                for j in range(cols):
                    G += grad[base + j]
                    if not math.isfinite(G):
                        raise ValueError(
                            "log_softmax_batch backward intermediate"
                            " must be finite"
                        )
                for j in range(cols):
                    k = base + j
                    e = math.exp(saved_l[k])
                    if not math.isfinite(e):
                        raise ValueError(
                            "log_softmax_batch backward intermediate"
                            " must be finite"
                        )
                    product = e * G
                    if not math.isfinite(product):
                        raise ValueError(
                            "log_softmax_batch backward intermediate"
                            " must be finite"
                        )
                    contribution = grad[k] - product
                    if not math.isfinite(contribution):
                        raise ValueError(
                            "log_softmax_batch backward intermediate"
                            " must be finite"
                        )
                    dx[k] = contribution
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def logcumsumexp(self):
        # Re-validate at call time since the data may have been mutated
        # after construction: a non-empty 1D float list. Non-list input
        # (including scalar floats, bools, ints, tuples, nested values)
        # and empty lists are a ValueError; non-float elements and a
        # non-bool requires_grad flag are a TypeError; non-finite values
        # are a ValueError.
        data = _require_nonempty_float_vector(self, "logcumsumexp")
        # Snapshot the input at call time so later caller-side mutation or
        # replacement of data can change neither the forward result nor a
        # pending backward pass.
        x = list(data)
        n = len(x)
        # Stable one-dimensional prefix log-sum-exp recurrence; exp is only
        # ever evaluated on values in (-inf, 0] and exp(x) directly is
        # never accumulated:
        #   y[0] = x[0]
        #   y[i] = m + log1p(exp(d)), m = max(y[i-1], x[i]),
        #          d = min(y[i-1], x[i]) - m
        # A non-finite intermediate aborts before a result tensor exists,
        # so no state can change on failure.
        y = [x[0]]
        for i in range(1, n):
            m = max(y[i - 1], x[i])
            d = min(y[i - 1], x[i]) - m
            if d == float("-inf"):
                # A difference collapsed to -inf: its exponential is 0.
                e = 0.0
            else:
                if not math.isfinite(d):
                    raise ValueError(
                        "logcumsumexp intermediate must be finite"
                    )
                e = math.exp(d)
                if not math.isfinite(e):
                    raise ValueError(
                        "logcumsumexp intermediate must be finite"
                    )
            log_term = math.log1p(e)
            if not math.isfinite(log_term):
                raise ValueError("logcumsumexp intermediate must be finite")
            value = m + log_term
            if not math.isfinite(value):
                raise ValueError("logcumsumexp intermediate must be finite")
            y.append(value)
        if not self.requires_grad:
            return Tensor._make(y, False, (), None)
        parent = self
        # Save private copies of x and y so later mutation of the input or
        # output data lists cannot change a pending backward pass.
        saved_x = list(x)
        saved_y = list(y)

        def backward_fn(grad):
            # dy[i]/dx[j] = exp(x[j] - y[i]) for j <= i (and 0 for j > i),
            # so dx[j] accumulates grad[i] * exp(x[j] - y[i]) from 0.0 in
            # ascending j, then ascending i from j to n-1. Every exponent
            # is non-positive; a difference of -inf contributes 0.0. Any
            # other non-finite difference, exponent, product, partial sum
            # or final contribution aborts the whole pass before anything
            # is returned, leaving every graph grad untouched.
            dx = [0.0] * n
            for j in range(n):
                total = 0.0
                for i in range(j, n):
                    diff = saved_x[j] - saved_y[i]
                    if diff == float("-inf"):
                        e = 0.0
                    else:
                        if not math.isfinite(diff):
                            raise ValueError(
                                "logcumsumexp backward intermediate "
                                "must be finite"
                            )
                        e = math.exp(diff)
                        if not math.isfinite(e):
                            raise ValueError(
                                "logcumsumexp backward intermediate "
                                "must be finite"
                            )
                    contribution_i = grad[i] * e
                    if not math.isfinite(contribution_i):
                        raise ValueError(
                            "logcumsumexp backward intermediate must be finite"
                        )
                    total += contribution_i
                    if not math.isfinite(total):
                        raise ValueError(
                            "logcumsumexp backward intermediate must be finite"
                        )
                dx[j] = total
            return [(parent, dx)]

        return Tensor._make(y, True, (parent,), backward_fn)

    def logsumexp(self):
        data = _require_nonempty_float_vector(self, "logsumexp")
        # Snapshot the input so later caller-side mutation cannot change
        # either the forward result or a pending backward pass.
        x = list(data)
        # Shift by the maximum so exp is only ever evaluated on values in
        # (-inf, 0]; exp(x_i) directly is never computed.
        m = max(x)
        z = []
        s = 0.0
        for x_i in x:
            diff = x_i - m
            if not math.isfinite(diff):
                raise ValueError("logsumexp intermediate must be finite")
            z_i = math.exp(diff)
            if not math.isfinite(z_i):
                raise ValueError("logsumexp intermediate must be finite")
            z.append(z_i)
            s += z_i
            if not math.isfinite(s):
                raise ValueError("logsumexp intermediate must be finite")
        log_s = math.log(s)
        if not math.isfinite(log_s):
            raise ValueError("logsumexp intermediate must be finite")
        out_data = m + log_s
        if not math.isfinite(out_data):
            raise ValueError("logsumexp intermediate must be finite")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        snapshot_x = x
        snapshot_z = z
        saved_s = s

        def backward_fn(grad):
            dx = [0.0] * len(snapshot_x)
            for i, z_i in enumerate(snapshot_z):
                product = grad * z_i
                if not math.isfinite(product):
                    raise ValueError(
                        "logsumexp backward intermediate must be finite"
                    )
                dx_i = product / saved_s
                if not math.isfinite(dx_i):
                    raise ValueError(
                        "logsumexp backward intermediate must be finite"
                    )
                dx[i] = dx_i
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def cross_entropy(self, target, label_smoothing=0.0):
        data = _require_nonempty_float_vector(self, "cross_entropy")
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("target must be a non-bool int")
        n = len(data)
        if target < 0 or target >= n:
            raise ValueError("target index out of range")
        if isinstance(label_smoothing, bool) or not isinstance(
            label_smoothing, float
        ):
            raise TypeError(
                "label_smoothing must be a finite float in [0.0, 1.0)"
            )
        if (
            not math.isfinite(label_smoothing)
            or label_smoothing < 0.0
            or label_smoothing >= 1.0
        ):
            raise ValueError(
                "label_smoothing must be a finite float in [0.0, 1.0)"
            )
        alpha = label_smoothing
        # Snapshot the input at call time so later caller-side mutation or
        # replacement of data can change neither the forward result nor a
        # pending backward pass.
        x = list(data)
        # Stable log-softmax in ascending index order: m = max(x),
        # z_i = exp(x_i - m), s = sum(z), l_i = x_i - m - log(s). exp is
        # only ever evaluated on values in (-inf, 0]. A non-finite
        # intermediate aborts before a result tensor exists, so no state
        # can change on failure.
        m = max(x)
        z = []
        s = 0.0
        for x_i in x:
            diff = x_i - m
            if not math.isfinite(diff):
                raise ValueError("cross_entropy intermediate must be finite")
            z_i = math.exp(diff)
            if not math.isfinite(z_i):
                raise ValueError("cross_entropy intermediate must be finite")
            z.append(z_i)
            s += z_i
            if not math.isfinite(s):
                raise ValueError("cross_entropy intermediate must be finite")
        log_s = math.log(s)
        if not math.isfinite(log_s):
            raise ValueError("cross_entropy intermediate must be finite")
        l = []
        for x_i in x:
            l_i = x_i - m - log_s
            if not math.isfinite(l_i):
                raise ValueError("cross_entropy intermediate must be finite")
            l.append(l_i)
        # Smoothed target distribution: q_i = alpha/n, with an extra
        # 1.0 - alpha added at the target index.
        base = alpha / n
        if not math.isfinite(base):
            raise ValueError("cross_entropy intermediate must be finite")
        q = [base] * n
        q_target = base + (1.0 - alpha)
        if not math.isfinite(q_target):
            raise ValueError("cross_entropy intermediate must be finite")
        q[target] = q_target
        # L = -sum_i q_i * l_i, subtracting each contribution from a
        # running total starting at 0.0 in ascending index order.
        out_data = 0.0
        for i in range(n):
            contribution = q[i] * l[i]
            if not math.isfinite(contribution):
                raise ValueError("cross_entropy intermediate must be finite")
            out_data -= contribution
            if not math.isfinite(out_data):
                raise ValueError("cross_entropy intermediate must be finite")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Save l and q with the graph so a pending backward pass is
        # independent of any subsequent data mutation.
        snapshot_l = l
        snapshot_q = q

        def backward_fn(grad):
            # dx_i = g * (exp(l_i) - q_i); exp is only evaluated on
            # l_i <= 0. Each intermediate is checked as it is produced,
            # and the generic engine validates the contribution itself and
            # every merge into an existing grad.
            dx = []
            for i in range(n):
                e_i = math.exp(snapshot_l[i])
                if not math.isfinite(e_i):
                    raise ValueError(
                        "cross_entropy backward intermediate must be finite"
                    )
                difference = e_i - snapshot_q[i]
                if not math.isfinite(difference):
                    raise ValueError(
                        "cross_entropy backward intermediate must be finite"
                    )
                value = grad * difference
                if not math.isfinite(value):
                    raise ValueError(
                        "cross_entropy backward intermediate must be finite"
                    )
                dx.append(value)
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def cross_entropy_batch(self, targets, classes):
        data = _require_nonempty_float_vector(self, "cross_entropy_batch")
        if isinstance(classes, bool) or not isinstance(classes, int):
            raise TypeError("classes must be a positive int")
        if classes <= 0:
            raise ValueError("classes must be a positive int")
        total = len(data)
        if total % classes != 0:
            raise ValueError("data length must be divisible by classes")
        B = total // classes
        if not isinstance(targets, list):
            raise TypeError("targets must be a list of non-bool ints")
        if len(targets) != B:
            raise ValueError("targets length must equal the batch size")
        for target in targets:
            if isinstance(target, bool) or not isinstance(target, int):
                raise TypeError("targets elements must be non-bool ints")
        for target in targets:
            if target < 0 or target >= classes:
                raise ValueError("targets entries must lie in [0, classes)")
        # Snapshot the inputs at call time so later caller-side mutation or
        # replacement of data or targets can change neither the forward
        # result nor a pending backward pass.
        x = list(data)
        snapshot_targets = list(targets)
        # Per-row stable log-softmax in ascending order: m = max(row),
        # z_i = exp(x_i - m), s = sum(z), l_i = x_i - m - log(s). exp is
        # only ever evaluated on values in (-inf, 0]. L accumulates
        # -l_target from 0.0 over rows in ascending order, then is divided
        # by B. A non-finite intermediate aborts before a result tensor
        # exists, so no state can change on failure.
        l = []
        out_data = 0.0
        for q in range(B):
            base = q * classes
            row = x[base:base + classes]
            m = max(row)
            s = 0.0
            for x_i in row:
                diff = x_i - m
                if not math.isfinite(diff):
                    raise ValueError(
                        "cross_entropy_batch intermediate must be finite"
                    )
                z_i = math.exp(diff)
                if not math.isfinite(z_i):
                    raise ValueError(
                        "cross_entropy_batch intermediate must be finite"
                    )
                s += z_i
                if not math.isfinite(s):
                    raise ValueError(
                        "cross_entropy_batch intermediate must be finite"
                    )
            log_s = math.log(s)
            if not math.isfinite(log_s):
                raise ValueError(
                    "cross_entropy_batch intermediate must be finite"
                )
            for x_i in row:
                l_i = x_i - m - log_s
                if not math.isfinite(l_i):
                    raise ValueError(
                        "cross_entropy_batch intermediate must be finite"
                    )
                l.append(l_i)
            out_data -= l[base + snapshot_targets[q]]
            if not math.isfinite(out_data):
                raise ValueError(
                    "cross_entropy_batch intermediate must be finite"
                )
        out_data = out_data / B
        if not math.isfinite(out_data):
            raise ValueError("cross_entropy_batch result must be finite")
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Save l and targets with the graph so a pending backward pass is
        # independent of any subsequent input mutation.
        snapshot_l = l

        def backward_fn(grad):
            # dx_i = g * (exp(l_i) - 1[i is the row target]) / B; exp is
            # only evaluated on l_i <= 0. Each intermediate is checked as
            # it is produced, and the generic engine validates the
            # contribution itself and every merge into an existing grad.
            dx = []
            for q in range(B):
                base = q * classes
                target = snapshot_targets[q]
                for i in range(classes):
                    e_i = math.exp(snapshot_l[base + i])
                    if not math.isfinite(e_i):
                        raise ValueError(
                            "cross_entropy_batch backward intermediate"
                            " must be finite"
                        )
                    difference = e_i - (1.0 if i == target else 0.0)
                    if not math.isfinite(difference):
                        raise ValueError(
                            "cross_entropy_batch backward intermediate"
                            " must be finite"
                        )
                    value = grad * difference / B
                    if not math.isfinite(value):
                        raise ValueError(
                            "cross_entropy_batch backward intermediate"
                            " must be finite"
                        )
                    dx.append(value)
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def soft_cross_entropy(self, target):
        if not isinstance(target, Tensor):
            raise TypeError("target must be a Tensor")
        x = _require_nonempty_float_vector(self, "soft_cross_entropy")
        q = _require_nonempty_float_vector(target, "soft_cross_entropy")
        if len(x) != len(q):
            raise ValueError("soft_cross_entropy vector lengths must match")
        for q_i in q:
            if q_i < 0.0 or q_i > 1.0:
                raise ValueError(
                    "soft_cross_entropy target values must lie in [0.0, 1.0]"
                )
        # Snapshot both operands at call time so later caller-side mutation or
        # replacement of either input can change neither the forward result
        # nor a pending backward pass.
        x = list(x)
        q = list(q)
        n = len(x)
        # The target must be a probability distribution: sum the target in
        # ascending index order starting from 0.0 and require exactly 1.0.
        q_total = 0.0
        for q_i in q:
            q_total += q_i
            if not math.isfinite(q_total):
                raise ValueError(
                    "soft_cross_entropy target must sum to 1.0"
                )
        if q_total != 1.0:
            raise ValueError(
                "soft_cross_entropy target must sum to 1.0"
            )
        # Stable log-softmax in ascending index order: m = max(x),
        # z_i = exp(x_i - m), s accumulated from 0.0, l_i = x_i - m -
        # log(s). exp is only ever evaluated on values in (-inf, 0]. A
        # non-finite intermediate aborts before a result tensor exists, so
        # no state can change on failure.
        m = max(x)
        s = 0.0
        l = []
        for x_i in x:
            diff = x_i - m
            if not math.isfinite(diff):
                raise ValueError(
                    "soft_cross_entropy intermediate must be finite"
                )
            z_i = math.exp(diff)
            if not math.isfinite(z_i):
                raise ValueError(
                    "soft_cross_entropy intermediate must be finite"
                )
            s += z_i
            if not math.isfinite(s):
                raise ValueError(
                    "soft_cross_entropy intermediate must be finite"
                )
        log_s = math.log(s)
        if not math.isfinite(log_s):
            raise ValueError("soft_cross_entropy intermediate must be finite")
        for x_i in x:
            l_i = x_i - m - log_s
            if not math.isfinite(l_i):
                raise ValueError(
                    "soft_cross_entropy intermediate must be finite"
                )
            l.append(l_i)
        # L = -sum_i q_i * l_i, subtracting each contribution from a
        # running total starting at 0.0 in ascending index order.
        out_data = 0.0
        for i in range(n):
            contribution = q[i] * l[i]
            if not math.isfinite(contribution):
                raise ValueError(
                    "soft_cross_entropy intermediate must be finite"
                )
            out_data -= contribution
            if not math.isfinite(out_data):
                raise ValueError(
                    "soft_cross_entropy intermediate must be finite"
                )
        if not (self.requires_grad or target.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_other = self, target
        # Save l and q with the graph so a pending backward pass is
        # independent of any subsequent data mutation.
        snapshot_l = l
        snapshot_q = q

        def backward_fn(grad):
            # dx_i = g * (exp(l_i) - q_i), dq_i = -g * l_i; exp is only
            # evaluated on l_i <= 0. Each intermediate is checked as it is
            # produced, and the generic engine validates the contributions
            # themselves and every merge into an existing grad. Only the
            # sides that require grad are computed and submitted; when the
            # same tensor is both sides the two contributions are returned
            # for that one parent and summed by the engine.
            contributions = []
            if parent_self.requires_grad:
                dx = []
                for i in range(n):
                    e_i = math.exp(snapshot_l[i])
                    if not math.isfinite(e_i):
                        raise ValueError(
                            "soft_cross_entropy backward intermediate must be"
                            " finite"
                        )
                    difference = e_i - snapshot_q[i]
                    if not math.isfinite(difference):
                        raise ValueError(
                            "soft_cross_entropy backward intermediate must be"
                            " finite"
                        )
                    dx_i = grad * difference
                    if not math.isfinite(dx_i):
                        raise ValueError(
                            "soft_cross_entropy backward intermediate must be"
                            " finite"
                        )
                    dx.append(dx_i)
                contributions.append((parent_self, dx))
            if parent_other.requires_grad:
                dq = []
                for i in range(n):
                    scaled = grad * snapshot_l[i]
                    if not math.isfinite(scaled):
                        raise ValueError(
                            "soft_cross_entropy backward intermediate must be"
                            " finite"
                        )
                    dq_i = -scaled
                    if not math.isfinite(dq_i):
                        raise ValueError(
                            "soft_cross_entropy backward intermediate must be"
                            " finite"
                        )
                    dq.append(dq_i)
                contributions.append((parent_other, dq))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_other), backward_fn
        )

    def conv1d(self, kernel, stride=1, padding=0, dilation=1):
        if not isinstance(kernel, Tensor):
            raise TypeError("kernel must be a Tensor")
        data = _require_nonempty_float_vector(self, "conv1d")
        weights = _require_nonempty_float_vector(kernel, "conv1d")
        if isinstance(stride, bool) or not isinstance(stride, int):
            raise TypeError("stride must be a positive int")
        if stride <= 0:
            raise ValueError("stride must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        if isinstance(dilation, bool) or not isinstance(dilation, int):
            raise TypeError("dilation must be a positive int")
        if dilation <= 0:
            raise ValueError("dilation must be a positive int")
        n = len(data)
        k = len(weights)
        effective = dilation * (k - 1) + 1
        out_len = (n + 2 * padding - effective) // stride + 1
        if out_len <= 0:
            raise ValueError("conv1d output length must be positive")
        # Cross-correlation: the kernel is never flipped; out-of-range
        # input positions (from padding) are skipped.
        out_data = []
        for o in range(out_len):
            acc = 0.0
            for i in range(k):
                j = o * stride + i * dilation - padding
                if 0 <= j < n:
                    product = data[j] * weights[i]
                    if not math.isfinite(product):
                        raise ValueError(
                            "conv1d intermediate must be finite"
                        )
                    acc += product
                    if not math.isfinite(acc):
                        raise ValueError(
                            "conv1d intermediate must be finite"
                        )
            out_data.append(acc)
        if not (self.requires_grad or kernel.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_kernel = self, kernel

        def backward_fn(grad):
            dx = [0.0] * n
            dk = [0.0] * k
            for o in range(out_len):
                g = grad[o]
                for i in range(k):
                    j = o * stride + i * dilation - padding
                    if 0 <= j < n:
                        contrib_x = g * weights[i]
                        if not math.isfinite(contrib_x):
                            raise ValueError(
                                "conv1d backward intermediate must be finite"
                            )
                        dx[j] += contrib_x
                        if not math.isfinite(dx[j]):
                            raise ValueError(
                                "conv1d backward intermediate must be finite"
                            )
                        contrib_k = g * data[j]
                        if not math.isfinite(contrib_k):
                            raise ValueError(
                                "conv1d backward intermediate must be finite"
                            )
                        dk[i] += contrib_k
                        if not math.isfinite(dk[i]):
                            raise ValueError(
                                "conv1d backward intermediate must be finite"
                            )
            contributions = []
            if parent_self.requires_grad:
                contributions.append((parent_self, dx))
            if parent_kernel.requires_grad:
                contributions.append((parent_kernel, dk))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_kernel), backward_fn
        )

    def conv1d_multi(self, kernel, channels, filters, length, size,
                     stride=1, padding=0, dilation=1, bias=None,
                     padding_mode="zeros"):
        if not isinstance(kernel, Tensor):
            raise TypeError("kernel must be a Tensor")
        if bias is not None and not isinstance(bias, Tensor):
            raise TypeError("bias must be a Tensor or None")
        # The operands must hold non-empty 1D finite float lists; data may
        # have been mutated after construction, so re-validate at call time.
        x_data = _require_nonempty_float_vector(self, "conv1d_multi")
        w_data = _require_nonempty_float_vector(kernel, "conv1d_multi")
        b_data = None
        if bias is not None:
            b_data = _require_nonempty_float_vector(bias, "conv1d_multi")
        # channels, filters, length, size, stride and dilation must be
        # non-bool positive ints, and padding a non-bool non-negative int.
        for name, value in (
            ("channels", channels),
            ("filters", filters),
            ("length", length),
            ("size", size),
            ("stride", stride),
            ("dilation", dilation),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        if not isinstance(padding_mode, str):
            raise TypeError("padding_mode must be a str")
        if padding_mode not in ("zeros", "reflect"):
            raise ValueError(
                "padding_mode must be 'zeros' or 'reflect'"
            )
        C, O, N, K, S, P, D = (
            channels, filters, length, size, stride, padding, dilation
        )
        # self stores C channels of N-length signals and kernel O filters of
        # C channels of K-length weights, all in row-major order.
        if len(x_data) != C * N:
            raise ValueError(
                "conv1d_multi input length must equal channels * length"
            )
        if len(w_data) != O * C * K:
            raise ValueError(
                "conv1d_multi kernel length must equal "
                "filters * channels * size"
            )
        # The bias holds one trainable offset per output filter.
        if b_data is not None and len(b_data) != O:
            raise ValueError(
                "conv1d_multi bias length must equal filters"
            )
        # Reflection padding folds back onto the signal at both edges, which
        # is only defined for a signal of at least two positions and a pad
        # width strictly smaller than that signal.
        if padding_mode == "reflect" and (N <= 1 or P >= N):
            raise ValueError(
                "conv1d_multi reflect padding requires length > 1 and "
                "padding < length"
            )
        # Dilation spaces the kernel taps by D, so the effective receptive
        # field spans E = D*(K-1)+1 positions; with D=1 this is simply K.
        E = D * (K - 1) + 1
        out_len = (N + 2 * P - E) // S + 1
        if out_len <= 0:
            raise ValueError("conv1d_multi output length must be positive")
        # Snapshot all operands at call time so later caller-side mutation
        # or replacement of any input can change neither the forward result
        # nor a pending backward pass.
        snapshot_x = list(x_data)
        snapshot_w = list(w_data)
        snapshot_b = list(b_data) if b_data is not None else None
        # Batched multi-channel 1D cross-correlation: the kernel is never
        # flipped. In ascending (o, t, c, r) order, each output starts at
        # bias[o] (0.0 without a bias) and accumulates
        # y[o*L+t] += x[c*N+j] * w[(o*C+c)*K+r],
        # with j = t*S+r*D-P. With zero padding, out-of-range input
        # positions from padding or the dilation gaps are skipped. With
        # reflection padding j is folded back into range first: j<0 maps
        # to -j, j>=N to 2*N-2-j. A non-finite product or partial sum
        # aborts before a result tensor exists, so no state can change.
        total_out = O * out_len
        out_data = [0.0] * total_out
        for o in range(O):
            bias_o = snapshot_b[o] if snapshot_b is not None else 0.0
            for t in range(out_len):
                oi = o * out_len + t
                out_data[oi] = bias_o
                for c in range(C):
                    for r in range(K):
                        j = t * S + r * D - P
                        if padding_mode == "zeros":
                            if not 0 <= j < N:
                                continue
                        else:
                            if j < 0:
                                j = -j
                            if j >= N:
                                j = 2 * N - 2 - j
                        xi = c * N + j
                        wi = (o * C + c) * K + r
                        product = snapshot_x[xi] * snapshot_w[wi]
                        if not math.isfinite(product):
                            raise ValueError(
                                "conv1d_multi intermediate must be finite"
                            )
                        out_data[oi] = out_data[oi] + product
                        if not math.isfinite(out_data[oi]):
                            raise ValueError(
                                "conv1d_multi intermediate must be finite"
                            )
        needs_grad = self.requires_grad or kernel.requires_grad
        if bias is not None and bias.requires_grad:
            needs_grad = True
        if not needs_grad:
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_kernel = self, kernel
        parent_bias = bias

        def backward_fn(grad):
            # dx[xi] += g*w[wi] and dw[wi] += g*x[xi], accumulated from
            # 0.0 in the same ascending (o, t, c, r) order as the forward
            # pass, using the same dilated j indexing, zero-padding skip
            # and reflection fold (several padded positions may fold onto
            # one xi and so merge into the same dx entry); db[o] += g is
            # accumulated from 0.0 in ascending t order per filter. A
            # non-finite product or partial sum aborts the whole pass
            # before any grad is written.
            dx = [0.0] * (C * N)
            dw = [0.0] * (O * C * K)
            db = [0.0] * O if snapshot_b is not None else None
            for o in range(O):
                for t in range(out_len):
                    g = grad[o * out_len + t]
                    if db is not None:
                        db[o] = db[o] + g
                        if not math.isfinite(db[o]):
                            raise ValueError(
                                "conv1d_multi backward "
                                "intermediate must be finite"
                            )
                    for c in range(C):
                        for r in range(K):
                            j = t * S + r * D - P
                            if padding_mode == "zeros":
                                if not 0 <= j < N:
                                    continue
                            else:
                                if j < 0:
                                    j = -j
                                if j >= N:
                                    j = 2 * N - 2 - j
                            xi = c * N + j
                            wi = (o * C + c) * K + r
                            contrib_x = g * snapshot_w[wi]
                            if not math.isfinite(contrib_x):
                                raise ValueError(
                                    "conv1d_multi backward "
                                    "intermediate must be finite"
                                )
                            dx[xi] = dx[xi] + contrib_x
                            if not math.isfinite(dx[xi]):
                                raise ValueError(
                                    "conv1d_multi backward "
                                    "intermediate must be finite"
                                )
                            contrib_w = g * snapshot_x[xi]
                            if not math.isfinite(contrib_w):
                                raise ValueError(
                                    "conv1d_multi backward "
                                    "intermediate must be finite"
                                )
                            dw[wi] = dw[wi] + contrib_w
                            if not math.isfinite(dw[wi]):
                                raise ValueError(
                                    "conv1d_multi backward "
                                    "intermediate must be finite"
                                )
            # The same object may play several roles (e.g. the bias is
            # also the input); merge such roles into one contribution per
            # tensor, and submit only parents that require grad.
            merged = {}
            roles = [
                (parent_self, dx),
                (parent_kernel, dw),
            ]
            if parent_bias is not None:
                roles.append((parent_bias, db))
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        parents = (parent_self, parent_kernel)
        if parent_bias is not None:
            parents = parents + (parent_bias,)
        return Tensor._make(out_data, True, parents, backward_fn)

    def conv_transpose1d(self, kernel, stride=1, padding=0, dilation=1):
        if not isinstance(kernel, Tensor):
            raise TypeError("kernel must be a Tensor")
        data = _require_nonempty_float_vector(self, "conv_transpose1d")
        weights = _require_nonempty_float_vector(kernel, "conv_transpose1d")
        if isinstance(stride, bool) or not isinstance(stride, int):
            raise TypeError("stride must be a positive int")
        if stride <= 0:
            raise ValueError("stride must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        if isinstance(dilation, bool) or not isinstance(dilation, int):
            raise TypeError("dilation must be a positive int")
        if dilation <= 0:
            raise ValueError("dilation must be a positive int")
        n = len(data)
        k = len(weights)
        s = stride
        p = padding
        d = dilation
        out_len = (n - 1) * s - 2 * p + d * (k - 1) + 1
        if out_len <= 0:
            raise ValueError("conv_transpose1d output length must be positive")
        # Snapshot both operands at call time so later caller-side mutation
        # or replacement of either input can change neither the forward
        # result nor a pending backward pass.
        a = list(data)
        b = list(weights)
        # Dilated transposed cross-correlation: each input element scatters
        # a scaled copy of the dilated kernel onto the output, accumulating
        # out[i*stride - padding + r*dilation] in ascending i then r order
        # and skipping out-of-range positions; a non-finite product or
        # partial sum aborts before a result tensor exists, so no state can
        # change on failure. With dilation=1 this is the plain transposed
        # cross-correlation, preserving the previous behavior.
        out_data = [0.0] * out_len
        for i in range(n):
            for r in range(k):
                j = i * s - p + r * d
                if not 0 <= j < out_len:
                    continue
                product = a[i] * b[r]
                if not math.isfinite(product):
                    raise ValueError(
                        "conv_transpose1d intermediate must be finite"
                    )
                out_data[j] += product
                if not math.isfinite(out_data[j]):
                    raise ValueError(
                        "conv_transpose1d intermediate must be finite"
                    )
        if not (self.requires_grad or kernel.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_kernel = self, kernel

        def backward_fn(grad):
            dx = [0.0] * n
            dk = [0.0] * k
            for i in range(n):
                for r in range(k):
                    j = i * s - p + r * d
                    if not 0 <= j < out_len:
                        continue
                    g = grad[j]
                    contrib_x = g * b[r]
                    if not math.isfinite(contrib_x):
                        raise ValueError(
                            "conv_transpose1d backward intermediate "
                            "must be finite"
                        )
                    dx[i] += contrib_x
                    if not math.isfinite(dx[i]):
                        raise ValueError(
                            "conv_transpose1d backward intermediate "
                            "must be finite"
                        )
                    contrib_k = g * a[i]
                    if not math.isfinite(contrib_k):
                        raise ValueError(
                            "conv_transpose1d backward intermediate "
                            "must be finite"
                        )
                    dk[r] += contrib_k
                    if not math.isfinite(dk[r]):
                        raise ValueError(
                            "conv_transpose1d backward intermediate "
                            "must be finite"
                        )
            contributions = []
            if parent_self.requires_grad:
                contributions.append((parent_self, dx))
            if parent_kernel.requires_grad:
                contributions.append((parent_kernel, dk))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_kernel), backward_fn
        )

    def conv2d(self, kernel, height, width, kernel_size, stride=1,
               padding=0, dilation=1):
        if not isinstance(kernel, Tensor):
            raise TypeError("kernel must be a Tensor")
        # Both operands must hold non-empty 1D finite float lists; data may
        # have been mutated after construction, so re-validate at call time.
        x_data = _require_nonempty_float_vector(self, "conv2d")
        k_data = _require_nonempty_float_vector(kernel, "conv2d")
        # H, W, kernel_size, stride and dilation must be non-bool positive
        # ints, and padding a non-bool non-negative int.
        for name, value in (
            ("height", height),
            ("width", width),
            ("kernel_size", kernel_size),
            ("stride", stride),
            ("dilation", dilation),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        H, W, K, S, P, D = height, width, kernel_size, stride, padding, dilation
        # self stores a H x W image and kernel a K x K filter, both in
        # row-major order. Dilation spaces the kernel taps by D, so the
        # effective receptive field spans E = D*(K-1)+1 positions.
        if len(x_data) != H * W:
            raise ValueError("conv2d input length must equal height * width")
        if len(k_data) != K * K:
            raise ValueError(
                "conv2d kernel length must equal kernel_size * kernel_size"
            )
        E = D * (K - 1) + 1
        out_h = (H + 2 * P - E) // S + 1
        out_w = (W + 2 * P - E) // S + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("conv2d output dimensions must be positive")
        # Snapshot both operands at call time so later caller-side mutation
        # or replacement of either input can change neither the forward
        # result nor a pending backward pass.
        snapshot_x = list(x_data)
        snapshot_k = list(k_data)
        # Dilated 2D cross-correlation: the kernel is never flipped. In
        # ascending (or, oc, ir, ic) order, y[or*OW+oc] += x[j] * k[q]
        # with q = ir*K+ic and
        # j = (or*S+ir*D-P)*W + (oc*S+ic*D-P); out-of-range input
        # positions (from padding or the dilation gaps) are skipped. A
        # non-finite product or partial sum aborts before a result tensor
        # exists, so no state can change.
        out_data = [0.0] * (out_h * out_w)
        for orow in range(out_h):
            for ocol in range(out_w):
                o = orow * out_w + ocol
                for ir in range(K):
                    row = orow * S + ir * D - P
                    if not 0 <= row < H:
                        continue
                    for ic in range(K):
                        col = ocol * S + ic * D - P
                        if not 0 <= col < W:
                            continue
                        j = row * W + col
                        q = ir * K + ic
                        product = snapshot_x[j] * snapshot_k[q]
                        if not math.isfinite(product):
                            raise ValueError(
                                "conv2d intermediate must be finite"
                            )
                        out_data[o] = out_data[o] + product
                        if not math.isfinite(out_data[o]):
                            raise ValueError(
                                "conv2d intermediate must be finite"
                            )
        if not (self.requires_grad or kernel.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_kernel = self, kernel

        def backward_fn(grad):
            # dx[j] += g[o]*k[q] and dk[q] += g[o]*x[j], accumulated from
            # 0.0 in the same ascending (or, oc, ir, ic) order as the
            # forward pass, using the same dilated j indexing and skipping
            # out-of-range positions. A non-finite product or partial sum
            # aborts the whole pass before any grad is written.
            dx = [0.0] * (H * W)
            dk = [0.0] * (K * K)
            for orow in range(out_h):
                for ocol in range(out_w):
                    g = grad[orow * out_w + ocol]
                    for ir in range(K):
                        row = orow * S + ir * D - P
                        if not 0 <= row < H:
                            continue
                        for ic in range(K):
                            col = ocol * S + ic * D - P
                            if not 0 <= col < W:
                                continue
                            j = row * W + col
                            q = ir * K + ic
                            contrib_x = g * snapshot_k[q]
                            if not math.isfinite(contrib_x):
                                raise ValueError(
                                    "conv2d backward intermediate must be"
                                    " finite"
                                )
                            dx[j] = dx[j] + contrib_x
                            if not math.isfinite(dx[j]):
                                raise ValueError(
                                    "conv2d backward intermediate must be"
                                    " finite"
                                )
                            contrib_k = g * snapshot_x[j]
                            if not math.isfinite(contrib_k):
                                raise ValueError(
                                    "conv2d backward intermediate must be"
                                    " finite"
                                )
                            dk[q] = dk[q] + contrib_k
                            if not math.isfinite(dk[q]):
                                raise ValueError(
                                    "conv2d backward intermediate must be"
                                    " finite"
                                )
            # The same object may play both roles; merge such roles into
            # one contribution per tensor, and submit only parents that
            # require grad.
            merged = {}
            roles = (
                (parent_self, dx),
                (parent_kernel, dk),
            )
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_kernel), backward_fn
        )

    def conv_transpose2d(self, kernel, size, kernel_size, stride=1,
                         padding=0, dilation=1):
        if not isinstance(kernel, Tensor):
            raise TypeError("kernel must be a Tensor")
        # Both operands must hold non-empty 1D finite float lists; data may
        # have been mutated after construction, so re-validate at call time.
        x_data = _require_nonempty_float_vector(self, "conv_transpose2d")
        k_data = _require_nonempty_float_vector(kernel, "conv_transpose2d")
        # size, kernel_size, stride and dilation must be non-bool positive
        # ints, and padding a non-bool non-negative int.
        for name, value in (
            ("size", size),
            ("kernel_size", kernel_size),
            ("stride", stride),
            ("dilation", dilation),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        N, K, S, P, D = size, kernel_size, stride, padding, dilation
        # self stores an N x N image and kernel a K x K filter, both in
        # row-major order; the transposed correlation emits an O x O output.
        if len(x_data) != N * N:
            raise ValueError("conv_transpose2d input length must equal size * size")
        if len(k_data) != K * K:
            raise ValueError(
                "conv_transpose2d kernel length must equal "
                "kernel_size * kernel_size"
            )
        O = (N - 1) * S - 2 * P + D * (K - 1) + 1
        if O <= 0:
            raise ValueError("conv_transpose2d output dimensions must be positive")
        # Snapshot both operands at call time so later caller-side mutation
        # or replacement of either input can change neither the forward
        # result nor a pending backward pass.
        snapshot_x = list(x_data)
        snapshot_k = list(k_data)
        # Square 2D transposed cross-correlation: the kernel is never
        # flipped. Each input element scatters a scaled copy of the kernel
        # onto the output; in ascending (r, c, kr, kc) order,
        # i = r*N+c, q = kr*K+kc, y = r*S-P+kr*D and x = c*S-P+kc*D, and
        # out[y*O+x] += a[i]*b[q], accumulated from 0.0 while skipping
        # out-of-range output positions (from stride, padding or the
        # dilation gaps). A non-finite product or partial sum aborts before
        # a result tensor exists, so no state can change.
        out_data = [0.0] * (O * O)
        for r in range(N):
            for c in range(N):
                i = r * N + c
                for kr in range(K):
                    y = r * S - P + kr * D
                    if not 0 <= y < O:
                        continue
                    for kc in range(K):
                        x = c * S - P + kc * D
                        if not 0 <= x < O:
                            continue
                        q = kr * K + kc
                        product = snapshot_x[i] * snapshot_k[q]
                        if not math.isfinite(product):
                            raise ValueError(
                                "conv_transpose2d intermediate must be finite"
                            )
                        out_data[y * O + x] = out_data[y * O + x] + product
                        if not math.isfinite(out_data[y * O + x]):
                            raise ValueError(
                                "conv_transpose2d intermediate must be finite"
                            )
        if not (self.requires_grad or kernel.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_kernel = self, kernel

        def backward_fn(grad):
            # dx[i] += g[y*O+x]*k[q] and dk[q] += g[y*O+x]*a[i],
            # accumulated from 0.0 in the same ascending (r, c, kr, kc)
            # order as the forward pass, using the same y/x indexing and
            # skipping out-of-range positions. A non-finite product or
            # partial sum aborts the whole pass before any grad is written.
            dx = [0.0] * (N * N)
            dk = [0.0] * (K * K)
            for r in range(N):
                for c in range(N):
                    i = r * N + c
                    for kr in range(K):
                        y = r * S - P + kr * D
                        if not 0 <= y < O:
                            continue
                        for kc in range(K):
                            x = c * S - P + kc * D
                            if not 0 <= x < O:
                                continue
                            q = kr * K + kc
                            g = grad[y * O + x]
                            contrib_x = g * snapshot_k[q]
                            if not math.isfinite(contrib_x):
                                raise ValueError(
                                    "conv_transpose2d backward intermediate "
                                    "must be finite"
                                )
                            dx[i] = dx[i] + contrib_x
                            if not math.isfinite(dx[i]):
                                raise ValueError(
                                    "conv_transpose2d backward intermediate "
                                    "must be finite"
                                )
                            contrib_k = g * snapshot_x[i]
                            if not math.isfinite(contrib_k):
                                raise ValueError(
                                    "conv_transpose2d backward intermediate "
                                    "must be finite"
                                )
                            dk[q] = dk[q] + contrib_k
                            if not math.isfinite(dk[q]):
                                raise ValueError(
                                    "conv_transpose2d backward intermediate "
                                    "must be finite"
                                )
            # The same object may play both roles; merge such roles into
            # one contribution per tensor, and submit only parents that
            # require grad.
            merged = {}
            roles = (
                (parent_self, dx),
                (parent_kernel, dk),
            )
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        return Tensor._make(
            out_data, True, (parent_self, parent_kernel), backward_fn
        )

    def conv2d_multi(self, kernel, channels, filters, height, width, size,
                     stride=1, padding=0, dilation=1, groups=1, bias=None,
                     padding_mode="zeros"):
        if not isinstance(kernel, Tensor):
            raise TypeError("kernel must be a Tensor")
        if bias is not None and not isinstance(bias, Tensor):
            raise TypeError("bias must be a Tensor or None")
        # Both operands must hold non-empty 1D finite float lists; data may
        # have been mutated after construction, so re-validate at call time.
        x_data = _require_nonempty_float_vector(self, "conv2d_multi")
        w_data = _require_nonempty_float_vector(kernel, "conv2d_multi")
        b_data = None
        if bias is not None:
            b_data = _require_nonempty_float_vector(bias, "conv2d_multi")
        # channels, filters, height, width, size, stride, dilation and
        # groups must be non-bool positive ints, and padding a non-bool
        # non-negative int.
        for name, value in (
            ("channels", channels),
            ("filters", filters),
            ("height", height),
            ("width", width),
            ("size", size),
            ("stride", stride),
            ("dilation", dilation),
            ("groups", groups),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        if not isinstance(padding_mode, str):
            raise TypeError("padding_mode must be a str")
        if padding_mode not in ("zeros", "reflect"):
            raise ValueError(
                "padding_mode must be 'zeros' or 'reflect'"
            )
        C, O, H, W, K, S, P, D, G = (
            channels, filters, height, width, size, stride, padding,
            dilation, groups,
        )
        # Groups partition the channels and filters into G disjoint blocks;
        # both counts must divide evenly for the split to be well defined.
        if C % G != 0 or O % G != 0:
            raise ValueError(
                "groups must evenly divide channels and filters"
            )
        CPG = C // G
        FPG = O // G
        # self stores C channels of H x W images and kernel O filters of
        # C/G channels of K x K weights, all in row-major order.
        if len(x_data) != C * H * W:
            raise ValueError(
                "conv2d_multi input length must equal "
                "channels * height * width"
            )
        kernel_len = O * CPG * K * K
        if len(w_data) != kernel_len:
            if G == 1:
                raise ValueError(
                    "conv2d_multi kernel length must equal "
                    "filters * channels * size * size"
                )
            raise ValueError(
                "conv2d_multi kernel length must equal "
                "filters * (channels // groups) * size * size"
            )
        # The bias holds one trainable offset per output channel.
        if b_data is not None and len(b_data) != O:
            raise ValueError(
                "conv2d_multi bias length must equal filters"
            )
        # Reflection padding folds back onto the image at all four edges,
        # which is only defined for spatial dimensions of at least two
        # positions and a pad width strictly smaller than each dimension.
        if padding_mode == "reflect" and (
            H <= 1 or W <= 1 or P >= H or P >= W
        ):
            raise ValueError(
                "conv2d_multi reflect padding requires height > 1, "
                "width > 1, padding < height and padding < width"
            )
        # Dilation spaces the kernel taps by D, so the effective receptive
        # field spans E = D*(K-1)+1 positions; with D=1 this is simply K.
        E = D * (K - 1) + 1
        out_h = (H + 2 * P - E) // S + 1
        out_w = (W + 2 * P - E) // S + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("conv2d_multi output dimensions must be positive")
        # Snapshot all operands at call time so later caller-side mutation
        # or replacement of any input can change neither the forward
        # result nor a pending backward pass.
        snapshot_x = list(x_data)
        snapshot_w = list(w_data)
        snapshot_b = list(b_data) if b_data is not None else None
        # Grouped multi-channel 2D cross-correlation: the kernel is never
        # flipped. Filter o belongs to group g = o // (O/G) and reads only
        # input channels c0 .. c0+C/G-1, where c0 = g*(C/G). In ascending
        # (o, or, oc) order each output starts at bias[o] (0.0 without a
        # bias), then in ascending (cl, kr, kc) order (cl the in-group
        # channel) accumulates
        # y[(o*OH+or)*OW+oc] += x[(c*H+r)*W+q]
        #   * w[((o*(C/G)+cl)*K+kr)*K+kc],
        # with c = c0+cl, r = or*S+kr*D-P and q = oc*S+kc*D-P. With zero
        # padding out-of-range input positions from padding or the dilation
        # gaps are skipped. With reflection padding r and q are folded back
        # into range first: a value < 0 maps to its negation and a value
        # >= H (resp. >= W) to 2*H-2-value (resp. 2*W-2-value); several
        # padded taps may fold onto the same input position. With G=1 this
        # reduces to dense (o, or, oc, c, kr, kc) order. A non-finite
        # product or partial sum aborts before a result tensor exists, so
        # no state can change.
        out_len = O * out_h * out_w
        out_data = [0.0] * out_len
        for o in range(O):
            g = o // FPG
            c0 = g * CPG
            bias_o = snapshot_b[o] if snapshot_b is not None else 0.0
            for orow in range(out_h):
                for ocol in range(out_w):
                    oi = (o * out_h + orow) * out_w + ocol
                    out_data[oi] = bias_o
                    for cl in range(CPG):
                        c = c0 + cl
                        for kr in range(K):
                            r = orow * S + kr * D - P
                            if padding_mode == "zeros":
                                if not 0 <= r < H:
                                    continue
                            else:
                                if r < 0:
                                    r = -r
                                if r >= H:
                                    r = 2 * H - 2 - r
                            for kc in range(K):
                                q = ocol * S + kc * D - P
                                if padding_mode == "zeros":
                                    if not 0 <= q < W:
                                        continue
                                else:
                                    if q < 0:
                                        q = -q
                                    if q >= W:
                                        q = 2 * W - 2 - q
                                xi = (c * H + r) * W + q
                                wi = ((o * CPG + cl) * K + kr) * K + kc
                                product = snapshot_x[xi] * snapshot_w[wi]
                                if not math.isfinite(product):
                                    raise ValueError(
                                        "conv2d_multi intermediate must be"
                                        " finite"
                                    )
                                out_data[oi] = out_data[oi] + product
                                if not math.isfinite(out_data[oi]):
                                    raise ValueError(
                                        "conv2d_multi intermediate must be"
                                        " finite"
                                    )
        needs_grad = self.requires_grad or kernel.requires_grad
        if bias is not None and bias.requires_grad:
            needs_grad = True
        if not needs_grad:
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_kernel = self, kernel
        parent_bias = bias

        def backward_fn(grad):
            # dx[xi] += g*w[wi] and dw[wi] += g*x[xi], accumulated from
            # 0.0 in the same ascending (o, or, oc, cl, kr, kc) order and
            # group-channel mapping as the forward pass, using the same
            # dilated r/q indexing, zero-padding skip and reflection fold
            # (several padded taps may fold onto the same xi and so merge
            # into the same dx/dw entry); db[o] += g, accumulated from
            # 0.0 in ascending (or, oc) order per filter. A non-finite
            # product or partial sum aborts the whole pass before any grad
            # is written.
            dx = [0.0] * (C * H * W)
            dw = [0.0] * (O * CPG * K * K)
            db = [0.0] * O if snapshot_b is not None else None
            for o in range(O):
                g = o // FPG
                c0 = g * CPG
                for orow in range(out_h):
                    for ocol in range(out_w):
                        gv = grad[(o * out_h + orow) * out_w + ocol]
                        if db is not None:
                            db[o] = db[o] + gv
                            if not math.isfinite(db[o]):
                                raise ValueError(
                                    "conv2d_multi backward "
                                    "intermediate must be finite"
                                )
                        for cl in range(CPG):
                            c = c0 + cl
                            for kr in range(K):
                                r = orow * S + kr * D - P
                                if padding_mode == "zeros":
                                    if not 0 <= r < H:
                                        continue
                                else:
                                    if r < 0:
                                        r = -r
                                    if r >= H:
                                        r = 2 * H - 2 - r
                                for kc in range(K):
                                    q = ocol * S + kc * D - P
                                    if padding_mode == "zeros":
                                        if not 0 <= q < W:
                                            continue
                                    else:
                                        if q < 0:
                                            q = -q
                                        if q >= W:
                                            q = 2 * W - 2 - q
                                    xi = (c * H + r) * W + q
                                    wi = (
                                        ((o * CPG + cl) * K + kr) * K + kc
                                    )
                                    contrib_x = gv * snapshot_w[wi]
                                    if not math.isfinite(contrib_x):
                                        raise ValueError(
                                            "conv2d_multi backward "
                                            "intermediate must be finite"
                                        )
                                    dx[xi] = dx[xi] + contrib_x
                                    if not math.isfinite(dx[xi]):
                                        raise ValueError(
                                            "conv2d_multi backward "
                                            "intermediate must be finite"
                                        )
                                    contrib_w = gv * snapshot_x[xi]
                                    if not math.isfinite(contrib_w):
                                        raise ValueError(
                                            "conv2d_multi backward "
                                            "intermediate must be finite"
                                        )
                                    dw[wi] = dw[wi] + contrib_w
                                    if not math.isfinite(dw[wi]):
                                        raise ValueError(
                                            "conv2d_multi backward "
                                            "intermediate must be finite"
                                        )
            # The same object may play several roles; merge such roles
            # into one contribution per tensor, and submit only parents
            # that require grad.
            merged = {}
            roles = [
                (parent_self, dx),
                (parent_kernel, dw),
            ]
            if parent_bias is not None:
                roles.append((parent_bias, db))
            for parent, value in roles:
                if not parent.requires_grad:
                    continue
                entry = merged.get(id(parent))
                if entry is None:
                    merged[id(parent)] = [parent, value]
                else:
                    combined = _merge_grad(entry[1], value)
                    _ensure_finite_grad(combined)
                    entry[1] = combined
            return [(entry[0], entry[1]) for entry in merged.values()]

        parents = (parent_self, parent_kernel)
        if parent_bias is not None:
            parents = parents + (parent_bias,)
        return Tensor._make(out_data, True, parents, backward_fn)

    def max_pool1d(self, kernel_size, stride=None, padding=0, dilation=1):
        data = _require_nonempty_float_vector(self, "max_pool1d")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
            raise TypeError("kernel_size must be a positive int")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive int")
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, bool) or not isinstance(stride, int):
            raise TypeError("stride must be a positive int")
        elif stride <= 0:
            raise ValueError("stride must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        if isinstance(dilation, bool) or not isinstance(dilation, int):
            raise TypeError("dilation must be a positive int")
        if dilation <= 0:
            raise ValueError("dilation must be a positive int")
        n = len(data)
        k = kernel_size
        # Effective window span: adjacent taps sit dilation positions apart.
        effective = dilation * (k - 1) + 1
        out_len = (n + 2 * padding - effective) // stride + 1
        if out_len <= 0:
            raise ValueError("max_pool1d output length must be positive")
        # Out-of-range input positions (from padding) are skipped; on ties
        # the smallest input index wins the gradient.
        out_data = []
        argmax_indices = []
        for o in range(out_len):
            best = None
            best_j = None
            for i in range(k):
                j = o * stride + i * dilation - padding
                if 0 <= j < n and (best is None or data[j] > best):
                    best = data[j]
                    best_j = j
            if best_j is None:
                raise ValueError("max_pool1d window has no valid positions")
            out_data.append(best)
            argmax_indices.append(best_j)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self

        def backward_fn(grad):
            dx = [0.0] * n
            for o in range(out_len):
                dx[argmax_indices[o]] += grad[o]
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def avg_pool1d(self, kernel_size, stride=None, padding=0, dilation=1):
        data = _require_nonempty_float_vector(self, "avg_pool1d")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
            raise TypeError("kernel_size must be a positive int")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive int")
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, bool) or not isinstance(stride, int):
            raise TypeError("stride must be a positive int")
        elif stride <= 0:
            raise ValueError("stride must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        if isinstance(dilation, bool) or not isinstance(dilation, int):
            raise TypeError("dilation must be a positive int")
        if dilation <= 0:
            raise ValueError("dilation must be a positive int")
        n = len(data)
        k = kernel_size
        s = stride
        p = padding
        d = dilation
        out_len = (n + 2 * p - d * (k - 1) - 1) // s + 1
        if out_len <= 0:
            raise ValueError("avg_pool1d output length must be positive")
        # Each window averages over a fixed denominator of k taps; taps
        # landing outside [0, n) (from padding) count as 0.0. Only valid
        # positions are accumulated, from 0.0 in ascending tap order, then
        # the sum is divided by k; a fully out-of-range window yields 0.0.
        # A non-finite partial sum or quotient aborts before a result
        # tensor exists, so no state can change on failure.
        out_data = []
        for o in range(out_len):
            acc = 0.0
            for i in range(k):
                j = o * s + i * d - p
                if 0 <= j < n:
                    acc += data[j]
                    if not math.isfinite(acc):
                        raise ValueError(
                            "avg_pool1d intermediate must be finite"
                        )
            mean = acc / k
            if not math.isfinite(mean):
                raise ValueError("avg_pool1d intermediate must be finite")
            out_data.append(mean)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        # Snapshot the input length and window parameters with the graph;
        # mutating the parent's data after the forward pass cannot change
        # what a pending backward pass uses.
        parent = self
        snapshot_n = n
        snapshot_k = k
        snapshot_s = s
        snapshot_p = p
        snapshot_d = d

        def backward_fn(grad):
            # dx[j] += grad[o] / k for every in-range tap, accumulated in
            # the same ascending (o, i) order as the forward pass; a
            # non-finite division or partial sum aborts the whole pass
            # before any grad is written.
            dx = [0.0] * snapshot_n
            for o in range(out_len):
                for i in range(snapshot_k):
                    j = o * snapshot_s + i * snapshot_d - snapshot_p
                    if 0 <= j < snapshot_n:
                        share = grad[o] / snapshot_k
                        if not math.isfinite(share):
                            raise ValueError(
                                "avg_pool1d backward intermediate must be"
                                " finite"
                            )
                        dx[j] += share
                        if not math.isfinite(dx[j]):
                            raise ValueError(
                                "avg_pool1d backward intermediate must be"
                                " finite"
                            )
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def max_pool2d(
        self, height, width, kernel_size, stride=None, padding=0, dilation=1
    ):
        data = _require_nonempty_float_vector(self, "max_pool2d")
        # height, width, kernel_size, a non-None stride and dilation must be
        # non-bool positive ints; stride=None falls back to kernel_size.
        # padding must be a non-bool non-negative int.
        for name, value in (
            ("height", height),
            ("width", width),
            ("kernel_size", kernel_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, bool) or not isinstance(stride, int):
            raise TypeError("stride must be a positive int")
        elif stride <= 0:
            raise ValueError("stride must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        if isinstance(dilation, bool) or not isinstance(dilation, int):
            raise TypeError("dilation must be a positive int")
        if dilation <= 0:
            raise ValueError("dilation must be a positive int")
        H, W, K, S, P, D = height, width, kernel_size, stride, padding, dilation
        # self stores a single-channel H x W image in row-major order.
        if len(data) != H * W:
            raise ValueError(
                "max_pool2d input length must equal height * width"
            )
        # Effective window span: taps in each axis sit dilation positions
        # apart, so a window covers E = D*(K-1)+1 positions.
        E = D * (K - 1) + 1
        out_h = (H + 2 * P - E) // S + 1
        out_w = (W + 2 * P - E) // S + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("max_pool2d output dimensions must be positive")
        # Scan taps in ascending (or, oc, kr, kc) order; the tap sits at row
        # or*S+kr*D-P and column oc*S+kc*D-P. Out-of-range positions (from
        # padding or dilation) are skipped; a window with no in-range tap is
        # an error. On ties the earliest tap in scan order wins the gradient.
        out_data = []
        argmax_indices = []
        for orow in range(out_h):
            for ocol in range(out_w):
                best = None
                best_j = None
                for kr in range(K):
                    r = orow * S + kr * D - P
                    for kc in range(K):
                        c = ocol * S + kc * D - P
                        if 0 <= r < H and 0 <= c < W:
                            j = r * W + c
                            if best is None or data[j] > best:
                                best = data[j]
                                best_j = j
                if best_j is None:
                    raise ValueError("max_pool2d window has no valid positions")
                out_data.append(best)
                argmax_indices.append(best_j)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        # Snapshot the input length and winning indices with the graph;
        # mutating the parent's data after the forward pass cannot change
        # what a pending backward pass uses.
        parent = self
        n = H * W

        def backward_fn(grad):
            # dx[j] += grad[o] for the winning tap of each output,
            # accumulated from 0.0 in ascending output order; a non-finite
            # partial sum aborts the whole pass before any grad is written.
            dx = [0.0] * n
            for o in range(out_h * out_w):
                j = argmax_indices[o]
                dx[j] = dx[j] + grad[o]
                if not math.isfinite(dx[j]):
                    raise ValueError(
                        "max_pool2d backward intermediate must be finite"
                    )
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def avg_pool2d(
        self, height, width, kernel_size, stride=None, padding=0, dilation=1
    ):
        data = _require_nonempty_float_vector(self, "avg_pool2d")
        # height, width, kernel_size, a non-None stride and dilation must be
        # non-bool positive ints; stride=None falls back to kernel_size.
        # padding must be a non-bool non-negative int.
        for name, value in (
            ("height", height),
            ("width", width),
            ("kernel_size", kernel_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be a positive int")
            if value <= 0:
                raise ValueError(name + " must be a positive int")
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, bool) or not isinstance(stride, int):
            raise TypeError("stride must be a positive int")
        elif stride <= 0:
            raise ValueError("stride must be a positive int")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be a non-negative int")
        if padding < 0:
            raise ValueError("padding must be a non-negative int")
        if isinstance(dilation, bool) or not isinstance(dilation, int):
            raise TypeError("dilation must be a positive int")
        if dilation <= 0:
            raise ValueError("dilation must be a positive int")
        H, W, K, S, P, D = height, width, kernel_size, stride, padding, dilation
        # self stores a single-channel H x W image in row-major order.
        if len(data) != H * W:
            raise ValueError(
                "avg_pool2d input length must equal height * width"
            )
        # Effective window span: taps in each axis sit dilation positions
        # apart, so a window covers E = D*(K-1)+1 positions.
        E = D * (K - 1) + 1
        out_h = (H + 2 * P - E) // S + 1
        out_w = (W + 2 * P - E) // S + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("avg_pool2d output dimensions must be positive")
        # Each window averages over a fixed denominator of K*K taps; the
        # taps sit at (or*S+kr*D-P, oc*S+kc*D-P) in ascending
        # (or, oc, kr, kc) order, and taps landing outside the image (from
        # padding) count as 0.0. Only valid positions are accumulated, from
        # 0.0, then the sum is divided by K*K; a fully out-of-range window
        # yields 0.0. A non-finite partial sum or quotient aborts before a
        # result tensor exists, so no state can change on failure.
        out_data = []
        for orow in range(out_h):
            for ocol in range(out_w):
                acc = 0.0
                for kr in range(K):
                    r = orow * S + kr * D - P
                    for kc in range(K):
                        c = ocol * S + kc * D - P
                        if 0 <= r < H and 0 <= c < W:
                            acc += data[r * W + c]
                            if not math.isfinite(acc):
                                raise ValueError(
                                    "avg_pool2d intermediate must be finite"
                                )
                mean = acc / (K * K)
                if not math.isfinite(mean):
                    raise ValueError("avg_pool2d intermediate must be finite")
                out_data.append(mean)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        # Snapshot the image shape and window parameters with the graph;
        # mutating the parent's data after the forward pass cannot change
        # what a pending backward pass uses.
        parent = self
        n = H * W
        snapshot_H = H
        snapshot_W = W
        snapshot_K = K
        snapshot_S = S
        snapshot_P = P
        snapshot_D = D

        def backward_fn(grad):
            # dx[(or*S+kr*D-P)*W+(oc*S+kc*D-P)] += grad[or*OW+oc] / (K*K)
            # for every in-range tap, accumulated in the same ascending
            # (or, oc, kr, kc) order as the forward pass; a non-finite
            # division or partial sum aborts the whole pass before any
            # grad is written.
            dx = [0.0] * n
            for orow in range(out_h):
                for ocol in range(out_w):
                    g = grad[orow * out_w + ocol]
                    for kr in range(snapshot_K):
                        r = orow * snapshot_S + kr * snapshot_D - snapshot_P
                        for kc in range(snapshot_K):
                            c = ocol * snapshot_S + kc * snapshot_D - snapshot_P
                            if 0 <= r < snapshot_H and 0 <= c < snapshot_W:
                                share = g / (snapshot_K * snapshot_K)
                                if not math.isfinite(share):
                                    raise ValueError(
                                        "avg_pool2d backward intermediate must be"
                                        " finite"
                                    )
                                j = r * snapshot_W + c
                                dx[j] += share
                                if not math.isfinite(dx[j]):
                                    raise ValueError(
                                        "avg_pool2d backward intermediate must be"
                                        " finite"
                                    )
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def dropout(self, p=0.5, seed=0):
        data = _require_nonempty_float_vector(self, "dropout")
        if isinstance(p, bool) or not isinstance(p, float):
            raise TypeError("p must be a finite float in [0.0, 1.0)")
        if not math.isfinite(p) or p < 0.0 or p >= 1.0:
            raise ValueError("p must be a finite float in [0.0, 1.0)")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer between 0 and 4294967295")
        if seed < 0 or seed > 4294967295:
            raise ValueError("seed must be an integer between 0 and 4294967295")
        # The mask comes from a local linear-congruential stream seeded by
        # the caller; no global random state is ever consulted.
        scale = 1.0 - p
        mask = []
        out_data = []
        s = seed
        for value in data:
            s = (1664525 * s + 1013904223) % 4294967296
            keep = s / 4294967296.0 >= p
            mask.append(keep)
            if keep:
                scaled = value / scale
                if not math.isfinite(scaled):
                    raise ValueError("dropout result must be finite")
                out_data.append(scaled)
            else:
                out_data.append(0.0)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self

        def backward_fn(grad):
            contribution = []
            for g, keep in zip(grad, mask):
                if keep:
                    scaled_grad = g / scale
                    if not math.isfinite(scaled_grad):
                        raise ValueError(
                            "dropout backward intermediate must be finite"
                        )
                    contribution.append(scaled_grad)
                else:
                    contribution.append(0.0)
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def layer_norm(self, eps=1e-5):
        data = _require_nonempty_float_vector(self, "layer_norm")
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        n = len(data)
        mu = sum(data) / n
        if not math.isfinite(mu):
            raise ValueError("layer_norm intermediate must be finite")
        centered = []
        for value in data:
            c_i = value - mu
            if not math.isfinite(c_i):
                raise ValueError("layer_norm intermediate must be finite")
            centered.append(c_i)
        var_sum = 0.0
        for c_i in centered:
            var_sum += c_i * c_i
            if not math.isfinite(var_sum):
                raise ValueError("layer_norm intermediate must be finite")
        var = var_sum / n
        if not math.isfinite(var):
            raise ValueError("layer_norm intermediate must be finite")
        denom = var + eps
        if not math.isfinite(denom):
            raise ValueError("layer_norm intermediate must be finite")
        r = 1.0 / math.sqrt(denom)
        if not math.isfinite(r):
            raise ValueError("layer_norm intermediate must be finite")
        out_data = []
        for c_i in centered:
            y_i = c_i * r
            if not math.isfinite(y_i):
                raise ValueError("layer_norm intermediate must be finite")
            out_data.append(y_i)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self

        def backward_fn(grad):
            G = 0.0
            for g in grad:
                G += g
                if not math.isfinite(G):
                    raise ValueError(
                        "layer_norm backward intermediate must be finite"
                    )
            H = 0.0
            for g, c_i in zip(grad, centered):
                product = g * c_i
                if not math.isfinite(product):
                    raise ValueError(
                        "layer_norm backward intermediate must be finite"
                    )
                H += product
                if not math.isfinite(H):
                    raise ValueError(
                        "layer_norm backward intermediate must be finite"
                    )
            contribution = []
            for g, c_i in zip(grad, centered):
                dx_i = (r / n) * (n * g - G - c_i * r * r * H)
                if not math.isfinite(dx_i):
                    raise ValueError(
                        "layer_norm backward intermediate must be finite"
                    )
                contribution.append(dx_i)
            return [(parent, contribution)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def layer_norm_affine(self, weight, bias, eps=1e-5):
        if not isinstance(weight, Tensor):
            raise TypeError("weight must be a Tensor")
        if not isinstance(bias, Tensor):
            raise TypeError("bias must be a Tensor")
        # All three data lists may have been mutated after construction, so
        # re-validate at call time: a non-list/empty list is a ValueError,
        # a non-float element or a non-bool requires_grad is a TypeError,
        # a non-finite element is a ValueError.
        x_data = _require_nonempty_float_vector(
            self, "layer_norm_affine"
        )
        w_data = _require_nonempty_float_vector(
            weight, "layer_norm_affine"
        )
        b_data = _require_nonempty_float_vector(
            bias, "layer_norm_affine"
        )
        n = len(x_data)
        if len(w_data) != n or len(b_data) != n:
            raise ValueError(
                "layer_norm_affine requires equal-length 1D float lists"
            )
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        # Snapshot all three inputs so later caller-side mutation or
        # replacement can change neither the forward result nor a pending
        # backward pass.
        x = list(x_data)
        w = list(w_data)
        b = list(b_data)
        # Accumulate from 0.0 in ascending index order: mu = sum(x)/n,
        # c_i = x_i - mu, v = sum(c_i*c_i)/n, r = 1/sqrt(v+eps),
        # h_i = c_i*r, y_i = w_i*h_i + b_i. A non-finite intermediate
        # aborts before a result tensor exists, so no state can change on
        # failure.
        total = 0.0
        for i in range(n):
            total += x[i]
            if not math.isfinite(total):
                raise ValueError(
                    "layer_norm_affine intermediate must be finite"
                )
        mu = total / n
        if not math.isfinite(mu):
            raise ValueError("layer_norm_affine intermediate must be finite")
        centered = []
        for i in range(n):
            c_i = x[i] - mu
            if not math.isfinite(c_i):
                raise ValueError(
                    "layer_norm_affine intermediate must be finite"
                )
            centered.append(c_i)
        var_sum = 0.0
        for i in range(n):
            square = centered[i] * centered[i]
            if not math.isfinite(square):
                raise ValueError(
                    "layer_norm_affine intermediate must be finite"
                )
            var_sum += square
            if not math.isfinite(var_sum):
                raise ValueError(
                    "layer_norm_affine intermediate must be finite"
                )
        var = var_sum / n
        if not math.isfinite(var):
            raise ValueError("layer_norm_affine intermediate must be finite")
        denom = var + eps
        if not math.isfinite(denom):
            raise ValueError("layer_norm_affine intermediate must be finite")
        r = 1.0 / math.sqrt(denom)
        if not math.isfinite(r):
            raise ValueError("layer_norm_affine intermediate must be finite")
        normalized = []
        for i in range(n):
            h_i = centered[i] * r
            if not math.isfinite(h_i):
                raise ValueError(
                    "layer_norm_affine intermediate must be finite"
                )
            normalized.append(h_i)
        out_data = []
        for i in range(n):
            scaled = w[i] * normalized[i]
            if not math.isfinite(scaled):
                raise ValueError(
                    "layer_norm_affine intermediate must be finite"
                )
            y_i = scaled + b[i]
            if not math.isfinite(y_i):
                raise ValueError(
                    "layer_norm_affine intermediate must be finite"
                )
            out_data.append(y_i)
        if not (
            self.requires_grad
            or weight.requires_grad
            or bias.requires_grad
        ):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_weight, parent_bias = self, weight, bias
        # Save w, r and the normalized values with the graph so a pending
        # backward pass is independent of any subsequent data mutation.
        saved_w = w
        saved_r = r
        saved_h = normalized

        def backward_fn(grad):
            # u_i = g_i*w_i, U = sum(u_i), H = sum(u_i*h_i), each
            # accumulated from 0.0 in ascending index order; dx_i =
            # (r/n)*(n*u_i - U - h_i*H), dw_i = g_i*h_i, db_i = g_i. Only
            # the roles that require grad are computed and submitted, and a
            # single Tensor playing several roles receives one elementwise
            # merged contribution. Any non-finite intermediate aborts the
            # whole pass before a grad is written.
            role_contribs = []
            if parent_self.requires_grad:
                u = []
                for i in range(n):
                    u_i = grad[i] * saved_w[i]
                    if not math.isfinite(u_i):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                    u.append(u_i)
                U = 0.0
                for i in range(n):
                    U += u[i]
                    if not math.isfinite(U):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                H = 0.0
                for i in range(n):
                    product = u[i] * saved_h[i]
                    if not math.isfinite(product):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                    H += product
                    if not math.isfinite(H):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                scale = saved_r / n
                if not math.isfinite(scale):
                    raise ValueError(
                        "layer_norm_affine backward intermediate must be"
                        " finite"
                    )
                dx = []
                for i in range(n):
                    scaled = n * u[i]
                    if not math.isfinite(scaled):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                    shifted = scaled - U
                    if not math.isfinite(shifted):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                    correction = saved_h[i] * H
                    if not math.isfinite(correction):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                    bracket = shifted - correction
                    if not math.isfinite(bracket):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                    dx_i = scale * bracket
                    if not math.isfinite(dx_i):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                    dx.append(dx_i)
                role_contribs.append((parent_self, dx))
            if parent_weight.requires_grad:
                dw = []
                for i in range(n):
                    dw_i = grad[i] * saved_h[i]
                    if not math.isfinite(dw_i):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
                    dw.append(dw_i)
                role_contribs.append((parent_weight, dw))
            if parent_bias.requires_grad:
                role_contribs.append((parent_bias, list(grad)))
            # Merge roles that the same Tensor object plays, element by
            # element in index order; a non-finite partial sum aborts.
            merged = {}
            order = []
            for parent, contribution in role_contribs:
                key = id(parent)
                if key not in merged:
                    merged[key] = list(contribution)
                    order.append(parent)
                    continue
                current = merged[key]
                for i in range(n):
                    current[i] += contribution[i]
                    if not math.isfinite(current[i]):
                        raise ValueError(
                            "layer_norm_affine backward intermediate must"
                            " be finite"
                        )
            return [(parent, merged[id(parent)]) for parent in order]

        return Tensor._make(
            out_data,
            True,
            (parent_self, parent_weight, parent_bias),
            backward_fn,
        )

    def norm(self, p=2.0):
        data = _require_nonempty_float_vector(self, "norm")
        if isinstance(p, bool) or not isinstance(p, float):
            raise TypeError("p must be a finite positive float")
        if not math.isfinite(p) or p <= 0.0:
            raise ValueError("p must be a finite positive float")
        # Snapshot the input so later caller-side mutation or replacement
        # can change neither the forward result nor a pending backward.
        x = list(data)
        # a_i = |x_i| in ascending index order and S = sum a_i**p
        # accumulated from 0.0, then n = S**(1.0/p); a non-finite power,
        # partial sum, reciprocal or result aborts before a result tensor
        # exists, so no state can change on failure. 0.0 raised to a
        # negative exponent raises ZeroDivisionError, which maps to
        # ValueError just like an overflow.
        S = 0.0
        for i in range(len(x)):
            a_i = abs(x[i])
            try:
                power = a_i ** p
            except (OverflowError, ZeroDivisionError):
                raise ValueError("norm power result must be finite")
            if not math.isfinite(power):
                raise ValueError("norm power result must be finite")
            S += power
            if not math.isfinite(S):
                raise ValueError("norm partial sum must be finite")
        inv_p = 1.0 / p
        if not math.isfinite(inv_p):
            raise ValueError("norm reciprocal must be finite")
        try:
            n = S ** inv_p
        except OverflowError:
            raise ValueError("norm result must be finite")
        if not math.isfinite(n):
            raise ValueError("norm result must be finite")
        if not self.requires_grad:
            return Tensor._make(n, False, (), None)
        parent = self
        # Save the snapshot x, p and n with the graph so a pending backward
        # pass is independent of any subsequent data mutation.
        saved_x = x
        saved_p = p
        saved_n = n

        def backward_fn(grad):
            # d_i = sign(x_i) * |x_i|**(p-1) / n**(p-1); n == 0 with p < 1
            # has no finite derivative, and an individual x_i == 0 with
            # p < 1 is non-differentiable even when n > 0. With p >= 1 the
            # derivative at a zero entry is 0. Each intermediate is checked
            # in turn, and the generic engine validates the contribution
            # itself and every merge into an existing grad.
            if saved_n == 0.0 and saved_p < 1.0:
                raise ValueError(
                    "norm is not differentiable at zero for p < 1"
                )
            exponent = saved_p - 1.0
            if not math.isfinite(exponent):
                raise ValueError(
                    "norm backward intermediate must be finite"
                )
            denom = _finite_power(
                saved_n,
                exponent,
                "norm backward intermediate must be finite",
            )
            contribution = []
            for i in range(len(saved_x)):
                x_i = saved_x[i]
                if x_i == 0.0:
                    if saved_p >= 1.0:
                        d_i = 0.0
                    else:
                        raise ValueError(
                            "norm is not differentiable at zero for p < 1"
                        )
                else:
                    a_i = abs(x_i)
                    power = _finite_power(
                        a_i,
                        exponent,
                        "norm backward intermediate must be finite",
                    )
                    sign = 1.0 if x_i > 0.0 else -1.0
                    numerator = sign * power
                    if not math.isfinite(numerator):
                        raise ValueError(
                            "norm backward intermediate must be finite"
                        )
                    d_i = numerator / denom
                    if not math.isfinite(d_i):
                        raise ValueError(
                            "norm backward intermediate must be finite"
                        )
                value = grad * d_i
                if not math.isfinite(value):
                    raise ValueError(
                        "norm backward intermediate must be finite"
                    )
                contribution.append(value)
            return [(parent, contribution)]

        return Tensor._make(n, True, (parent,), backward_fn)

    def l2_normalize(self, eps=1e-12):
        data = _require_nonempty_float_vector(self, "l2_normalize")
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        # Snapshot the input so later caller-side mutation or replacement
        # can change neither the forward result nor a pending backward.
        x = list(data)
        # Accumulate the squared norm from 0.0 in ascending index order;
        # a non-finite square or partial sum aborts before a result tensor
        # exists, so no state can change on failure.
        A = 0.0
        for i in range(len(x)):
            square = x[i] * x[i]
            if not math.isfinite(square):
                raise ValueError("l2_normalize intermediate must be finite")
            A += square
            if not math.isfinite(A):
                raise ValueError("l2_normalize intermediate must be finite")
        total = A + eps
        if not math.isfinite(total):
            raise ValueError("l2_normalize intermediate must be finite")
        root = math.sqrt(total)
        if not math.isfinite(root):
            raise ValueError("l2_normalize intermediate must be finite")
        r = 1.0 / root
        if not math.isfinite(r):
            raise ValueError("l2_normalize intermediate must be finite")
        y = []
        for i in range(len(x)):
            y_i = x[i] * r
            if not math.isfinite(y_i):
                raise ValueError("l2_normalize intermediate must be finite")
            y.append(y_i)
        if not self.requires_grad:
            return Tensor._make(y, False, (), None)
        parent = self
        # Save only the forward outputs r and y; both are freshly computed
        # values, so mutating the input afterwards cannot change a pending
        # backward pass.
        saved_r = r
        saved_y = y

        def backward_fn(grad):
            # C = sum_i grad_i * y_i accumulated from 0.0 in ascending
            # index order; each intermediate is checked as it is produced.
            C = 0.0
            for i in range(len(saved_y)):
                product = grad[i] * saved_y[i]
                if not math.isfinite(product):
                    raise ValueError(
                        "l2_normalize backward intermediate must be finite"
                    )
                C += product
                if not math.isfinite(C):
                    raise ValueError(
                        "l2_normalize backward intermediate must be finite"
                    )
            # dx_i = r * (grad_i - y_i * C): each term is checked in turn,
            # and the generic engine validates the contribution itself and
            # every merge into an existing grad.
            contribution = []
            for i in range(len(saved_y)):
                projection = saved_y[i] * C
                if not math.isfinite(projection):
                    raise ValueError(
                        "l2_normalize backward intermediate must be finite"
                    )
                diff = grad[i] - projection
                if not math.isfinite(diff):
                    raise ValueError(
                        "l2_normalize backward intermediate must be finite"
                    )
                dx_i = saved_r * diff
                if not math.isfinite(dx_i):
                    raise ValueError(
                        "l2_normalize backward intermediate must be finite"
                    )
                contribution.append(dx_i)
            return [(parent, contribution)]

        return Tensor._make(y, True, (parent,), backward_fn)

    def group_norm(self, groups, eps=1e-5):
        data = _require_nonempty_float_vector(self, "group_norm")
        if isinstance(groups, bool) or not isinstance(groups, int):
            raise TypeError("groups must be a positive int")
        if groups <= 0:
            raise ValueError("groups must be a positive int")
        n = len(data)
        if n % groups != 0:
            raise ValueError("groups must divide the input length")
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        # Snapshot the input so later caller-side mutation or replacement
        # can change neither the forward result nor a pending backward.
        x = list(data)
        m = n // groups
        # Normalize each equal-length group independently, accumulating
        # from 0.0 in ascending index order: mu = sum(x)/m, c_i = x_i - mu,
        # v = sum(c_i*c_i)/m, r = 1/sqrt(v+eps), y_i = c_i*r. A non-finite
        # intermediate aborts before a result tensor exists, so no state
        # can change on failure.
        means = []
        scales = []
        out_data = []
        for g in range(groups):
            base = g * m
            total = 0.0
            for i in range(base, base + m):
                total += x[i]
                if not math.isfinite(total):
                    raise ValueError(
                        "group_norm intermediate must be finite"
                    )
            mu = total / m
            if not math.isfinite(mu):
                raise ValueError("group_norm intermediate must be finite")
            centered = []
            for i in range(base, base + m):
                c_i = x[i] - mu
                if not math.isfinite(c_i):
                    raise ValueError(
                        "group_norm intermediate must be finite"
                    )
                centered.append(c_i)
            var_sum = 0.0
            for k in range(m):
                square = centered[k] * centered[k]
                if not math.isfinite(square):
                    raise ValueError(
                        "group_norm intermediate must be finite"
                    )
                var_sum += square
                if not math.isfinite(var_sum):
                    raise ValueError(
                        "group_norm intermediate must be finite"
                    )
            var = var_sum / m
            if not math.isfinite(var):
                raise ValueError("group_norm intermediate must be finite")
            denom = var + eps
            if not math.isfinite(denom):
                raise ValueError("group_norm intermediate must be finite")
            r = 1.0 / math.sqrt(denom)
            if not math.isfinite(r):
                raise ValueError("group_norm intermediate must be finite")
            for k in range(m):
                y_i = centered[k] * r
                if not math.isfinite(y_i):
                    raise ValueError(
                        "group_norm intermediate must be finite"
                    )
                out_data.append(y_i)
            means.append(mu)
            scales.append(r)
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self
        # Save x, groups, the per-group means and the per-group scales with
        # the graph so a pending backward pass is independent of any
        # subsequent data mutation.
        saved_x = x
        saved_groups = groups
        saved_means = means
        saved_scales = scales

        def backward_fn(grad):
            # Per group, first accumulate G = sum(g_i) and H = sum(g_i*c_i)
            # from 0.0 in ascending index order, then dx_i =
            # (r/m)*(m*g_i - G - c_i*r*r*H) in ascending index order. Only
            # self is submitted; a non-finite intermediate aborts the whole
            # pass before any grad is written.
            dx = []
            for g in range(saved_groups):
                base = g * m
                mu = saved_means[g]
                r = saved_scales[g]
                G = 0.0
                for i in range(base, base + m):
                    G += grad[i]
                    if not math.isfinite(G):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                centered = []
                H = 0.0
                for i in range(base, base + m):
                    c_i = saved_x[i] - mu
                    if not math.isfinite(c_i):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    centered.append(c_i)
                    product = grad[i] * c_i
                    if not math.isfinite(product):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    H += product
                    if not math.isfinite(H):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                scale = r / m
                if not math.isfinite(scale):
                    raise ValueError(
                        "group_norm backward intermediate must be finite"
                    )
                for k in range(m):
                    scaled = m * grad[base + k]
                    if not math.isfinite(scaled):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    shifted = scaled - G
                    if not math.isfinite(shifted):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    correction = centered[k] * r
                    if not math.isfinite(correction):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    correction = correction * r
                    if not math.isfinite(correction):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    correction = correction * H
                    if not math.isfinite(correction):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    bracket = shifted - correction
                    if not math.isfinite(bracket):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    dx_i = scale * bracket
                    if not math.isfinite(dx_i):
                        raise ValueError(
                            "group_norm backward intermediate must be finite"
                        )
                    dx.append(dx_i)
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def batch_norm(self, weight, bias, eps=1e-5):
        if not isinstance(weight, Tensor):
            raise TypeError("weight must be a Tensor")
        if not isinstance(bias, Tensor):
            raise TypeError("bias must be a Tensor")
        # weight and bias must each hold a finite float scalar; the data
        # may have been mutated after construction, so re-validate at call
        # time. A list, bool, int or anything else is a TypeError; a
        # non-finite float is a ValueError.
        w = weight.data
        if isinstance(w, bool) or not isinstance(w, float):
            raise TypeError("weight data must be a finite float scalar")
        if not math.isfinite(w):
            raise ValueError("weight data must be finite")
        b = bias.data
        if isinstance(b, bool) or not isinstance(b, float):
            raise TypeError("bias data must be a finite float scalar")
        if not math.isfinite(b):
            raise ValueError("bias data must be finite")
        data = _require_nonempty_float_vector(self, "batch_norm")
        if not isinstance(weight.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(bias.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        # Snapshot the input and both parameters so later caller-side
        # mutation or replacement can change neither the forward result
        # nor a pending backward pass.
        x = list(data)
        n = len(x)
        # Accumulate from 0.0 in ascending index order: mu = sum(x)/n,
        # c_i = x_i - mu, v = sum(c_i*c_i)/n, r = 1/sqrt(v+eps),
        # h_i = c_i*r, y_i = w*h_i + b. A non-finite intermediate aborts
        # before a result tensor exists, so no state can change on
        # failure.
        total = 0.0
        for i in range(n):
            total += x[i]
            if not math.isfinite(total):
                raise ValueError("batch_norm intermediate must be finite")
        mu = total / n
        if not math.isfinite(mu):
            raise ValueError("batch_norm intermediate must be finite")
        centered = []
        for i in range(n):
            c_i = x[i] - mu
            if not math.isfinite(c_i):
                raise ValueError("batch_norm intermediate must be finite")
            centered.append(c_i)
        var_sum = 0.0
        for c_i in centered:
            var_sum += c_i * c_i
            if not math.isfinite(var_sum):
                raise ValueError("batch_norm intermediate must be finite")
        var = var_sum / n
        if not math.isfinite(var):
            raise ValueError("batch_norm intermediate must be finite")
        denom = var + eps
        if not math.isfinite(denom):
            raise ValueError("batch_norm intermediate must be finite")
        r = 1.0 / math.sqrt(denom)
        if not math.isfinite(r):
            raise ValueError("batch_norm intermediate must be finite")
        normalized = []
        for c_i in centered:
            h_i = c_i * r
            if not math.isfinite(h_i):
                raise ValueError("batch_norm intermediate must be finite")
            normalized.append(h_i)
        out_data = []
        for h_i in normalized:
            product = w * h_i
            if not math.isfinite(product):
                raise ValueError("batch_norm intermediate must be finite")
            y_i = product + b
            if not math.isfinite(y_i):
                raise ValueError("batch_norm intermediate must be finite")
            out_data.append(y_i)
        if not (
            self.requires_grad
            or weight.requires_grad
            or bias.requires_grad
        ):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_weight, parent_bias = self, weight, bias
        # Save w, r and the normalized values with the graph so a pending
        # backward pass is independent of any subsequent data mutation.
        saved_w = w
        saved_r = r
        saved_h = normalized

        def backward_fn(grad):
            # G = sum(g) and H = sum(g_i*h_i), each accumulated from 0.0
            # in ascending index order; dx_i = w*r*(n*g_i - G - h_i*H)/n,
            # dw = H, db = G. Only the sides that require grad are
            # computed and submitted; a non-finite intermediate aborts the
            # whole pass before any grad is written.
            G = 0.0
            for g in grad:
                G += g
                if not math.isfinite(G):
                    raise ValueError(
                        "batch_norm backward intermediate must be finite"
                    )
            H = 0.0
            for i in range(n):
                product = grad[i] * saved_h[i]
                if not math.isfinite(product):
                    raise ValueError(
                        "batch_norm backward intermediate must be finite"
                    )
                H += product
                if not math.isfinite(H):
                    raise ValueError(
                        "batch_norm backward intermediate must be finite"
                    )
            contributions = []
            if parent_self.requires_grad:
                wr = saved_w * saved_r
                if not math.isfinite(wr):
                    raise ValueError(
                        "batch_norm backward intermediate must be finite"
                    )
                dx = []
                for i in range(n):
                    scaled = n * grad[i]
                    if not math.isfinite(scaled):
                        raise ValueError(
                            "batch_norm backward intermediate must be"
                            " finite"
                        )
                    shifted = scaled - G
                    if not math.isfinite(shifted):
                        raise ValueError(
                            "batch_norm backward intermediate must be"
                            " finite"
                        )
                    correction = saved_h[i] * H
                    if not math.isfinite(correction):
                        raise ValueError(
                            "batch_norm backward intermediate must be"
                            " finite"
                        )
                    bracket = shifted - correction
                    if not math.isfinite(bracket):
                        raise ValueError(
                            "batch_norm backward intermediate must be"
                            " finite"
                        )
                    product = wr * bracket
                    if not math.isfinite(product):
                        raise ValueError(
                            "batch_norm backward intermediate must be"
                            " finite"
                        )
                    dx_i = product / n
                    if not math.isfinite(dx_i):
                        raise ValueError(
                            "batch_norm backward intermediate must be"
                            " finite"
                        )
                    dx.append(dx_i)
                contributions.append((parent_self, dx))
            if parent_weight is parent_bias:
                # A single object playing both roles receives one merged
                # scalar contribution dw + db.
                if parent_weight.requires_grad:
                    merged = H + G
                    if not math.isfinite(merged):
                        raise ValueError(
                            "batch_norm backward intermediate must be"
                            " finite"
                        )
                    contributions.append((parent_weight, merged))
            else:
                if parent_weight.requires_grad:
                    contributions.append((parent_weight, H))
                if parent_bias.requires_grad:
                    contributions.append((parent_bias, G))
            return contributions

        return Tensor._make(
            out_data,
            True,
            (parent_self, parent_weight, parent_bias),
            backward_fn,
        )

    def gather(self, indices):
        data = _require_nonempty_float_vector(self, "gather")
        if not isinstance(indices, list):
            raise TypeError("indices must be a list of non-bool ints")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("indices elements must be non-bool ints")
        if len(indices) == 0:
            raise ValueError("indices must be non-empty")
        n = len(data)
        for index in indices:
            if index < 0 or index >= n:
                raise ValueError("gather index out of range")
        # Snapshot the indices so later caller-side mutation of the list
        # cannot change what a pending backward pass will scatter.
        snapshot = list(indices)
        out_data = [data[index] for index in snapshot]
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self

        def backward_fn(grad):
            dx = [0.0] * n
            for o in range(len(snapshot)):
                dx[snapshot[o]] += grad[o]
                if not math.isfinite(dx[snapshot[o]]):
                    raise ValueError(
                        "gather backward intermediate must be finite"
                    )
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def topk(self, k):
        """Return the k largest entries as (values, indices).

        Values are ordered by descending value, ties by ascending original
        index. ``values`` is a Tensor holding fresh copies of the selected
        values; ``indices`` is a brand-new list[int] of original indices.
        When self requires grad, a single-parent graph is built with private
        snapshots of the original length and the index list, so later
        mutation of self.data (or the returned indices) cannot change a
        pending backward pass.
        """
        data = _require_nonempty_float_vector(self, "topk")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be a non-bool int")
        n = len(data)
        if k < 1 or k > n:
            raise ValueError("k must satisfy 1 <= k <= len(data)")
        # Sort by descending value, then ascending original index for ties.
        # All data is finite at this point, so negation cannot produce NaN.
        order = sorted(range(n), key=lambda i: (-data[i], i))[:k]
        out_data = [data[i] for i in order]
        indices = list(order)
        if not self.requires_grad:
            out = Tensor._make(out_data, False, (), None)
            return out, indices
        parent = self
        length = n
        snapshot = list(order)

        def backward_fn(grad):
            dx = [0.0] * length
            for o in range(k):
                dx[snapshot[o]] += grad[o]
                if not math.isfinite(dx[snapshot[o]]):
                    raise ValueError(
                        "topk backward intermediate must be finite"
                    )
            return [(parent, dx)]

        out = Tensor._make(out_data, True, (parent,), backward_fn)
        return out, indices

    def embedding(self, indices, dim):
        data = _require_nonempty_float_vector(self, "embedding")
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("dim must be a positive int")
        if dim <= 0:
            raise ValueError("dim must be a positive int")
        # data stores the embedding table as len(data) // dim rows of dim
        # values each, laid out row by row.
        if len(data) % dim != 0:
            raise ValueError("embedding data length must be a multiple of dim")
        if not isinstance(indices, list):
            raise TypeError("indices must be a list of non-bool ints")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("indices elements must be non-bool ints")
        if len(indices) == 0:
            raise ValueError("indices must be non-empty")
        rows = len(data) // dim
        for index in indices:
            if index < 0 or index >= rows:
                raise ValueError("embedding index out of range")
        # Snapshot the indices, dim and the original data length so later
        # caller-side mutation of the indices list or of self.data cannot
        # change what a pending backward pass will scatter.
        snapshot = list(indices)
        length = len(data)
        # Concatenate the dim values of row indices[o] for each output row
        # o in ascending (o, d) order.
        out_data = []
        for index in snapshot:
            base = index * dim
            for d in range(dim):
                out_data.append(data[base + d])
        if not self.requires_grad:
            return Tensor._make(out_data, False, (), None)
        parent = self

        def backward_fn(grad):
            # dx[indices[o]*dim+d] += grad[o*dim+d], accumulated from 0.0
            # in ascending (o, d) order; repeated indices accumulate. A
            # non-finite partial sum aborts the whole pass before any grad
            # is written.
            dx = [0.0] * length
            for o in range(len(snapshot)):
                base = snapshot[o] * dim
                for d in range(dim):
                    dx[base + d] += grad[o * dim + d]
                    if not math.isfinite(dx[base + d]):
                        raise ValueError(
                            "embedding backward intermediate must be finite"
                        )
            return [(parent, dx)]

        return Tensor._make(out_data, True, (parent,), backward_fn)

    def scatter_add(self, indices, source):
        """Copy self and accumulate source values at indices, in o order.

        out starts as a fresh copy of self.data; for each o in ascending
        order, out[indices[o]] += source.data[o] (a scalar source is
        broadcast to every o). Repeated indices accumulate. TypeError on
        bad container/element types, ValueError on empty/non-finite/out of
        range inputs or non-finite results. When either parent requires
        grad, a two-parent graph is built with private snapshots of the
        indices and both shapes, so later caller-side mutation cannot
        change a pending backward pass.
        """
        if not isinstance(source, Tensor):
            raise TypeError("source must be a Tensor")
        data = self.data
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(
                "scatter_add requires a non-empty 1D float list"
            )
        src = source.data
        src_vector = isinstance(src, list)
        if isinstance(src, bool) or (
            not isinstance(src, float) and not src_vector
        ):
            raise TypeError(
                "source data must be a finite float scalar or a non-empty"
                " 1D float list"
            )
        if src_vector and len(src) == 0:
            raise ValueError("source data list must be non-empty")
        for value in data:
            if isinstance(value, bool) or not isinstance(value, float):
                raise TypeError("scatter_add data elements must be floats")
        src_values = src if src_vector else [src]
        for value in src_values:
            if isinstance(value, bool) or not isinstance(value, float):
                raise TypeError("scatter_add source elements must be floats")
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        if not isinstance(source.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        for value in data:
            if not math.isfinite(value):
                raise ValueError("scatter_add data elements must be finite")
        for value in src_values:
            if not math.isfinite(value):
                raise ValueError("scatter_add source elements must be finite")
        if not isinstance(indices, list):
            raise TypeError("indices must be a list of non-bool ints")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("indices elements must be non-bool ints")
        if len(indices) == 0:
            raise ValueError("indices must be non-empty")
        n = len(data)
        for index in indices:
            if index < 0 or index >= n:
                raise ValueError("scatter_add index out of range")
        if src_vector and len(src) != len(indices):
            raise ValueError("source length must match indices length")
        # Snapshot the indices and the self length so later caller-side
        # mutation of the indices list or of either data cannot change a
        # pending backward pass.
        snapshot = list(indices)
        out_data = list(data)
        for o in range(len(snapshot)):
            value = src[o] if src_vector else src
            out_data[snapshot[o]] += value
            if not math.isfinite(out_data[snapshot[o]]):
                raise ValueError("scatter_add result must be finite")
        if not (self.requires_grad or source.requires_grad):
            return Tensor._make(out_data, False, (), None)
        parent_self, parent_source = self, source
        length = n
        source_was_vector = src_vector

        def backward_fn(grad):
            contributions = []
            if parent_self.requires_grad:
                # The base passes through unchanged: a fresh copy of grad.
                contributions.append((parent_self, list(grad)))
            if parent_source.requires_grad:
                if source_was_vector:
                    ds = []
                    for o in range(len(snapshot)):
                        value = grad[snapshot[o]]
                        if not math.isfinite(value):
                            raise ValueError(
                                "scatter_add backward intermediate"
                                " must be finite"
                            )
                        ds.append(value)
                    contributions.append((parent_source, ds))
                else:
                    # A broadcast scalar source reduces by accumulating
                    # from 0.0 in ascending o order.
                    total = 0.0
                    for o in range(len(snapshot)):
                        total += grad[snapshot[o]]
                        if not math.isfinite(total):
                            raise ValueError(
                                "scatter_add backward intermediate"
                                " must be finite"
                            )
                    contributions.append((parent_source, total))
            return contributions

        return Tensor._make(
            out_data, True, (parent_self, parent_source), backward_fn
        )

    def zero_grad(self):
        self.grad = None
        return None

    def backward(self, grad=_MISSING):
        if not self._parents:
            raise ValueError("cannot call backward on a tensor without a graph")
        if grad is _MISSING:
            if isinstance(self.data, list):
                raise ValueError(
                    "grad must be provided for a non-scalar tensor"
                )
            grad = 1.0
        elif grad is None:
            raise TypeError("grad must not be None")
        else:
            grad = _validate_grad(grad, self.data)

        # Repeated backward on the same result accumulates the upstream
        # grad and recomputes this graph's contributions from the running
        # total, replacing this root's own previous leaf contributions
        # instead of merging a second copy into them. backward(a) followed
        # by backward(b) therefore leaves exactly the leaf grads a single
        # backward(a + b) on an identical fresh graph would leave, while
        # contributions from any other graph still merge in. The record is
        # only updated after a fully successful pass, so a failed backward
        # changes nothing.
        record = self._backward_record
        if record is not None and _same_grad_shape(record[0], grad):
            upstream = _merge_grad(record[0], grad)
            _ensure_finite_grad(upstream)
            previous = record[1]
        else:
            upstream = grad
            previous = None

        topo = []
        visited = set()

        def build(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

        build(self)

        # Phase 1: compute every gradient without mutating any node, so a
        # non-finite gradient aborts the pass with all grads untouched.
        grads = {id(self): upstream}
        for node in reversed(topo):
            if node._backward_fn is None:
                continue
            node_grad = grads.get(id(node))
            if node_grad is None:
                continue
            for parent, contribution in node._backward_fn(node_grad):
                if not parent.requires_grad:
                    continue
                _ensure_finite_grad(contribution)
                existing = grads.get(id(parent))
                if existing is None:
                    grads[id(parent)] = contribution
                else:
                    merged = _merge_grad(existing, contribution)
                    _ensure_finite_grad(merged)
                    grads[id(parent)] = merged

        # Phase 2: resolve the final grad for every node, validate, then
        # apply. Non-leaf grads are recomputed; leaf grads accumulate.
        assignments = []
        leaves = {}
        for node in topo:
            if node._parents:
                assignments.append((node, grads.get(id(node))))
            elif node.requires_grad:
                contribution = grads.get(id(node))
                if contribution is not None:
                    leaves[id(node)] = (node, contribution)
                    value = contribution
                    if node.grad is not None:
                        base = node.grad
                        if previous is not None:
                            entry = previous.get(id(node))
                            if entry is not None and _same_grad_shape(
                                entry[1], node.grad
                            ):
                                base = _sub_grad(node.grad, entry[1])
                        value = _merge_grad(base, contribution)
                    assignments.append((node, value))
        for node, value in assignments:
            if value is not None:
                _ensure_finite_grad(value)
        for node, value in assignments:
            node.grad = value
        self._backward_record = (upstream, leaves)

    def to_json(self):
        return (
            '{"data":'
            + _json_value(self.data)
            + ',"grad":'
            + ("null" if self.grad is None else _json_value(self.grad))
            + ',"requires_grad":'
            + ("true" if self.requires_grad else "false")
            + "}"
        )


def xavier_uniform(length, fan_in, fan_out, seed=0, requires_grad=True):
    """Return a Tensor of `length` Xavier-uniform samples, drawn deterministically.

    Uses limit = sqrt(6 / (fan_in + fan_out)) and a local LCG seeded by `seed`;
    no global random state is touched. TypeError on bad argument types,
    ValueError on out-of-range arguments or non-finite results.
    """
    for name, value in (
        ("length", length),
        ("fan_in", fan_in),
        ("fan_out", fan_out),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(name + " must be a positive int")
        if value <= 0:
            raise ValueError(name + " must be a positive int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int in [0, 4294967295]")
    if seed < 0 or seed > 4294967295:
        raise ValueError("seed must be in [0, 4294967295]")
    if not isinstance(requires_grad, bool):
        raise TypeError("requires_grad must be a bool")
    try:
        limit = math.sqrt(6.0 / (fan_in + fan_out))
    except OverflowError:
        raise ValueError("xavier limit must be finite")
    if not math.isfinite(limit):
        raise ValueError("xavier limit must be finite")
    data = []
    s = seed
    for _ in range(length):
        s = (1664525 * s + 1013904223) % 4294967296
        u = s / 4294967296.0
        v = 2.0 * u - 1.0
        value = v * limit
        if not (
            math.isfinite(u) and math.isfinite(v) and math.isfinite(value)
        ):
            raise ValueError("xavier_uniform values must be finite")
        data.append(value)
    return Tensor(data, requires_grad)


def kaiming_uniform(length, fan_in, seed=0, requires_grad=True):
    """Return a Tensor of `length` Kaiming-uniform samples, drawn deterministically.

    Uses bound = sqrt(6 / fan_in) and a local LCG seeded by `seed`; no
    global random state is touched. The same arguments always produce
    byte-identical data and to_json output. TypeError on bad argument
    types, ValueError on out-of-range arguments or non-finite
    intermediates.
    """
    for name, value in (("length", length), ("fan_in", fan_in)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(name + " must be a positive int")
        if value <= 0:
            raise ValueError(name + " must be a positive int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int in [0, 4294967295]")
    if seed < 0 or seed > 4294967295:
        raise ValueError("seed must be in [0, 4294967295]")
    if not isinstance(requires_grad, bool):
        raise TypeError("requires_grad must be a bool")
    try:
        bound = math.sqrt(6.0 / fan_in)
    except OverflowError:
        raise ValueError("kaiming bound must be finite")
    if not math.isfinite(bound):
        raise ValueError("kaiming bound must be finite")
    data = []
    s = seed
    for _ in range(length):
        s = (1664525 * s + 1013904223) % 4294967296
        u = s / 4294967296.0
        value = (2.0 * u - 1.0) * bound
        if not (math.isfinite(u) and math.isfinite(value)):
            raise ValueError("kaiming_uniform values must be finite")
        data.append(value)
    return Tensor(data, requires_grad)


def orthogonal(size, seed=0, gain=1.0, requires_grad=True):
    """Return a square orthogonal matrix as a row-major flattened leaf Tensor.

    A local LCG seeded by `seed` fills the matrix with uniform values in
    [-1, 1); columns are orthonormalized with modified Gram-Schmidt. No
    global random state is touched. TypeError on bad argument types,
    ValueError on out-of-range arguments, a zero-length column during
    orthogonalization, or a non-finite intermediate.
    """
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be a positive int")
    if size <= 0:
        raise ValueError("size must be a positive int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int in [0, 4294967295]")
    if seed < 0 or seed > 4294967295:
        raise ValueError("seed must be in [0, 4294967295]")
    if isinstance(gain, bool) or not isinstance(gain, float):
        raise TypeError("gain must be a positive finite float")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("gain must be a positive finite float")
    if not isinstance(requires_grad, bool):
        raise TypeError("requires_grad must be a bool")
    # Fill the square matrix A in ascending (r, c) order from a local LCG;
    # only local state exists until the result Tensor is constructed, so a
    # failure below cannot mutate any observable state.
    s = seed
    A = [[0.0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            s = (1664525 * s + 1013904223) % 4294967296
            u = s / 4294967296.0
            value = 2.0 * u - 1.0
            if not (math.isfinite(u) and math.isfinite(value)):
                raise ValueError("orthogonal intermediate must be finite")
            A[r][c] = value
    # Modified Gram-Schmidt over columns in ascending c order.
    Q = [[0.0] * size for _ in range(size)]
    for c in range(size):
        v = [A[r][c] for r in range(size)]
        for j in range(c):
            d = 0.0
            for r in range(size):
                d += Q[r][j] * v[r]
            if not math.isfinite(d):
                raise ValueError("orthogonal intermediate must be finite")
            for r in range(size):
                v[r] -= d * Q[r][j]
                if not math.isfinite(v[r]):
                    raise ValueError("orthogonal intermediate must be finite")
        h = 0.0
        for r in range(size):
            h += v[r] * v[r]
        if not math.isfinite(h):
            raise ValueError("orthogonal intermediate must be finite")
        n = math.sqrt(h)
        if n == 0.0:
            raise ValueError("orthogonal matrix must have full rank")
        if not math.isfinite(n):
            raise ValueError("orthogonal intermediate must be finite")
        for r in range(size):
            normalized = v[r] / n
            if not math.isfinite(normalized):
                raise ValueError("orthogonal intermediate must be finite")
            Q[r][c] = normalized
    data = []
    for r in range(size):
        for c in range(size):
            value = gain * Q[r][c]
            if not math.isfinite(value):
                raise ValueError("orthogonal result must be finite")
            data.append(value)
    return Tensor(data, requires_grad)


class SGD:
    """Stochastic gradient descent over a fixed list of Tensors."""

    def __init__(self, parameters, lr):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        # Compute and validate every new value before mutating anything, so
        # a failure leaves all data, grad, and requires_grad untouched.
        updates = []
        for parameter in self.parameters:
            if not parameter.requires_grad or parameter.grad is None:
                continue
            updates.append(
                (parameter, self._updated_data(parameter.data, parameter.grad))
            )
        for parameter, new_data in updates:
            parameter.data = new_data
        return None

    def _updated_data(self, data, grad):
        if isinstance(data, list):
            if isinstance(grad, list):
                if len(grad) != len(data):
                    raise ValueError("grad shape must match tensor shape")
                return [
                    self._updated_value(value, g)
                    for value, g in zip(data, grad)
                ]
            if isinstance(grad, float) and not isinstance(grad, bool):
                raise ValueError("grad shape must match tensor shape")
            raise TypeError("grad must be a list of floats")
        if isinstance(grad, list):
            raise ValueError("grad shape must match tensor shape")
        return self._updated_value(data, grad)

    def _updated_value(self, value, grad):
        if isinstance(grad, bool) or not isinstance(grad, float):
            raise TypeError("grad elements must be floats")
        if not math.isfinite(grad):
            raise ValueError("grad must be finite")
        new_value = value - self.lr * grad
        if not math.isfinite(new_value):
            raise ValueError("updated data must be finite")
        return new_value

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


def _check_sgd_state(optimizer):
    """Validate an SGD optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    if isinstance(optimizer.lr, bool) or not isinstance(optimizer.lr, float):
        raise TypeError("lr must be a positive finite float")
    if not math.isfinite(optimizer.lr) or optimizer.lr <= 0.0:
        raise ValueError("lr must be a positive finite float")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)


def dump_sgd(optimizer):
    """Serialize an SGD optimizer's parameters to a compact JSON string.

    The output contains no whitespace and no trailing newline; the only
    top-level key is parameters; entries follow the parameters list in
    order, each with the keys data then requires_grad; data keeps its
    scalar or 1-D shape; floats are written with exactly six decimals and
    negative zero as 0.000000. lr is not serialized.
    """
    if not isinstance(optimizer, SGD):
        raise TypeError("optimizer must be an SGD instance")
    _check_sgd_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    return '{"parameters":[' + ",".join(entries) + "]}"


class _SGDParser:
    """Strict parser for the exact textual form produced by dump_sgd."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_sgd format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect("]}")
        if self._pos != len(self._text):
            self._fail()
        return parameters


def load_sgd(optimizer, text):
    """Restore an SGD optimizer's parameters from a dump_sgd string.

    On full validation success, atomically overwrite every parameter's
    data and requires_grad and clear every grad; the parameters list
    identities and lr are left unchanged. On any failure no state is
    touched.
    """
    if not isinstance(optimizer, SGD):
        raise TypeError("optimizer must be an SGD instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_sgd_state(optimizer)
    parameters = _SGDParser(text).parse()
    if len(parameters) != len(optimizer.parameters):
        raise ValueError("state must contain exactly one entry per parameter")
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    return None


class Adam:
    """Adam optimizer over a fixed list of Tensors."""

    def __init__(self, parameters, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        self._check_hyperparameters(lr, beta1, beta2, eps)
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [self._zeros_like(parameter.data) for parameter in parameters]
        self.v = [self._zeros_like(parameter.data) for parameter in parameters]
        self.t = 0

    @staticmethod
    def _check_hyperparameters(lr, beta1, beta2, eps):
        """Validate lr, beta1, beta2 and eps exactly as the constructor does."""
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        for name, beta in (("beta1", beta1), ("beta2", beta2)):
            if isinstance(beta, bool) or not isinstance(beta, float):
                raise TypeError(
                    name
                    + " must be a finite float in [0.0, 1.0)"
                )
            if not math.isfinite(beta) or beta < 0.0 or beta >= 1.0:
                raise ValueError(
                    name
                    + " must be a finite float in [0.0, 1.0)"
                )
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")

    @staticmethod
    def _zeros_like(data):
        if isinstance(data, list):
            return [0.0 for _ in data]
        return 0.0

    def step(self):
        # Revalidate every parameter and validate every active grad first;
        # compute all new values before mutating anything, so a failure leaves
        # all data, m, v and t untouched.
        for index, parameter in enumerate(self.parameters):
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            data = _validate_data(parameter.data)
            moment = self.m[index]
            velocity = self.v[index]
            self._check_buffer(moment, data, "m")
            self._check_buffer(velocity, data, "v")
            if parameter.requires_grad and parameter.grad is not None:
                self._check_grad(parameter.grad, data)
        active = [
            index
            for index, parameter in enumerate(self.parameters)
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not active:
            return None
        k = self.t + 1
        bias1 = 1.0 - _finite_power(
            self.beta1, k, "bias correction must be finite"
        )
        bias2 = 1.0 - _finite_power(
            self.beta2, k, "bias correction must be finite"
        )
        if not math.isfinite(bias1) or not math.isfinite(bias2):
            raise ValueError("bias correction must be finite")
        updates = []
        for index in active:
            parameter = self.parameters[index]
            updates.append(
                (
                    index,
                    self._updated(
                        parameter.data,
                        parameter.grad,
                        self.m[index],
                        self.v[index],
                        bias1,
                        bias2,
                    ),
                )
            )
        for index, (new_data, new_m, new_v) in updates:
            self.parameters[index].data = new_data
            self.m[index] = new_m
            self.v[index] = new_v
        self.t = k
        return None

    @staticmethod
    def _check_buffer(buffer, data, name):
        """Type/shape/finiteness check for an m or v moment buffer."""
        if isinstance(data, list):
            if not isinstance(buffer, list):
                if isinstance(buffer, float) and not isinstance(buffer, bool):
                    raise ValueError(
                        name + " shape must match parameter shape"
                    )
                raise TypeError(name + " must be a list of floats")
            if len(buffer) != len(data):
                raise ValueError(name + " shape must match parameter shape")
            for value in buffer:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError(name + " elements must be floats")
            for value in buffer:
                if not math.isfinite(value):
                    raise ValueError(name + " must be finite")
        else:
            if isinstance(buffer, list):
                raise ValueError(name + " shape must match parameter shape")
            if isinstance(buffer, bool) or not isinstance(buffer, float):
                raise TypeError(name + " must be a float")
            if not math.isfinite(buffer):
                raise ValueError(name + " must be finite")

    @staticmethod
    def _check_grad(grad, data):
        """Type/shape/finiteness check for an active parameter's grad."""
        if isinstance(data, list):
            if not isinstance(grad, list):
                if isinstance(grad, float) and not isinstance(grad, bool):
                    raise ValueError("grad shape must match tensor shape")
                raise TypeError("grad must be a list of floats")
            if len(grad) != len(data):
                raise ValueError("grad shape must match tensor shape")
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("grad elements must be floats")
            for value in grad:
                if not math.isfinite(value):
                    raise ValueError("grad must be finite")
        else:
            if isinstance(grad, list):
                raise ValueError("grad shape must match tensor shape")
            if isinstance(grad, bool) or not isinstance(grad, float):
                raise TypeError("grad must be a float")
            if not math.isfinite(grad):
                raise ValueError("grad must be finite")

    def _updated(self, data, grad, moment, velocity, bias1, bias2):
        """Compute (new_data, new_m, new_v) for one active parameter."""
        one_minus_b1 = 1.0 - self.beta1
        one_minus_b2 = 1.0 - self.beta2
        if isinstance(data, list):
            new_m = []
            new_v = []
            new_data = []
            for value, g, m_value, v_value in zip(
                data, grad, moment, velocity
            ):
                m_next = self.beta1 * m_value + one_minus_b1 * g
                v_next = self.beta2 * v_value + one_minus_b2 * g * g
                if not math.isfinite(m_next) or not math.isfinite(v_next):
                    raise ValueError("adam buffers must be finite")
                m_hat = m_next / bias1
                v_hat = v_next / bias2
                if not math.isfinite(m_hat) or not math.isfinite(v_hat):
                    raise ValueError("bias-corrected moments must be finite")
                try:
                    denominator = math.sqrt(v_hat) + self.eps
                except (ValueError, OverflowError):
                    raise ValueError("update denominator must be finite")
                if not math.isfinite(denominator):
                    raise ValueError("update denominator must be finite")
                value_next = value - self.lr * m_hat / denominator
                if not math.isfinite(value_next):
                    raise ValueError("updated data must be finite")
                new_m.append(m_next)
                new_v.append(v_next)
                new_data.append(value_next)
            return new_data, new_m, new_v
        m_next = self.beta1 * moment + one_minus_b1 * grad
        v_next = self.beta2 * velocity + one_minus_b2 * grad * grad
        if not math.isfinite(m_next) or not math.isfinite(v_next):
            raise ValueError("adam buffers must be finite")
        m_hat = m_next / bias1
        v_hat = v_next / bias2
        if not math.isfinite(m_hat) or not math.isfinite(v_hat):
            raise ValueError("bias-corrected moments must be finite")
        try:
            denominator = math.sqrt(v_hat) + self.eps
        except (ValueError, OverflowError):
            raise ValueError("update denominator must be finite")
        if not math.isfinite(denominator):
            raise ValueError("update denominator must be finite")
        value_next = data - self.lr * m_hat / denominator
        if not math.isfinite(value_next):
            raise ValueError("updated data must be finite")
        return value_next, m_next, v_next

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


class AMSGrad:
    """AMSGrad optimizer over a fixed list of Tensors."""

    def __init__(self, parameters, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        self._check_hyperparameters(lr, beta1, beta2, eps)
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [self._zeros_like(parameter.data) for parameter in parameters]
        self.v = [self._zeros_like(parameter.data) for parameter in parameters]
        self.v_max = [
            self._zeros_like(parameter.data) for parameter in parameters
        ]
        self.t = 0

    @staticmethod
    def _check_hyperparameters(lr, beta1, beta2, eps):
        """Validate lr, beta1, beta2 and eps exactly as the constructor does."""
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        for name, beta in (("beta1", beta1), ("beta2", beta2)):
            if isinstance(beta, bool) or not isinstance(beta, float):
                raise TypeError(
                    name
                    + " must be a finite float in [0.0, 1.0)"
                )
            if not math.isfinite(beta) or beta < 0.0 or beta >= 1.0:
                raise ValueError(
                    name
                    + " must be a finite float in [0.0, 1.0)"
                )
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")

    @staticmethod
    def _zeros_like(data):
        if isinstance(data, list):
            return [0.0 for _ in data]
        return 0.0

    def step(self):
        # Revalidate every parameter, hyperparameter and state slot first,
        # then validate every active grad; compute all new values before
        # mutating anything, so a failure leaves all data, m, v, v_max and
        # t untouched.
        parameters = self.parameters
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        self._check_hyperparameters(
            self.lr, self.beta1, self.beta2, self.eps
        )
        if isinstance(self.t, bool) or not isinstance(self.t, int):
            raise TypeError("t must be a non-bool non-negative int")
        if self.t < 0:
            raise ValueError("t must be a non-bool non-negative int")
        for name, buffer in (
            ("m", self.m),
            ("v", self.v),
            ("v_max", self.v_max),
        ):
            if not isinstance(buffer, list):
                raise TypeError(
                    name + " must be a list matching the parameters"
                )
            if len(buffer) != len(parameters):
                raise ValueError(
                    name + " must match the parameters in length"
                )
        for index, parameter in enumerate(parameters):
            data = _validate_data(parameter.data)
            self._check_buffer(self.m[index], data, "m")
            self._check_buffer(self.v[index], data, "v", non_negative=True)
            self._check_buffer(
                self.v_max[index], data, "v_max", non_negative=True
            )
            self._check_max_ge(self.v_max[index], self.v[index])
            if parameter.requires_grad and parameter.grad is not None:
                self._check_grad(parameter.grad, data)
        active = [
            index
            for index, parameter in enumerate(parameters)
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not active:
            return None
        k = self.t + 1
        bias1 = 1.0 - _finite_power(
            self.beta1, k, "bias correction must be finite"
        )
        bias2 = 1.0 - _finite_power(
            self.beta2, k, "bias correction must be finite"
        )
        if not math.isfinite(bias1) or not math.isfinite(bias2):
            raise ValueError("bias correction must be finite")
        updates = []
        for index in active:
            parameter = self.parameters[index]
            updates.append(
                (
                    index,
                    self._updated(
                        parameter.data,
                        parameter.grad,
                        self.m[index],
                        self.v[index],
                        self.v_max[index],
                        bias1,
                        bias2,
                    ),
                )
            )
        for index, (new_data, new_m, new_v, new_v_max) in updates:
            self.parameters[index].data = new_data
            self.m[index] = new_m
            self.v[index] = new_v
            self.v_max[index] = new_v_max
        self.t = k
        return None

    @staticmethod
    def _check_buffer(buffer, data, name, non_negative=False):
        """Type/shape/finiteness (and optionally range) check for a buffer."""
        if isinstance(data, list):
            if not isinstance(buffer, list):
                if isinstance(buffer, float) and not isinstance(buffer, bool):
                    raise ValueError(
                        name + " shape must match parameter shape"
                    )
                raise TypeError(name + " must be a list of floats")
            if len(buffer) != len(data):
                raise ValueError(name + " shape must match parameter shape")
            for value in buffer:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError(name + " elements must be floats")
            for value in buffer:
                if not math.isfinite(value):
                    raise ValueError(name + " must be finite")
            if non_negative:
                for value in buffer:
                    if value < 0.0:
                        raise ValueError(name + " must be non-negative")
        else:
            if isinstance(buffer, list):
                raise ValueError(name + " shape must match parameter shape")
            if isinstance(buffer, bool) or not isinstance(buffer, float):
                raise TypeError(name + " must be a float")
            if not math.isfinite(buffer):
                raise ValueError(name + " must be finite")
            if non_negative and buffer < 0.0:
                raise ValueError(name + " must be non-negative")

    @staticmethod
    def _check_max_ge(v_max, velocity):
        """ValueError unless v_max >= v holds elementwise."""
        if isinstance(v_max, list):
            for top, value in zip(v_max, velocity):
                if top < value:
                    raise ValueError("v_max must be elementwise >= v")
        elif v_max < velocity:
            raise ValueError("v_max must be elementwise >= v")

    @staticmethod
    def _check_grad(grad, data):
        """Type/shape/finiteness check for an active parameter's grad."""
        if isinstance(data, list):
            if not isinstance(grad, list):
                if isinstance(grad, float) and not isinstance(grad, bool):
                    raise ValueError("grad shape must match tensor shape")
                raise TypeError("grad must be a list of floats")
            if len(grad) != len(data):
                raise ValueError("grad shape must match tensor shape")
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("grad elements must be floats")
            for value in grad:
                if not math.isfinite(value):
                    raise ValueError("grad must be finite")
        else:
            if isinstance(grad, list):
                raise ValueError("grad shape must match tensor shape")
            if isinstance(grad, bool) or not isinstance(grad, float):
                raise TypeError("grad must be a float")
            if not math.isfinite(grad):
                raise ValueError("grad must be finite")

    def _updated(self, data, grad, moment, velocity, v_max, bias1, bias2):
        """Compute (new_data, new_m, new_v, new_v_max) for one parameter."""
        one_minus_b1 = 1.0 - self.beta1
        one_minus_b2 = 1.0 - self.beta2
        if isinstance(data, list):
            new_m = []
            new_v = []
            new_v_max = []
            new_data = []
            for value, g, m_value, v_value, h_value in zip(
                data, grad, moment, velocity, v_max
            ):
                m_next = self.beta1 * m_value + one_minus_b1 * g
                v_next = self.beta2 * v_value + one_minus_b2 * g * g
                if not math.isfinite(m_next) or not math.isfinite(v_next):
                    raise ValueError("amsgrad buffers must be finite")
                h_next = max(h_value, v_next)
                if not math.isfinite(h_next):
                    raise ValueError("amsgrad buffers must be finite")
                m_hat = m_next / bias1
                h_hat = h_next / bias2
                if not math.isfinite(m_hat) or not math.isfinite(h_hat):
                    raise ValueError("bias-corrected moments must be finite")
                try:
                    denominator = math.sqrt(h_hat) + self.eps
                except (ValueError, OverflowError):
                    raise ValueError("update denominator must be finite")
                if not math.isfinite(denominator):
                    raise ValueError("update denominator must be finite")
                value_next = value - self.lr * m_hat / denominator
                if not math.isfinite(value_next):
                    raise ValueError("updated data must be finite")
                new_m.append(m_next)
                new_v.append(v_next)
                new_v_max.append(h_next)
                new_data.append(value_next)
            return new_data, new_m, new_v, new_v_max
        m_next = self.beta1 * moment + one_minus_b1 * grad
        v_next = self.beta2 * velocity + one_minus_b2 * grad * grad
        if not math.isfinite(m_next) or not math.isfinite(v_next):
            raise ValueError("amsgrad buffers must be finite")
        h_next = max(v_max, v_next)
        if not math.isfinite(h_next):
            raise ValueError("amsgrad buffers must be finite")
        m_hat = m_next / bias1
        h_hat = h_next / bias2
        if not math.isfinite(m_hat) or not math.isfinite(h_hat):
            raise ValueError("bias-corrected moments must be finite")
        try:
            denominator = math.sqrt(h_hat) + self.eps
        except (ValueError, OverflowError):
            raise ValueError("update denominator must be finite")
        if not math.isfinite(denominator):
            raise ValueError("update denominator must be finite")
        value_next = data - self.lr * m_hat / denominator
        if not math.isfinite(value_next):
            raise ValueError("updated data must be finite")
        return value_next, m_next, v_next, h_next

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


class MomentumSGD:
    """SGD with (optionally Nesterov) momentum over a fixed list of Tensors."""

    def __init__(self, parameters, lr, momentum=0.9, nesterov=False):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        if isinstance(momentum, bool) or not isinstance(momentum, float):
            raise TypeError(
                "momentum must be a finite float in [0.0, 1.0)"
            )
        if not math.isfinite(momentum) or momentum < 0.0 or momentum >= 1.0:
            raise ValueError(
                "momentum must be a finite float in [0.0, 1.0)"
            )
        if not isinstance(nesterov, bool):
            raise TypeError("nesterov must be a bool")
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = [
            self._zeros_like(parameter.data) for parameter in parameters
        ]

    @staticmethod
    def _zeros_like(data):
        if isinstance(data, list):
            return [0.0 for _ in data]
        return 0.0

    def step(self):
        # Revalidate the parameters list structurally (non-empty list of
        # unique Tensors) before reading any element attribute, then every
        # velocity slot and every active grad; compute all new values
        # before mutating anything, so a failure leaves all data, grad and
        # velocity untouched.
        parameters = self.parameters
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        if not isinstance(self.velocity, list):
            raise TypeError("velocity must be a list")
        if len(self.velocity) != len(parameters):
            raise ValueError("velocity must match parameters")
        for index, parameter in enumerate(parameters):
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            data = _validate_data(parameter.data)
            self._check_buffer(self.velocity[index], data, "velocity")
            if parameter.requires_grad and parameter.grad is not None:
                self._check_grad(parameter.grad, data)
        active = [
            index
            for index, parameter in enumerate(parameters)
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not active:
            return None
        updates = []
        for index in active:
            parameter = self.parameters[index]
            updates.append(
                (
                    index,
                    self._updated(
                        parameter.data,
                        parameter.grad,
                        self.velocity[index],
                    ),
                )
            )
        for index, (new_data, new_velocity) in updates:
            self.parameters[index].data = new_data
            self.velocity[index] = new_velocity
        return None

    @staticmethod
    def _check_buffer(buffer, data, name):
        """Type/shape/finiteness check for a velocity slot."""
        if isinstance(data, list):
            if not isinstance(buffer, list):
                if isinstance(buffer, float) and not isinstance(buffer, bool):
                    raise ValueError(
                        name + " shape must match parameter shape"
                    )
                raise TypeError(name + " must be a list of floats")
            if len(buffer) != len(data):
                raise ValueError(name + " shape must match parameter shape")
            for value in buffer:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError(name + " elements must be floats")
            for value in buffer:
                if not math.isfinite(value):
                    raise ValueError(name + " must be finite")
        else:
            if isinstance(buffer, list):
                raise ValueError(name + " shape must match parameter shape")
            if isinstance(buffer, bool) or not isinstance(buffer, float):
                raise TypeError(name + " must be a float")
            if not math.isfinite(buffer):
                raise ValueError(name + " must be finite")

    @staticmethod
    def _check_grad(grad, data):
        """Type/shape/finiteness check for an active parameter's grad."""
        if isinstance(data, list):
            if not isinstance(grad, list):
                if isinstance(grad, float) and not isinstance(grad, bool):
                    raise ValueError("grad shape must match tensor shape")
                raise TypeError("grad must be a list of floats")
            if len(grad) != len(data):
                raise ValueError("grad shape must match tensor shape")
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("grad elements must be floats")
            for value in grad:
                if not math.isfinite(value):
                    raise ValueError("grad must be finite")
        else:
            if isinstance(grad, list):
                raise ValueError("grad shape must match tensor shape")
            if isinstance(grad, bool) or not isinstance(grad, float):
                raise TypeError("grad must be a float")
            if not math.isfinite(grad):
                raise ValueError("grad must be finite")

    def _updated(self, data, grad, velocity):
        """Compute (new_data, new_velocity) for one active parameter."""
        if isinstance(data, list):
            new_velocity = []
            new_data = []
            for value, g, v in zip(data, grad, velocity):
                value_next, v_next = self._updated_value(value, g, v)
                new_velocity.append(v_next)
                new_data.append(value_next)
            return new_data, new_velocity
        return self._updated_value(data, grad, velocity)

    def _updated_value(self, value, g, v):
        v_next = self.momentum * v + g
        if not math.isfinite(v_next):
            raise ValueError("velocity must be finite")
        if self.nesterov:
            direction = g + self.momentum * v_next
        else:
            direction = v_next
        if not math.isfinite(direction):
            raise ValueError("update direction must be finite")
        value_next = value - self.lr * direction
        if not math.isfinite(value_next):
            raise ValueError("updated data must be finite")
        return value_next, v_next

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


class AdamW:
    """AdamW optimizer over a fixed list of Tensors."""

    def __init__(
        self, parameters, lr, beta1=0.9, beta2=0.999, eps=1e-8,
        weight_decay=0.01,
    ):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        self._check_hyperparameters(lr, beta1, beta2, eps, weight_decay)
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [self._zeros_like(parameter.data) for parameter in parameters]
        self.v = [self._zeros_like(parameter.data) for parameter in parameters]
        self.t = 0

    @staticmethod
    def _check_hyperparameters(lr, beta1, beta2, eps, weight_decay):
        """Validate lr, betas, eps and weight_decay exactly as __init__ does."""
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        for name, beta in (("beta1", beta1), ("beta2", beta2)):
            if isinstance(beta, bool) or not isinstance(beta, float):
                raise TypeError(
                    name
                    + " must be a finite float in [0.0, 1.0)"
                )
            if not math.isfinite(beta) or beta < 0.0 or beta >= 1.0:
                raise ValueError(
                    name
                    + " must be a finite float in [0.0, 1.0)"
                )
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        if isinstance(weight_decay, bool) or not isinstance(weight_decay, float):
            raise TypeError("weight_decay must be a non-negative finite float")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be a non-negative finite float")

    @staticmethod
    def _zeros_like(data):
        if isinstance(data, list):
            return [0.0 for _ in data]
        return 0.0

    @staticmethod
    def _finite(value, message):
        """ValueError if the result of an arithmetic step is non-finite."""
        if not math.isfinite(value):
            raise ValueError(message)
        return value

    def step(self):
        # Revalidate every parameter and validate every active grad first;
        # compute all new values before mutating anything, so a failure leaves
        # all data, m, v and t untouched.
        for index, parameter in enumerate(self.parameters):
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            data = _validate_data(parameter.data)
            moment = self.m[index]
            velocity = self.v[index]
            self._check_buffer(moment, data, "m")
            self._check_buffer(velocity, data, "v")
            if parameter.requires_grad and parameter.grad is not None:
                self._check_grad(parameter.grad, data)
        active = [
            index
            for index, parameter in enumerate(self.parameters)
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not active:
            return None
        k = self.t + 1
        bias1 = 1.0 - _finite_power(
            self.beta1, k, "bias correction must be finite"
        )
        bias2 = 1.0 - _finite_power(
            self.beta2, k, "bias correction must be finite"
        )
        if not math.isfinite(bias1) or not math.isfinite(bias2):
            raise ValueError("bias correction must be finite")
        updates = []
        for index in active:
            parameter = self.parameters[index]
            updates.append(
                (
                    index,
                    self._updated(
                        parameter.data,
                        parameter.grad,
                        self.m[index],
                        self.v[index],
                        bias1,
                        bias2,
                    ),
                )
            )
        for index, (new_data, new_m, new_v) in updates:
            self.parameters[index].data = new_data
            self.m[index] = new_m
            self.v[index] = new_v
        self.t = k
        return None

    @staticmethod
    def _check_buffer(buffer, data, name):
        """Type/shape/finiteness check for an m or v moment buffer."""
        if isinstance(data, list):
            if not isinstance(buffer, list):
                if isinstance(buffer, float) and not isinstance(buffer, bool):
                    raise ValueError(
                        name + " shape must match parameter shape"
                    )
                raise TypeError(name + " must be a list of floats")
            if len(buffer) != len(data):
                raise ValueError(name + " shape must match parameter shape")
            for value in buffer:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError(name + " elements must be floats")
            for value in buffer:
                if not math.isfinite(value):
                    raise ValueError(name + " must be finite")
        else:
            if isinstance(buffer, list):
                raise ValueError(name + " shape must match parameter shape")
            if isinstance(buffer, bool) or not isinstance(buffer, float):
                raise TypeError(name + " must be a float")
            if not math.isfinite(buffer):
                raise ValueError(name + " must be finite")

    @staticmethod
    def _check_grad(grad, data):
        """Type/shape/finiteness check for an active parameter's grad."""
        if isinstance(data, list):
            if not isinstance(grad, list):
                if isinstance(grad, float) and not isinstance(grad, bool):
                    raise ValueError("grad shape must match tensor shape")
                raise TypeError("grad must be a list of floats")
            if len(grad) != len(data):
                raise ValueError("grad shape must match tensor shape")
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("grad elements must be floats")
            for value in grad:
                if not math.isfinite(value):
                    raise ValueError("grad must be finite")
        else:
            if isinstance(grad, list):
                raise ValueError("grad shape must match tensor shape")
            if isinstance(grad, bool) or not isinstance(grad, float):
                raise TypeError("grad must be a float")
            if not math.isfinite(grad):
                raise ValueError("grad must be finite")

    def _updated(self, data, grad, moment, velocity, bias1, bias2):
        """Compute (new_data, new_m, new_v) for one active parameter."""
        lr_weight_decay = self._finite(
            self.lr * self.weight_decay, "weight decay must be finite"
        )
        if isinstance(data, list):
            new_m = []
            new_v = []
            new_data = []
            for value, g, m_value, v_value in zip(
                data, grad, moment, velocity
            ):
                value_next, m_next, v_next = self._element(
                    value, g, m_value, v_value, bias1, bias2, lr_weight_decay
                )
                new_data.append(value_next)
                new_m.append(m_next)
                new_v.append(v_next)
            return new_data, new_m, new_v
        return self._element(
            data, grad, moment, velocity, bias1, bias2, lr_weight_decay
        )

    def _element(
        self, value, g, m_value, v_value, bias1, bias2, lr_weight_decay
    ):
        """Single-element AdamW update, validating every arithmetic step."""
        one_minus_b1 = 1.0 - self.beta1
        one_minus_b2 = 1.0 - self.beta2
        m_decayed = self._finite(
            self.beta1 * m_value, "adamw multiply must be finite"
        )
        m_grad = self._finite(
            one_minus_b1 * g, "adamw multiply must be finite"
        )
        m_next = self._finite(
            m_decayed + m_grad, "adamw add must be finite"
        )
        v_decayed = self._finite(
            self.beta2 * v_value, "adamw multiply must be finite"
        )
        g_scaled = self._finite(
            one_minus_b2 * g, "adamw multiply must be finite"
        )
        v_grad = self._finite(
            g_scaled * g, "adamw multiply must be finite"
        )
        v_next = self._finite(
            v_decayed + v_grad, "adamw add must be finite"
        )
        m_hat = self._finite(m_next / bias1, "adamw divide must be finite")
        v_hat = self._finite(v_next / bias2, "adamw divide must be finite")
        try:
            sqrt_v_hat = math.sqrt(v_hat)
        except (ValueError, OverflowError):
            raise ValueError("update denominator must be finite")
        sqrt_v_hat = self._finite(
            sqrt_v_hat, "update denominator must be finite"
        )
        denominator = self._finite(
            sqrt_v_hat + self.eps, "adamw add must be finite"
        )
        decay_step = self._finite(
            lr_weight_decay * value, "adamw multiply must be finite"
        )
        decayed = self._finite(value - decay_step, "updated data must be finite")
        m_step = self._finite(
            self.lr * m_hat, "adamw multiply must be finite"
        )
        m_step = self._finite(
            m_step / denominator, "adamw divide must be finite"
        )
        value_next = self._finite(
            decayed - m_step, "updated data must be finite"
        )
        return value_next, m_next, v_next

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


class Adamax:
    """Adamax optimizer over a fixed list of Tensors."""

    def __init__(self, parameters, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        self._check_hyperparameters(lr, beta1, beta2, eps)
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [self._zeros_like(parameter.data) for parameter in parameters]
        self.u = [self._zeros_like(parameter.data) for parameter in parameters]
        self.t = 0

    @staticmethod
    def _check_hyperparameters(lr, beta1, beta2, eps):
        """Validate lr, beta1, beta2 and eps exactly as the constructor does."""
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        for name, beta in (("beta1", beta1), ("beta2", beta2)):
            if isinstance(beta, bool) or not isinstance(beta, float):
                raise TypeError(
                    name
                    + " must be a finite float in [0.0, 1.0)"
                )
            if not math.isfinite(beta) or beta < 0.0 or beta >= 1.0:
                raise ValueError(
                    name
                    + " must be a finite float in [0.0, 1.0)"
                )
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")

    @staticmethod
    def _zeros_like(data):
        if isinstance(data, list):
            return [0.0 for _ in data]
        return 0.0

    def step(self):
        # Revalidate every parameter, hyperparameter and state slot first,
        # then validate every active grad; compute all new values before
        # mutating anything, so a failure leaves all data, m, u and t
        # untouched.
        parameters = self.parameters
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        self._check_hyperparameters(
            self.lr, self.beta1, self.beta2, self.eps
        )
        if isinstance(self.t, bool) or not isinstance(self.t, int):
            raise TypeError("t must be a non-bool non-negative int")
        if self.t < 0:
            raise ValueError("t must be a non-bool non-negative int")
        for name, buffer in (("m", self.m), ("u", self.u)):
            if not isinstance(buffer, list):
                raise TypeError(
                    name + " must be a list matching the parameters"
                )
            if len(buffer) != len(parameters):
                raise ValueError(
                    name + " must match the parameters in length"
                )
        for index, parameter in enumerate(parameters):
            data = _validate_data(parameter.data)
            self._check_buffer(self.m[index], data, "m")
            self._check_buffer(self.u[index], data, "u", non_negative=True)
            if parameter.requires_grad and parameter.grad is not None:
                self._check_grad(parameter.grad, data)
        active = [
            index
            for index, parameter in enumerate(parameters)
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not active:
            return None
        k = self.t + 1
        bias1 = 1.0 - _finite_power(
            self.beta1, k, "bias correction must be finite"
        )
        if not math.isfinite(bias1):
            raise ValueError("bias correction must be finite")
        updates = []
        for index in active:
            parameter = self.parameters[index]
            updates.append(
                (
                    index,
                    self._updated(
                        parameter.data,
                        parameter.grad,
                        self.m[index],
                        self.u[index],
                        bias1,
                    ),
                )
            )
        for index, (new_data, new_m, new_u) in updates:
            self.parameters[index].data = new_data
            self.m[index] = new_m
            self.u[index] = new_u
        self.t = k
        return None

    @staticmethod
    def _check_buffer(buffer, data, name, non_negative=False):
        """Type/shape/finiteness (and optionally range) check for m or u."""
        if isinstance(data, list):
            if not isinstance(buffer, list):
                if isinstance(buffer, float) and not isinstance(buffer, bool):
                    raise ValueError(
                        name + " shape must match parameter shape"
                    )
                raise TypeError(name + " must be a list of floats")
            if len(buffer) != len(data):
                raise ValueError(name + " shape must match parameter shape")
            for value in buffer:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError(name + " elements must be floats")
            for value in buffer:
                if not math.isfinite(value):
                    raise ValueError(name + " must be finite")
            if non_negative:
                for value in buffer:
                    if value < 0.0:
                        raise ValueError(name + " must be non-negative")
        else:
            if isinstance(buffer, list):
                raise ValueError(name + " shape must match parameter shape")
            if isinstance(buffer, bool) or not isinstance(buffer, float):
                raise TypeError(name + " must be a float")
            if not math.isfinite(buffer):
                raise ValueError(name + " must be finite")
            if non_negative and buffer < 0.0:
                raise ValueError(name + " must be non-negative")

    @staticmethod
    def _check_grad(grad, data):
        """Type/shape/finiteness check for an active parameter's grad."""
        if isinstance(data, list):
            if not isinstance(grad, list):
                if isinstance(grad, float) and not isinstance(grad, bool):
                    raise ValueError("grad shape must match tensor shape")
                raise TypeError("grad must be a list of floats")
            if len(grad) != len(data):
                raise ValueError("grad shape must match tensor shape")
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("grad elements must be floats")
            for value in grad:
                if not math.isfinite(value):
                    raise ValueError("grad must be finite")
        else:
            if isinstance(grad, list):
                raise ValueError("grad shape must match tensor shape")
            if isinstance(grad, bool) or not isinstance(grad, float):
                raise TypeError("grad must be a float")
            if not math.isfinite(grad):
                raise ValueError("grad must be finite")

    def _updated(self, data, grad, moment, u_max, bias1):
        """Compute (new_data, new_m, new_u) for one active parameter."""
        one_minus_b1 = 1.0 - self.beta1
        if isinstance(data, list):
            new_m = []
            new_u = []
            new_data = []
            for value, g, m_value, u_value in zip(
                data, grad, moment, u_max
            ):
                m_next = self.beta1 * m_value + one_minus_b1 * g
                if not math.isfinite(m_next):
                    raise ValueError("adamax buffers must be finite")
                u_next = max(self.beta2 * u_value, abs(g))
                if not math.isfinite(u_next):
                    raise ValueError("adamax buffers must be finite")
                denominator = u_next + self.eps
                if not math.isfinite(denominator):
                    raise ValueError("update denominator must be finite")
                step = self.lr * m_next / bias1 / denominator
                if not math.isfinite(step):
                    raise ValueError("update step must be finite")
                value_next = value - step
                if not math.isfinite(value_next):
                    raise ValueError("updated data must be finite")
                new_m.append(m_next)
                new_u.append(u_next)
                new_data.append(value_next)
            return new_data, new_m, new_u
        m_next = self.beta1 * moment + one_minus_b1 * grad
        if not math.isfinite(m_next):
            raise ValueError("adamax buffers must be finite")
        u_next = max(self.beta2 * u_max, abs(grad))
        if not math.isfinite(u_next):
            raise ValueError("adamax buffers must be finite")
        denominator = u_next + self.eps
        if not math.isfinite(denominator):
            raise ValueError("update denominator must be finite")
        step = self.lr * m_next / bias1 / denominator
        if not math.isfinite(step):
            raise ValueError("update step must be finite")
        value_next = data - step
        if not math.isfinite(value_next):
            raise ValueError("updated data must be finite")
        return value_next, m_next, u_next

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


class RMSprop:
    """RMSprop optimizer over a fixed list of Tensors."""

    def __init__(self, parameters, lr, alpha=0.99, eps=1e-8):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        if isinstance(alpha, bool) or not isinstance(alpha, float):
            raise TypeError("alpha must be a finite float in [0.0, 1.0)")
        if not math.isfinite(alpha) or alpha < 0.0 or alpha >= 1.0:
            raise ValueError("alpha must be a finite float in [0.0, 1.0)")
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        self.parameters = list(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg = [
            self._zeros_like(parameter.data) for parameter in parameters
        ]

    @staticmethod
    def _zeros_like(data):
        if isinstance(data, list):
            return [0.0 for _ in data]
        return 0.0

    def step(self):
        # Revalidate every parameter and validate every active grad first;
        # compute all new values before mutating anything, so a failure leaves
        # all data and square_avg untouched.
        if not isinstance(self.square_avg, list):
            raise TypeError("square_avg must be a list of slots")
        if len(self.square_avg) != len(self.parameters):
            raise ValueError("square_avg shape must match parameter shape")
        for index, parameter in enumerate(self.parameters):
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            data = _validate_data(parameter.data)
            self._check_slot(self.square_avg[index], data)
            if parameter.requires_grad and parameter.grad is not None:
                self._check_grad(parameter.grad, data)
        active = [
            index
            for index, parameter in enumerate(self.parameters)
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not active:
            return None
        one_minus_alpha = 1.0 - self.alpha
        updates = []
        for index in active:
            parameter = self.parameters[index]
            updates.append(
                (
                    index,
                    self._updated(
                        parameter.data,
                        parameter.grad,
                        self.square_avg[index],
                        one_minus_alpha,
                    ),
                )
            )
        for index, (new_data, new_square_avg) in updates:
            self.parameters[index].data = new_data
            self.square_avg[index] = new_square_avg
        return None

    @staticmethod
    def _check_slot(slot, data):
        """Type/shape/finiteness check for a square_avg slot."""
        if isinstance(data, list):
            if not isinstance(slot, list):
                if isinstance(slot, float) and not isinstance(slot, bool):
                    raise ValueError(
                        "square_avg shape must match parameter shape"
                    )
                raise TypeError("square_avg must be a list of floats")
            if len(slot) != len(data):
                raise ValueError("square_avg shape must match parameter shape")
            for value in slot:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("square_avg elements must be floats")
            for value in slot:
                if not math.isfinite(value):
                    raise ValueError("square_avg must be finite")
        else:
            if isinstance(slot, list):
                raise ValueError("square_avg shape must match parameter shape")
            if isinstance(slot, bool) or not isinstance(slot, float):
                raise TypeError("square_avg must be a float")
            if not math.isfinite(slot):
                raise ValueError("square_avg must be finite")

    @staticmethod
    def _check_grad(grad, data):
        """Type/shape/finiteness check for an active parameter's grad."""
        if isinstance(data, list):
            if not isinstance(grad, list):
                if isinstance(grad, float) and not isinstance(grad, bool):
                    raise ValueError("grad shape must match tensor shape")
                raise TypeError("grad must be a list of floats")
            if len(grad) != len(data):
                raise ValueError("grad shape must match tensor shape")
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("grad elements must be floats")
            for value in grad:
                if not math.isfinite(value):
                    raise ValueError("grad must be finite")
        else:
            if isinstance(grad, list):
                raise ValueError("grad shape must match tensor shape")
            if isinstance(grad, bool) or not isinstance(grad, float):
                raise TypeError("grad must be a float")
            if not math.isfinite(grad):
                raise ValueError("grad must be finite")

    def _updated(self, data, grad, square_avg, one_minus_alpha):
        """Compute (new_data, new_square_avg) for one active parameter."""
        if isinstance(data, list):
            new_square_avg = []
            new_data = []
            for value, g, old in zip(data, grad, square_avg):
                next_avg = (
                    self.alpha * old + one_minus_alpha * g * g
                )
                if not math.isfinite(next_avg):
                    raise ValueError("square_avg must be finite")
                try:
                    denominator = math.sqrt(next_avg) + self.eps
                except (ValueError, OverflowError):
                    raise ValueError("update denominator must be finite")
                if not math.isfinite(denominator):
                    raise ValueError("update denominator must be finite")
                value_next = value - self.lr * g / denominator
                if not math.isfinite(value_next):
                    raise ValueError("updated data must be finite")
                new_square_avg.append(next_avg)
                new_data.append(value_next)
            return new_data, new_square_avg
        next_avg = self.alpha * square_avg + one_minus_alpha * grad * grad
        if not math.isfinite(next_avg):
            raise ValueError("square_avg must be finite")
        try:
            denominator = math.sqrt(next_avg) + self.eps
        except (ValueError, OverflowError):
            raise ValueError("update denominator must be finite")
        if not math.isfinite(denominator):
            raise ValueError("update denominator must be finite")
        value_next = data - self.lr * grad / denominator
        if not math.isfinite(value_next):
            raise ValueError("updated data must be finite")
        return value_next, next_avg

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


class Adagrad:
    """Adagrad optimizer over a fixed list of Tensors."""

    def __init__(self, parameters, lr, eps=1e-8):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        if isinstance(lr, bool) or not isinstance(lr, float):
            raise TypeError("lr must be a positive finite float")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite float")
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        self.parameters = list(parameters)
        self.lr = lr
        self.eps = eps
        self.sum_sq = [
            self._zeros_like(parameter.data) for parameter in parameters
        ]

    @staticmethod
    def _zeros_like(data):
        if isinstance(data, list):
            return [0.0 for _ in data]
        return 0.0

    def step(self):
        # Revalidate every parameter and every sum_sq slot first; validate
        # every active grad, then compute all new values before mutating
        # anything, so a failure leaves all data, sum_sq, grad and
        # requires_grad untouched.
        if not isinstance(self.sum_sq, list):
            raise TypeError("sum_sq must be a list of slots")
        if len(self.sum_sq) != len(self.parameters):
            raise ValueError("sum_sq shape must match parameter shape")
        for index, parameter in enumerate(self.parameters):
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            data = _validate_data(parameter.data)
            self._check_slot(self.sum_sq[index], data)
            if parameter.requires_grad and parameter.grad is not None:
                self._check_grad(parameter.grad, data)
        updates = []
        for index, parameter in enumerate(self.parameters):
            if not parameter.requires_grad or parameter.grad is None:
                continue
            updates.append(
                (
                    index,
                    self._updated(
                        parameter.data,
                        parameter.grad,
                        self.sum_sq[index],
                    ),
                )
            )
        for index, (new_data, new_sum_sq) in updates:
            self.parameters[index].data = new_data
            self.sum_sq[index] = new_sum_sq
        return None

    @staticmethod
    def _check_slot(slot, data):
        """Type/shape/finiteness check for a sum_sq slot."""
        if isinstance(data, list):
            if not isinstance(slot, list):
                if isinstance(slot, float) and not isinstance(slot, bool):
                    raise ValueError(
                        "sum_sq shape must match parameter shape"
                    )
                raise TypeError("sum_sq must be a list of floats")
            if len(slot) != len(data):
                raise ValueError("sum_sq shape must match parameter shape")
            for value in slot:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("sum_sq elements must be floats")
            for value in slot:
                if not math.isfinite(value):
                    raise ValueError("sum_sq must be finite")
        else:
            if isinstance(slot, list):
                raise ValueError("sum_sq shape must match parameter shape")
            if isinstance(slot, bool) or not isinstance(slot, float):
                raise TypeError("sum_sq must be a float")
            if not math.isfinite(slot):
                raise ValueError("sum_sq must be finite")

    @staticmethod
    def _check_grad(grad, data):
        """Type/shape/finiteness check for an active parameter's grad."""
        if isinstance(data, list):
            if not isinstance(grad, list):
                if isinstance(grad, float) and not isinstance(grad, bool):
                    raise ValueError("grad shape must match tensor shape")
                raise TypeError("grad must be a list of floats")
            if len(grad) != len(data):
                raise ValueError("grad shape must match tensor shape")
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("grad elements must be floats")
            for value in grad:
                if not math.isfinite(value):
                    raise ValueError("grad must be finite")
        else:
            if isinstance(grad, list):
                raise ValueError("grad shape must match tensor shape")
            if isinstance(grad, bool) or not isinstance(grad, float):
                raise TypeError("grad must be a float")
            if not math.isfinite(grad):
                raise ValueError("grad must be finite")

    def _updated(self, data, grad, sum_sq):
        """Compute (new_data, new_sum_sq) for one active parameter."""
        if isinstance(data, list):
            new_sum_sq = []
            new_data = []
            for value, g, old in zip(data, grad, sum_sq):
                next_sum = old + g * g
                if not math.isfinite(next_sum):
                    raise ValueError("sum_sq must be finite")
                try:
                    denominator = math.sqrt(next_sum) + self.eps
                except (ValueError, OverflowError):
                    raise ValueError("update denominator must be finite")
                if not math.isfinite(denominator):
                    raise ValueError("update denominator must be finite")
                value_next = value - self.lr * g / denominator
                if not math.isfinite(value_next):
                    raise ValueError("updated data must be finite")
                new_sum_sq.append(next_sum)
                new_data.append(value_next)
            return new_data, new_sum_sq
        next_sum = sum_sq + grad * grad
        if not math.isfinite(next_sum):
            raise ValueError("sum_sq must be finite")
        try:
            denominator = math.sqrt(next_sum) + self.eps
        except (ValueError, OverflowError):
            raise ValueError("update denominator must be finite")
        if not math.isfinite(denominator):
            raise ValueError("update denominator must be finite")
        value_next = data - self.lr * grad / denominator
        if not math.isfinite(value_next):
            raise ValueError("updated data must be finite")
        return value_next, next_sum

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None
        return None


class Adadelta:
    """Adadelta optimizer over a fixed list of Tensors."""

    def __init__(self, parameters, rho=0.9, eps=1e-6):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)
        if isinstance(rho, bool) or not isinstance(rho, float):
            raise TypeError("rho must be a finite float in [0.0, 1.0)")
        if not math.isfinite(rho) or rho < 0.0 or rho >= 1.0:
            raise ValueError("rho must be a finite float in [0.0, 1.0)")
        if isinstance(eps, bool) or not isinstance(eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        self.parameters = list(parameters)
        self.rho = rho
        self.eps = eps
        self.square_avg = [
            self._zeros_like(parameter.data) for parameter in parameters
        ]
        self.acc_delta = [
            self._zeros_like(parameter.data) for parameter in parameters
        ]

    @staticmethod
    def _zeros_like(data):
        if isinstance(data, list):
            return [0.0 for _ in data]
        return 0.0

    @staticmethod
    def _check_parameters(parameters):
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a non-empty list of Tensors")
        if len(parameters) == 0:
            raise ValueError("parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("parameters must contain only Tensors")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("parameters must not contain duplicate Tensors")
        for parameter in parameters:
            if not isinstance(parameter.requires_grad, bool):
                raise TypeError("requires_grad must be a bool")
            _validate_data(parameter.data)

    def step(self):
        # Revalidate every parameter, rho, eps and both slot lists first;
        # validate every active grad, then compute all new values before
        # mutating anything, so a failure leaves all data, square_avg,
        # acc_delta, grad and requires_grad untouched.
        self._check_parameters(self.parameters)
        if isinstance(self.rho, bool) or not isinstance(self.rho, float):
            raise TypeError("rho must be a finite float in [0.0, 1.0)")
        if not math.isfinite(self.rho) or self.rho < 0.0 or self.rho >= 1.0:
            raise ValueError("rho must be a finite float in [0.0, 1.0)")
        if isinstance(self.eps, bool) or not isinstance(self.eps, float):
            raise TypeError("eps must be a positive finite float")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        if not isinstance(self.square_avg, list):
            raise TypeError("square_avg must be a list of slots")
        if len(self.square_avg) != len(self.parameters):
            raise ValueError("square_avg shape must match parameter shape")
        if not isinstance(self.acc_delta, list):
            raise TypeError("acc_delta must be a list of slots")
        if len(self.acc_delta) != len(self.parameters):
            raise ValueError("acc_delta shape must match parameter shape")
        for index, parameter in enumerate(self.parameters):
            data = _validate_data(parameter.data)
            self._check_slot(self.square_avg[index], data, "square_avg")
            self._check_slot(self.acc_delta[index], data, "acc_delta")
            if parameter.requires_grad and parameter.grad is not None:
                self._check_grad(parameter.grad, data)
        updates = []
        for index, parameter in enumerate(self.parameters):
            if not parameter.requires_grad or parameter.grad is None:
                continue
            updates.append(
                (
                    index,
                    self._updated(
                        parameter.data,
                        parameter.grad,
                        self.square_avg[index],
                        self.acc_delta[index],
                    ),
                )
            )
        for index, (new_data, new_square_avg, new_acc_delta) in updates:
            self.parameters[index].data = new_data
            self.square_avg[index] = new_square_avg
            self.acc_delta[index] = new_acc_delta
        return None

    @staticmethod
    def _check_slot(slot, data, name):
        """Type/shape/non-negativity/finiteness check for one slot."""
        if isinstance(data, list):
            if not isinstance(slot, list):
                if isinstance(slot, float) and not isinstance(slot, bool):
                    raise ValueError(
                        name + " shape must match parameter shape"
                    )
                raise TypeError(name + " must be a list of floats")
            if len(slot) != len(data):
                raise ValueError(name + " shape must match parameter shape")
            for value in slot:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError(name + " elements must be floats")
            for value in slot:
                if not math.isfinite(value):
                    raise ValueError(name + " must be finite")
                if value < 0.0:
                    raise ValueError(name + " must be non-negative")
        else:
            if isinstance(slot, list):
                raise ValueError(name + " shape must match parameter shape")
            if isinstance(slot, bool) or not isinstance(slot, float):
                raise TypeError(name + " must be a float")
            if not math.isfinite(slot):
                raise ValueError(name + " must be finite")
            if slot < 0.0:
                raise ValueError(name + " must be non-negative")

    @staticmethod
    def _check_grad(grad, data):
        """Type/shape/finiteness check for an active parameter's grad."""
        if isinstance(data, list):
            if not isinstance(grad, list):
                if isinstance(grad, float) and not isinstance(grad, bool):
                    raise ValueError("grad shape must match tensor shape")
                raise TypeError("grad must be a list of floats")
            if len(grad) != len(data):
                raise ValueError("grad shape must match tensor shape")
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("grad elements must be floats")
            for value in grad:
                if not math.isfinite(value):
                    raise ValueError("grad must be finite")
        else:
            if isinstance(grad, list):
                raise ValueError("grad shape must match tensor shape")
            if isinstance(grad, bool) or not isinstance(grad, float):
                raise TypeError("grad must be a float")
            if not math.isfinite(grad):
                raise ValueError("grad must be finite")

    def _updated(self, data, grad, square_avg, acc_delta):
        """Compute (new_data, new_square_avg, new_acc_delta) for one active."""
        if isinstance(data, list):
            new_square_avg = []
            new_acc_delta = []
            new_data = []
            for value, g, s, a in zip(data, grad, square_avg, acc_delta):
                value_next, s_next, a_next = self._updated_value(value, g, s, a)
                new_data.append(value_next)
                new_square_avg.append(s_next)
                new_acc_delta.append(a_next)
            return new_data, new_square_avg, new_acc_delta
        return self._updated_value(data, grad, square_avg, acc_delta)

    def _updated_value(self, value, g, s, a):
        q = 1.0 - self.rho
        s1 = self.rho * s + q * g * g
        if not math.isfinite(s1):
            raise ValueError("square_avg must be finite")
        ratio = (a + self.eps) / (s1 + self.eps)
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError("update ratio must be finite")
        try:
            d = math.sqrt(ratio) * g
        except (ValueError, OverflowError):
            raise ValueError("update delta must be finite")
        if not math.isfinite(d):
            raise ValueError("update delta must be finite")
        a1 = self.rho * a + q * d * d
        if not math.isfinite(a1):
            raise ValueError("acc_delta must be finite")
        value_next = value - d
        if not math.isfinite(value_next):
            raise ValueError("updated data must be finite")
        return value_next, s1, a1

    def zero_grad(self):
        # Validate the full parameter list first, then clear every grad,
        # so a failure leaves every grad and all other state untouched.
        self._check_parameters(self.parameters)
        for parameter in self.parameters:
            parameter.grad = None
        return None


def _check_adagrad_state(optimizer):
    """Validate an Adagrad optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)
    sum_sq = optimizer.sum_sq
    if not isinstance(sum_sq, list):
        raise TypeError("sum_sq must be a list matching the parameters")
    if len(sum_sq) != len(parameters):
        raise ValueError("sum_sq must match the parameters in length")
    for slot, parameter in zip(sum_sq, parameters):
        Adagrad._check_slot(slot, parameter.data)


def dump_adagrad(optimizer):
    """Serialize an Adagrad optimizer's full state to a compact JSON string.

    The output contains no whitespace and no trailing newline; top-level
    keys are parameters, sum_sq in that order; each parameter entry has
    data then requires_grad; sum_sq entries are a scalar or an array.
    Floats are written with exactly six decimals and negative zero as
    0.000000. lr and eps are not serialized.
    """
    if not isinstance(optimizer, Adagrad):
        raise TypeError("optimizer must be an Adagrad instance")
    _check_adagrad_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    slots = (
        "["
        + ",".join(_json_value(item) for item in optimizer.sum_sq)
        + "]"
    )
    return (
        '{"parameters":['
        + ",".join(entries)
        + '],"sum_sq":'
        + slots
        + "}"
    )


class _AdagradParser:
    """Strict parser for the exact textual form produced by dump_adagrad."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_adagrad format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def _parse_slots(self):
        self._expect("[")
        items = [self._parse_data()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            items.append(self._parse_data())
        self._expect("]")
        return items

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect('],"sum_sq":')
        sum_sq = self._parse_slots()
        self._expect("}")
        if self._pos != len(self._text):
            self._fail()
        return parameters, sum_sq


def load_adagrad(optimizer, text):
    """Restore an Adagrad optimizer's full state from a dump_adagrad string.

    On full validation success, atomically replace every parameter's data
    and requires_grad, clear every grad, and replace sum_sq; lr, eps and
    the parameter identities are left unchanged. On any failure no state
    is touched.
    """
    if not isinstance(optimizer, Adagrad):
        raise TypeError("optimizer must be an Adagrad instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_adagrad_state(optimizer)
    parameters, sum_sq = _AdagradParser(text).parse()
    count = len(optimizer.parameters)
    if len(parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(sum_sq) != count:
        raise ValueError("state sum_sq must match the parameters in length")
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for slot, parameter in zip(sum_sq, optimizer.parameters):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(slot, list) or len(slot) != len(current):
                raise ValueError(
                    "state sum_sq shape must match parameter shape"
                )
        elif isinstance(slot, list):
            raise ValueError("state sum_sq shape must match parameter shape")
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.sum_sq = sum_sq
    return None


def _check_adadelta_state(optimizer):
    """Validate an Adadelta optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)
    if isinstance(optimizer.rho, bool) or not isinstance(optimizer.rho, float):
        raise TypeError("rho must be a finite float in [0.0, 1.0)")
    if not math.isfinite(optimizer.rho) or optimizer.rho < 0.0 or optimizer.rho >= 1.0:
        raise ValueError("rho must be a finite float in [0.0, 1.0)")
    if isinstance(optimizer.eps, bool) or not isinstance(optimizer.eps, float):
        raise TypeError("eps must be a positive finite float")
    if not math.isfinite(optimizer.eps) or optimizer.eps <= 0.0:
        raise ValueError("eps must be a positive finite float")
    square_avg = optimizer.square_avg
    if not isinstance(square_avg, list):
        raise TypeError("square_avg must be a list matching the parameters")
    if len(square_avg) != len(parameters):
        raise ValueError("square_avg must match the parameters in length")
    acc_delta = optimizer.acc_delta
    if not isinstance(acc_delta, list):
        raise TypeError("acc_delta must be a list matching the parameters")
    if len(acc_delta) != len(parameters):
        raise ValueError("acc_delta must match the parameters in length")
    for index, parameter in enumerate(parameters):
        Adadelta._check_slot(
            square_avg[index], parameter.data, "square_avg"
        )
        Adadelta._check_slot(
            acc_delta[index], parameter.data, "acc_delta"
        )


def dump_adadelta(optimizer):
    """Serialize an Adadelta optimizer's full state to a compact JSON string.

    The output contains no whitespace and no trailing newline; top-level
    keys are parameters, square_avg, acc_delta in that order; each
    parameter entry has data then requires_grad; the two slot arrays keep
    each parameter's scalar/array shape. Floats are written with exactly
    six decimals and negative zero as 0.000000. rho and eps are not
    serialized.
    """
    if not isinstance(optimizer, Adadelta):
        raise TypeError("optimizer must be an Adadelta instance")
    _check_adadelta_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    square_avg = (
        "["
        + ",".join(_json_value(item) for item in optimizer.square_avg)
        + "]"
    )
    acc_delta = (
        "["
        + ",".join(_json_value(item) for item in optimizer.acc_delta)
        + "]"
    )
    return (
        '{"parameters":['
        + ",".join(entries)
        + '],"square_avg":'
        + square_avg
        + ',"acc_delta":'
        + acc_delta
        + "}"
    )


class _AdadeltaParser:
    """Strict parser for the exact textual form produced by dump_adadelta."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_adadelta format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def _parse_slots(self):
        self._expect("[")
        items = [self._parse_data()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            items.append(self._parse_data())
        self._expect("]")
        return items

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect('],"square_avg":')
        square_avg = self._parse_slots()
        self._expect(',"acc_delta":')
        acc_delta = self._parse_slots()
        self._expect("}")
        if self._pos != len(self._text):
            self._fail()
        return parameters, square_avg, acc_delta


def _check_adadelta_slot_values(slots, name):
    """Reject negative (already-finite) parsed values in a slot array."""
    for slot in slots:
        values = slot if isinstance(slot, list) else [slot]
        for value in values:
            if value < 0.0:
                raise ValueError(name + " must be non-negative")


def load_adadelta(optimizer, text):
    """Restore an Adadelta optimizer's full state from a dump_adadelta string.

    On full validation success, atomically replace every parameter's data
    and requires_grad, clear every grad, and replace square_avg and
    acc_delta; rho, eps and the parameter identities are left unchanged.
    On any failure no state is touched.
    """
    if not isinstance(optimizer, Adadelta):
        raise TypeError("optimizer must be an Adadelta instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_adadelta_state(optimizer)
    parameters, square_avg, acc_delta = _AdadeltaParser(text).parse()
    count = len(optimizer.parameters)
    if len(parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(square_avg) != count:
        raise ValueError("state square_avg must match the parameters in length")
    if len(acc_delta) != count:
        raise ValueError("state acc_delta must match the parameters in length")
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for slot, parameter in zip(square_avg, optimizer.parameters):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(slot, list) or len(slot) != len(current):
                raise ValueError(
                    "state square_avg shape must match parameter shape"
                )
        elif isinstance(slot, list):
            raise ValueError("state square_avg shape must match parameter shape")
    for slot, parameter in zip(acc_delta, optimizer.parameters):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(slot, list) or len(slot) != len(current):
                raise ValueError(
                    "state acc_delta shape must match parameter shape"
                )
        elif isinstance(slot, list):
            raise ValueError("state acc_delta shape must match parameter shape")
    _check_adadelta_slot_values(square_avg, "square_avg")
    _check_adadelta_slot_values(acc_delta, "acc_delta")
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.square_avg = square_avg
    optimizer.acc_delta = acc_delta
    return None


def _check_momentum_sgd_state(optimizer):
    """Validate a MomentumSGD optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)
    if isinstance(optimizer.lr, bool) or not isinstance(optimizer.lr, float):
        raise TypeError("lr must be a positive finite float")
    if not math.isfinite(optimizer.lr) or optimizer.lr <= 0.0:
        raise ValueError("lr must be a positive finite float")
    if (
        isinstance(optimizer.momentum, bool)
        or not isinstance(optimizer.momentum, float)
    ):
        raise TypeError("momentum must be a finite float in [0.0, 1.0)")
    if (
        not math.isfinite(optimizer.momentum)
        or optimizer.momentum < 0.0
        or optimizer.momentum >= 1.0
    ):
        raise ValueError("momentum must be a finite float in [0.0, 1.0)")
    if not isinstance(optimizer.nesterov, bool):
        raise TypeError("nesterov must be a bool")
    velocity = optimizer.velocity
    if not isinstance(velocity, list):
        raise TypeError("velocity must be a list matching the parameters")
    if len(velocity) != len(parameters):
        raise ValueError("velocity must match the parameters in length")
    for slot, parameter in zip(velocity, parameters):
        MomentumSGD._check_buffer(slot, parameter.data, "velocity")


def dump_momentum_sgd(optimizer):
    """Serialize a MomentumSGD optimizer's full state to a compact JSON string.

    The output contains no whitespace and no trailing newline; top-level
    keys are parameters, velocity in that order; each parameter entry has
    data then requires_grad; velocity entries are a scalar or an array.
    Floats are written with exactly six decimals and negative zero as
    0.000000. lr, momentum and nesterov are not serialized.
    """
    if not isinstance(optimizer, MomentumSGD):
        raise TypeError("optimizer must be a MomentumSGD instance")
    _check_momentum_sgd_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    slots = (
        "["
        + ",".join(_json_value(item) for item in optimizer.velocity)
        + "]"
    )
    return (
        '{"parameters":['
        + ",".join(entries)
        + '],"velocity":'
        + slots
        + "}"
    )


class _MomentumSGDParser:
    """Strict parser for the exact textual form produced by dump_momentum_sgd."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_momentum_sgd format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def _parse_slots(self):
        self._expect("[")
        items = [self._parse_data()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            items.append(self._parse_data())
        self._expect("]")
        return items

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect('],"velocity":')
        velocity = self._parse_slots()
        self._expect("}")
        if self._pos != len(self._text):
            self._fail()
        return parameters, velocity


def load_momentum_sgd(optimizer, text):
    """Restore a MomentumSGD optimizer's full state from a dump string.

    On full validation success, atomically replace every parameter's data
    and requires_grad, clear every grad, and replace velocity; lr,
    momentum, nesterov and the parameter identities are left unchanged.
    On any failure no state is touched.
    """
    if not isinstance(optimizer, MomentumSGD):
        raise TypeError("optimizer must be a MomentumSGD instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_momentum_sgd_state(optimizer)
    parameters, velocity = _MomentumSGDParser(text).parse()
    count = len(optimizer.parameters)
    if len(parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(velocity) != count:
        raise ValueError("state velocity must match the parameters in length")
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for slot, parameter in zip(velocity, optimizer.parameters):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(slot, list) or len(slot) != len(current):
                raise ValueError(
                    "state velocity shape must match parameter shape"
                )
        elif isinstance(slot, list):
            raise ValueError("state velocity shape must match parameter shape")
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.velocity = velocity
    return None


def gradcheck(fn, data, eps=1e-6, atol=1e-5):
    """Compare analytic gradients from backward() with central differences.

    Returns (passed, max_abs_error) where passed is True only when every
    per-element absolute error is at most atol.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if isinstance(eps, bool) or not isinstance(eps, float):
        raise TypeError("eps must be a finite float")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")
    if isinstance(atol, bool) or not isinstance(atol, float):
        raise TypeError("atol must be a finite float")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a non-negative finite float")
    tensor = Tensor(data, True)

    def scalar_data(result):
        if not isinstance(result, Tensor):
            raise TypeError("fn must return a Tensor")
        if isinstance(result.data, list):
            raise ValueError("fn must return a scalar Tensor")
        return result.data

    result = fn(tensor)
    scalar_data(result)
    if not result._parents:
        raise ValueError("fn result must be part of a graph")
    result.backward()
    if tensor.grad is None:
        raise ValueError("fn result is not connected to the input")

    vector = isinstance(tensor.data, list)
    base = tensor.data if vector else [tensor.data]
    analytic = tensor.grad if vector else [tensor.grad]

    max_abs_error = 0.0
    passed = True
    for index in range(len(base)):
        plus = list(base)
        minus = list(base)
        plus[index] = base[index] + eps
        minus[index] = base[index] - eps
        f_plus = scalar_data(fn(Tensor(plus if vector else plus[0])))
        f_minus = scalar_data(fn(Tensor(minus if vector else minus[0])))
        numeric = (f_plus - f_minus) / (2.0 * eps)
        error = abs(analytic[index] - numeric)
        if error > max_abs_error:
            max_abs_error = error
        if error > atol:
            passed = False
    return (passed, max_abs_error)


def jacobiancheck(fn, data, eps=1e-6, atol=1e-5):
    """Compare analytic vector-Jacobian rows with central differences.

    ``fn`` maps a finite float scalar or a non-empty 1D float-list
    Tensor to a non-empty 1D float-list Tensor.  Each output row r is
    obtained analytically by backpropagating a length-m one-hot float
    list whose only 1.0 is at index r; each input column c is obtained
    numerically from central differences with +/- eps perturbations.

    A scalar input must receive a finite float analytic gradient; a
    vector input must receive a finite float list of the same length.
    A missing (None) analytic gradient, bad container or element types
    raise TypeError; wrong lengths or non-finite values raise ValueError.

    Returns (passed, max_abs_error) where passed is True only when every
    per-element absolute error is at most atol (a pair of exact zeros
    always passes).  Exceptions raised by ``fn`` propagate untouched and
    ``data`` is never modified.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    # Validate data up front so a scalar float input yields a float
    # analytic gradient and a list input yields an equal-length list.
    # The input list is copied; the caller's data object is never mutated.
    if isinstance(data, bool) or (
        not isinstance(data, float) and not isinstance(data, list)
    ):
        raise TypeError(
            "data must be a finite float scalar or a 1D float list"
        )
    if isinstance(data, float):
        if not math.isfinite(data):
            raise ValueError("data must be finite")
        vector_input = False
        base = [data]
    else:
        if len(data) == 0:
            raise ValueError("data list must be non-empty")
        base = []
        for value in data:
            if isinstance(value, bool) or not isinstance(value, float):
                raise TypeError("data elements must be floats")
            if not math.isfinite(value):
                raise ValueError("data elements must be finite")
            base.append(value)
        vector_input = True
    n = len(base)
    if isinstance(eps, bool) or not isinstance(eps, float):
        raise TypeError("eps must be a finite float")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")
    if isinstance(atol, bool) or not isinstance(atol, float):
        raise TypeError("atol must be a finite float")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a non-negative finite float")
    two_eps = 2.0 * eps
    if not math.isfinite(two_eps) or two_eps <= 0.0:
        raise ValueError("two_eps must be positive and finite")

    def check_output(result, expected_length):
        if not isinstance(result, Tensor):
            raise TypeError("fn must return a Tensor")
        output = result.data
        if not isinstance(output, list):
            raise TypeError("fn must return a 1D float list Tensor")
        if len(output) == 0:
            raise ValueError("fn must return a non-empty list")
        for value in output:
            if isinstance(value, bool) or not isinstance(value, float):
                raise TypeError("fn output elements must be floats")
        if not all(math.isfinite(v) for v in output):
            raise ValueError("fn output must be finite")
        if expected_length is not None and len(output) != expected_length:
            raise ValueError("fn output length must stay fixed")
        return output

    def check_input_grad(grad):
        if grad is None:
            raise TypeError("fn result is not connected to the input")
        if vector_input:
            if not isinstance(grad, list):
                raise TypeError("input grad must be a list of floats")
            if len(grad) != n:
                raise ValueError("input grad length must match the input")
            row = []
            for value in grad:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise TypeError("input grad elements must be floats")
                if not math.isfinite(value):
                    raise ValueError("input grad must be finite")
                row.append(value)
            return row
        if isinstance(grad, bool) or not isinstance(grad, float):
            raise TypeError("input grad must be a finite float")
        if not math.isfinite(grad):
            raise ValueError("input grad must be finite")
        return [grad]

    def analytic_row(r, expected_length):
        values = list(base) if vector_input else base[0]
        tensor = Tensor(values, True)
        result = fn(tensor)
        output = check_output(result, expected_length)
        if not result._parents:
            raise ValueError("fn result must be part of a graph")
        one_hot = [0.0] * len(output)
        one_hot[r] = 1.0
        result.backward(one_hot)
        row = check_input_grad(tensor.grad)
        return row, len(output)

    # The first call fixes the output length m and supplies row 0; each
    # remaining row rebuilds the graph from a fresh input Tensor.
    row0, m = analytic_row(0, None)
    analytic = [row0]
    for r in range(1, m):
        analytic.append(analytic_row(r, m)[0])

    # Central differences fill the numeric Jacobian one input column at
    # a time, using requires_grad False Tensors. Every perturbation,
    # difference and quotient must stay finite.
    numeric = [[0.0] * n for _ in range(m)]
    for c in range(n):
        plus = list(base)
        minus = list(base)
        plus[c] = base[c] + eps
        minus[c] = base[c] - eps
        if not math.isfinite(plus[c]) or not math.isfinite(minus[c]):
            raise ValueError("jacobiancheck perturbation must be finite")
        plus_arg = plus if vector_input else plus[0]
        minus_arg = minus if vector_input else minus[0]
        out_plus = check_output(fn(Tensor(plus_arg, False)), m)
        out_minus = check_output(fn(Tensor(minus_arg, False)), m)
        for r in range(m):
            difference = out_plus[r] - out_minus[r]
            if not math.isfinite(difference):
                raise ValueError("jacobiancheck difference must be finite")
            value = difference / two_eps
            if not math.isfinite(value):
                raise ValueError("jacobiancheck quotient must be finite")
            numeric[r][c] = value

    # Compare in row-major order so the reported maximum is deterministic.
    # error <= atol passes element by element; two exact 0.0 values give
    # error 0.0 and therefore always pass.
    max_abs_error = 0.0
    passed = True
    for r in range(m):
        for c in range(n):
            error = abs(analytic[r][c] - numeric[r][c])
            if not math.isfinite(error):
                raise ValueError("jacobiancheck absolute error must be finite")
            if error > max_abs_error:
                max_abs_error = error
            if error > atol:
                passed = False
    return (passed, max_abs_error)


def clip_grad_norm_(parameters, max_norm, eps=1e-12):
    """Clip the total gradient norm of a list of Tensors in place.

    Only Tensors with requires_grad True and a non-None grad participate.
    When the total norm exceeds max_norm every participating grad is
    scaled by max_norm / (norm + eps); otherwise grads are left as they
    are. Returns the total norm before clipping as a float, or 0.0 when
    no parameter participates. On any failure no Tensor is modified.
    """
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    if isinstance(max_norm, bool) or not isinstance(max_norm, float):
        raise TypeError("max_norm must be a positive finite float")
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be a positive finite float")
    if isinstance(eps, bool) or not isinstance(eps, float):
        raise TypeError("eps must be a positive finite float")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")
    for parameter in parameters:
        _validate_data(parameter.data)
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
    active = []
    for parameter in parameters:
        if not parameter.requires_grad or parameter.grad is None:
            continue
        Adam._check_grad(parameter.grad, parameter.data)
        active.append(parameter)
    if not active:
        return 0.0
    # Accumulate the squared norm from 0.0 in parameter order and
    # ascending vector-index order; a non-finite product, partial sum or
    # norm aborts before any grad is written.
    total = 0.0
    for parameter in active:
        grad = parameter.grad
        values = grad if isinstance(grad, list) else [grad]
        for value in values:
            product = value * value
            if not math.isfinite(product):
                raise ValueError(
                    "clip_grad_norm_ intermediate must be finite"
                )
            total += product
            if not math.isfinite(total):
                raise ValueError(
                    "clip_grad_norm_ intermediate must be finite"
                )
    norm = math.sqrt(total)
    if not math.isfinite(norm):
        raise ValueError("clip_grad_norm_ intermediate must be finite")
    if norm > max_norm:
        scale = max_norm / (norm + eps)
    else:
        scale = 1.0
    if not math.isfinite(scale):
        raise ValueError("clip_grad_norm_ intermediate must be finite")
    # Compute and validate every new grad before mutating anything, so a
    # non-finite intermediate aborts with all grads untouched.
    updates = []
    for parameter in active:
        grad = parameter.grad
        if isinstance(grad, list):
            new_grad = []
            for value in grad:
                scaled = value * scale
                if not math.isfinite(scaled):
                    raise ValueError(
                        "clip_grad_norm_ intermediate must be finite"
                    )
                new_grad.append(scaled)
        else:
            new_grad = grad * scale
            if not math.isfinite(new_grad):
                raise ValueError(
                    "clip_grad_norm_ intermediate must be finite"
                )
        updates.append((parameter, new_grad))
    for parameter, new_grad in updates:
        parameter.grad = new_grad
    return norm


_STATE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STATE_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}")


def _check_state_parameters(parameters):
    """Validate a dump_state/load_state parameters mapping."""
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dict of name -> Tensor")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    seen = set()
    for name, tensor in parameters.items():
        if not isinstance(name, str):
            raise TypeError("parameter names must be strings")
        if _STATE_NAME.fullmatch(name) is None:
            raise ValueError("invalid parameter name")
        if not isinstance(tensor, Tensor):
            raise TypeError("parameter values must be Tensors")
        if id(tensor) in seen:
            raise ValueError("parameters must not contain duplicate Tensors")
        seen.add(id(tensor))
        _validate_data(tensor.data)
        if not isinstance(tensor.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")


def dump_state(parameters):
    """Serialize named Tensors to a compact JSON state string.

    The output contains no whitespace and no trailing newline; floats are
    written with exactly six decimals and negative zero as 0.000000.
    """
    _check_state_parameters(parameters)
    entries = []
    for name, tensor in parameters.items():
        entries.append(
            '{"name":"'
            + name
            + '","data":'
            + _json_value(tensor.data)
            + ',"requires_grad":'
            + ("true" if tensor.requires_grad else "false")
            + "}"
        )
    return '{"parameters":[' + ",".join(entries) + "]}"


class _StateParser:
    """Strict parser for the exact textual form produced by dump_state."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_state format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_entry(self):
        self._expect('{"name":"')
        match = _STATE_NAME.match(self._text, self._pos)
        if match is None:
            self._fail()
        name = match.group(0)
        self._pos = match.end()
        self._expect('","data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (name, data, requires_grad)

    def parse(self):
        self._expect('{"parameters":[')
        entries = [self._parse_entry()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            entries.append(self._parse_entry())
        self._expect("]}")
        if self._pos != len(self._text):
            self._fail()
        return entries


def load_state(parameters, text):
    """Restore named Tensors from a string produced by dump_state.

    On full validation success, overwrite each Tensor's data and
    requires_grad and clear its grad; on any failure no Tensor is touched.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_state_parameters(parameters)
    entries = _StateParser(text).parse()
    if len(entries) != len(parameters):
        raise ValueError("state must contain exactly one entry per parameter")
    updates = []
    for (name, data, requires_grad), (expected_name, tensor) in zip(
        entries, parameters.items()
    ):
        if name != expected_name:
            raise ValueError("state names must match parameters in order")
        current = tensor.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match tensor shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match tensor shape")
        updates.append((tensor, data, requires_grad))
    for tensor, data, requires_grad in updates:
        tensor.data = data
        tensor.requires_grad = requires_grad
        tensor.grad = None
    return None


def _check_adam_state(optimizer):
    """Validate an Adam optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)
    for name, buffer in (("m", optimizer.m), ("v", optimizer.v)):
        if not isinstance(buffer, list):
            raise TypeError(name + " must be a list matching the parameters")
        if len(buffer) != len(parameters):
            raise ValueError(name + " must match the parameters in length")
        for item, parameter in zip(buffer, parameters):
            Adam._check_buffer(item, parameter.data, name)
    if isinstance(optimizer.t, bool) or not isinstance(optimizer.t, int):
        raise TypeError("t must be a non-bool non-negative int")
    if optimizer.t < 0:
        raise ValueError("t must be a non-bool non-negative int")


def dump_adam(optimizer):
    """Serialize an Adam optimizer's full state to a compact JSON string.

    The output contains no whitespace and no trailing newline; top-level
    keys are parameters, m, v, t in that order; floats are written with
    exactly six decimals and negative zero as 0.000000; t is a JSON
    integer.
    """
    if not isinstance(optimizer, Adam):
        raise TypeError("optimizer must be an Adam instance")
    _check_adam_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    moments = "[" + ",".join(_json_value(item) for item in optimizer.m) + "]"
    velocities = (
        "[" + ",".join(_json_value(item) for item in optimizer.v) + "]"
    )
    return (
        '{"parameters":['
        + ",".join(entries)
        + '],"m":'
        + moments
        + ',"v":'
        + velocities
        + ',"t":'
        + str(optimizer.t)
        + "}"
    )


_ADAM_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)")


class _AdamParser:
    """Strict parser for the exact textual form produced by dump_adam."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_adam format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def _parse_buffer(self):
        self._expect("[")
        items = [self._parse_data()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            items.append(self._parse_data())
        self._expect("]")
        return items

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect('],"m":')
        moments = self._parse_buffer()
        self._expect(',"v":')
        velocities = self._parse_buffer()
        self._expect(',"t":')
        match = _ADAM_INTEGER.match(self._text, self._pos)
        if match is None:
            self._fail()
        self._pos = match.end()
        t = int(match.group(0))
        self._expect("}")
        if self._pos != len(self._text):
            self._fail()
        return parameters, moments, velocities, t


def load_adam(optimizer, text):
    """Restore an Adam optimizer's full state from a dump_adam string.

    On full validation success, atomically replace every parameter's data
    and requires_grad, clear every grad, and replace m, v and t; lr,
    beta1, beta2 and eps are left unchanged. On any failure no state is
    touched.
    """
    if not isinstance(optimizer, Adam):
        raise TypeError("optimizer must be an Adam instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_adam_state(optimizer)
    parameters, moments, velocities, t = _AdamParser(text).parse()
    count = len(optimizer.parameters)
    if len(parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(moments) != count or len(velocities) != count:
        raise ValueError("state m and v must match the parameters in length")
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for name, buffers in (("m", moments), ("v", velocities)):
        for buffer, parameter in zip(buffers, optimizer.parameters):
            current = parameter.data
            if isinstance(current, list):
                if not isinstance(buffer, list) or len(buffer) != len(current):
                    raise ValueError(
                        "state " + name + " shape must match parameter shape"
                    )
            elif isinstance(buffer, list):
                raise ValueError(
                    "state " + name + " shape must match parameter shape"
                )
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.m = moments
    optimizer.v = velocities
    optimizer.t = t
    return None


def _check_adamw_state(optimizer):
    """Validate an AdamW optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)
    for name, buffer in (("m", optimizer.m), ("v", optimizer.v)):
        if not isinstance(buffer, list):
            raise TypeError(name + " must be a list matching the parameters")
        if len(buffer) != len(parameters):
            raise ValueError(name + " must match the parameters in length")
        for item, parameter in zip(buffer, parameters):
            AdamW._check_buffer(item, parameter.data, name)
    if isinstance(optimizer.t, bool) or not isinstance(optimizer.t, int):
        raise TypeError("t must be a non-bool non-negative int")
    if optimizer.t < 0:
        raise ValueError("t must be a non-bool non-negative int")


def dump_adamw(optimizer):
    """Serialize an AdamW optimizer's full state to a compact JSON string.

    The output contains no whitespace and no trailing newline; top-level
    keys are parameters, m, v, t in that order; floats are written with
    exactly six decimals and negative zero as 0.000000; t is a JSON
    integer.
    """
    if not isinstance(optimizer, AdamW):
        raise TypeError("optimizer must be an AdamW instance")
    _check_adamw_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    moments = "[" + ",".join(_json_value(item) for item in optimizer.m) + "]"
    velocities = (
        "[" + ",".join(_json_value(item) for item in optimizer.v) + "]"
    )
    return (
        '{"parameters":['
        + ",".join(entries)
        + '],"m":'
        + moments
        + ',"v":'
        + velocities
        + ',"t":'
        + str(optimizer.t)
        + "}"
    )


_ADAMW_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)")


class _AdamWParser:
    """Strict parser for the exact textual form produced by dump_adamw."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_adamw format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def _parse_buffer(self):
        self._expect("[")
        items = [self._parse_data()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            items.append(self._parse_data())
        self._expect("]")
        return items

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect('],"m":')
        moments = self._parse_buffer()
        self._expect(',"v":')
        velocities = self._parse_buffer()
        self._expect(',"t":')
        match = _ADAMW_INTEGER.match(self._text, self._pos)
        if match is None:
            self._fail()
        self._pos = match.end()
        t = int(match.group(0))
        self._expect("}")
        if self._pos != len(self._text):
            self._fail()
        return parameters, moments, velocities, t


def load_adamw(optimizer, text):
    """Restore an AdamW optimizer's full state from a dump_adamw string.

    On full validation success, atomically replace every parameter's data
    and requires_grad, clear every grad, and replace m, v and t; lr,
    beta1, beta2, eps and weight_decay are left unchanged. On any failure
    no state is touched.
    """
    if not isinstance(optimizer, AdamW):
        raise TypeError("optimizer must be an AdamW instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_adamw_state(optimizer)
    parameters, moments, velocities, t = _AdamWParser(text).parse()
    count = len(optimizer.parameters)
    if len(parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(moments) != count or len(velocities) != count:
        raise ValueError("state m and v must match the parameters in length")
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for name, buffers in (("m", moments), ("v", velocities)):
        for buffer, parameter in zip(buffers, optimizer.parameters):
            current = parameter.data
            if isinstance(current, list):
                if not isinstance(buffer, list) or len(buffer) != len(current):
                    raise ValueError(
                        "state " + name + " shape must match parameter shape"
                    )
            elif isinstance(buffer, list):
                raise ValueError(
                    "state " + name + " shape must match parameter shape"
                )
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.m = moments
    optimizer.v = velocities
    optimizer.t = t
    return None


def _check_rmsprop_state(optimizer):
    """Validate an RMSprop optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)
    square_avg = optimizer.square_avg
    if not isinstance(square_avg, list):
        raise TypeError("square_avg must be a list matching the parameters")
    if len(square_avg) != len(parameters):
        raise ValueError("square_avg must match the parameters in length")
    for slot, parameter in zip(square_avg, parameters):
        RMSprop._check_slot(slot, parameter.data)


def _check_rmsprop_hyperparameters(optimizer):
    """Validate an RMSprop optimizer's lr, alpha and eps for dump/load."""
    lr = optimizer.lr
    if isinstance(lr, bool) or not isinstance(lr, float):
        raise TypeError("lr must be a positive finite float")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be a positive finite float")
    alpha = optimizer.alpha
    if isinstance(alpha, bool) or not isinstance(alpha, float):
        raise TypeError("alpha must be a finite float in [0.0, 1.0)")
    if not math.isfinite(alpha) or alpha < 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be a finite float in [0.0, 1.0)")
    eps = optimizer.eps
    if isinstance(eps, bool) or not isinstance(eps, float):
        raise TypeError("eps must be a positive finite float")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")


def dump_rmsprop(optimizer):
    """Serialize an RMSprop optimizer's full state to a compact JSON string.

    The output contains no whitespace and no trailing newline; top-level
    keys are parameters, square_avg in that order; each parameter entry has
    data then requires_grad; square_avg entries are a scalar or an array.
    Floats are written with exactly six decimals and negative zero as
    0.000000.
    """
    if not isinstance(optimizer, RMSprop):
        raise TypeError("optimizer must be an RMSprop instance")
    _check_rmsprop_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    slots = (
        "["
        + ",".join(_json_value(item) for item in optimizer.square_avg)
        + "]"
    )
    return (
        '{"parameters":['
        + ",".join(entries)
        + '],"square_avg":'
        + slots
        + "}"
    )


class _RMSpropParser:
    """Strict parser for the exact textual form produced by dump_rmsprop."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_rmsprop format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def _parse_slots(self):
        self._expect("[")
        items = [self._parse_data()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            items.append(self._parse_data())
        self._expect("]")
        return items

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect('],"square_avg":')
        square_avg = self._parse_slots()
        self._expect("}")
        if self._pos != len(self._text):
            self._fail()
        return parameters, square_avg


def load_rmsprop(optimizer, text):
    """Restore an RMSprop optimizer's full state from a dump_rmsprop string.

    On full validation success, atomically replace every parameter's data
    and requires_grad, clear every grad, and replace square_avg; lr, alpha,
    eps and the parameter identities are left unchanged. On any failure no
    state is touched.
    """
    if not isinstance(optimizer, RMSprop):
        raise TypeError("optimizer must be an RMSprop instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_rmsprop_state(optimizer)
    parameters, square_avg = _RMSpropParser(text).parse()
    count = len(optimizer.parameters)
    if len(parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(square_avg) != count:
        raise ValueError("state square_avg must match the parameters in length")
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for slot, parameter in zip(square_avg, optimizer.parameters):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(slot, list) or len(slot) != len(current):
                raise ValueError(
                    "state square_avg shape must match parameter shape"
                )
        elif isinstance(slot, list):
            raise ValueError("state square_avg shape must match parameter shape")
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.square_avg = square_avg
    return None


def _check_adamax_state(optimizer):
    """Validate an Adamax optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)
    if not isinstance(optimizer.m, list):
        raise TypeError("m must be a list matching the parameters")
    if len(optimizer.m) != len(parameters):
        raise ValueError("m must match the parameters in length")
    if not isinstance(optimizer.u, list):
        raise TypeError("u must be a list matching the parameters")
    if len(optimizer.u) != len(parameters):
        raise ValueError("u must match the parameters in length")
    for item, parameter in zip(optimizer.m, parameters):
        Adamax._check_buffer(item, parameter.data, "m")
    for item, parameter in zip(optimizer.u, parameters):
        Adamax._check_buffer(item, parameter.data, "u", non_negative=True)
    if isinstance(optimizer.t, bool) or not isinstance(optimizer.t, int):
        raise TypeError("t must be a non-bool non-negative int")
    if optimizer.t < 0:
        raise ValueError("t must be a non-bool non-negative int")


def dump_adamax(optimizer):
    """Serialize an Adamax optimizer's full state to a compact JSON string.

    The output contains no whitespace and no trailing newline; top-level
    keys are parameters, m, u, t in that order; floats are written with
    exactly six decimals and negative zero as 0.000000; t is a JSON
    integer.
    """
    if not isinstance(optimizer, Adamax):
        raise TypeError("optimizer must be an Adamax instance")
    _check_adamax_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    moments = "[" + ",".join(_json_value(item) for item in optimizer.m) + "]"
    infinity_norms = (
        "[" + ",".join(_json_value(item) for item in optimizer.u) + "]"
    )
    return (
        '{"parameters":['
        + ",".join(entries)
        + '],"m":'
        + moments
        + ',"u":'
        + infinity_norms
        + ',"t":'
        + str(optimizer.t)
        + "}"
    )


_ADAMAX_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)")


class _AdamaxParser:
    """Strict parser for the exact textual form produced by dump_adamax."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_adamax format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def _parse_buffer(self):
        self._expect("[")
        items = [self._parse_data()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            items.append(self._parse_data())
        self._expect("]")
        return items

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect('],"m":')
        moments = self._parse_buffer()
        self._expect(',"u":')
        infinity_norms = self._parse_buffer()
        self._expect(',"t":')
        match = _ADAMAX_INTEGER.match(self._text, self._pos)
        if match is None:
            self._fail()
        self._pos = match.end()
        t = int(match.group(0))
        self._expect("}")
        if self._pos != len(self._text):
            self._fail()
        return parameters, moments, infinity_norms, t


def load_adamax(optimizer, text):
    """Restore an Adamax optimizer's full state from a dump_adamax string.

    On full validation success, atomically replace every parameter's data
    and requires_grad, clear every grad, and replace m, u and t; lr,
    beta1, beta2, eps and the parameter identities are left unchanged. On
    any failure no state is touched.
    """
    if not isinstance(optimizer, Adamax):
        raise TypeError("optimizer must be an Adamax instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_adamax_state(optimizer)
    parameters, moments, infinity_norms, t = _AdamaxParser(text).parse()
    for buffer in infinity_norms:
        values = buffer if isinstance(buffer, list) else [buffer]
        for value in values:
            if value < 0.0:
                raise ValueError("state u must be non-negative")
    count = len(optimizer.parameters)
    if len(parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(moments) != count or len(infinity_norms) != count:
        raise ValueError("state m and u must match the parameters in length")
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for name, buffers in (("m", moments), ("u", infinity_norms)):
        for buffer, parameter in zip(buffers, optimizer.parameters):
            current = parameter.data
            if isinstance(current, list):
                if not isinstance(buffer, list) or len(buffer) != len(current):
                    raise ValueError(
                        "state " + name + " shape must match parameter shape"
                    )
            elif isinstance(buffer, list):
                raise ValueError(
                    "state " + name + " shape must match parameter shape"
                )
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.m = moments
    optimizer.u = infinity_norms
    optimizer.t = t
    return None


def _check_amsgrad_state(optimizer):
    """Validate an AMSGrad optimizer's full mutable state for dump/load."""
    parameters = optimizer.parameters
    if not isinstance(parameters, list):
        raise TypeError("parameters must be a non-empty list of Tensors")
    if len(parameters) == 0:
        raise ValueError("parameters must be non-empty")
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("parameters must not contain duplicate Tensors")
    for parameter in parameters:
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        _validate_data(parameter.data)
    for name, buffer in (
        ("m", optimizer.m),
        ("v", optimizer.v),
        ("v_max", optimizer.v_max),
    ):
        if not isinstance(buffer, list):
            raise TypeError(name + " must be a list matching the parameters")
        if len(buffer) != len(parameters):
            raise ValueError(name + " must match the parameters in length")
    for index, parameter in enumerate(parameters):
        data = _validate_data(parameter.data)
        AMSGrad._check_buffer(optimizer.m[index], data, "m")
        AMSGrad._check_buffer(
            optimizer.v[index], data, "v", non_negative=True
        )
        AMSGrad._check_buffer(
            optimizer.v_max[index], data, "v_max", non_negative=True
        )
        AMSGrad._check_max_ge(optimizer.v_max[index], optimizer.v[index])
    if isinstance(optimizer.t, bool) or not isinstance(optimizer.t, int):
        raise TypeError("t must be a non-bool non-negative int")
    if optimizer.t < 0:
        raise ValueError("t must be a non-bool non-negative int")


def dump_amsgrad(optimizer):
    """Serialize an AMSGrad optimizer's full state to a compact JSON string.

    The output contains no whitespace and no trailing newline; top-level
    keys are parameters, m, v, v_max, t in that order; each parameter
    entry has data then requires_grad; the three buffers keep each
    parameter's scalar/array shape. Floats are written with exactly six
    decimals and negative zero as 0.000000; t is a JSON integer. lr,
    beta1, beta2 and eps are not serialized.
    """
    if not isinstance(optimizer, AMSGrad):
        raise TypeError("optimizer must be an AMSGrad instance")
    _check_amsgrad_state(optimizer)
    entries = []
    for parameter in optimizer.parameters:
        entries.append(
            '{"data":'
            + _json_value(parameter.data)
            + ',"requires_grad":'
            + ("true" if parameter.requires_grad else "false")
            + "}"
        )
    moments = "[" + ",".join(_json_value(item) for item in optimizer.m) + "]"
    velocities = (
        "[" + ",".join(_json_value(item) for item in optimizer.v) + "]"
    )
    max_velocities = (
        "[" + ",".join(_json_value(item) for item in optimizer.v_max) + "]"
    )
    return (
        '{"parameters":['
        + ",".join(entries)
        + '],"m":'
        + moments
        + ',"v":'
        + velocities
        + ',"v_max":'
        + max_velocities
        + ',"t":'
        + str(optimizer.t)
        + "}"
    )


_AMSGRAD_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)")


class _AMSGradParser:
    """Strict parser for the exact textual form produced by dump_amsgrad."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_amsgrad format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_data(self):
        if not self._text.startswith("[", self._pos):
            return self._parse_number()
        self._pos += 1
        values = [self._parse_number()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            values.append(self._parse_number())
        self._expect("]")
        return values

    def _parse_parameter(self):
        self._expect('{"data":')
        data = self._parse_data()
        self._expect(',"requires_grad":')
        if self._text.startswith("true", self._pos):
            self._pos += 4
            requires_grad = True
        elif self._text.startswith("false", self._pos):
            self._pos += 5
            requires_grad = False
        else:
            self._fail()
        self._expect("}")
        return (data, requires_grad)

    def _parse_buffer(self):
        self._expect("[")
        items = [self._parse_data()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            items.append(self._parse_data())
        self._expect("]")
        return items

    def parse(self):
        self._expect('{"parameters":[')
        parameters = [self._parse_parameter()]
        while self._text.startswith(",", self._pos):
            self._pos += 1
            parameters.append(self._parse_parameter())
        self._expect('],"m":')
        moments = self._parse_buffer()
        self._expect(',"v":')
        velocities = self._parse_buffer()
        self._expect(',"v_max":')
        max_velocities = self._parse_buffer()
        self._expect(',"t":')
        match = _AMSGRAD_INTEGER.match(self._text, self._pos)
        if match is None:
            self._fail()
        self._pos = match.end()
        t = int(match.group(0))
        self._expect("}")
        if self._pos != len(self._text):
            self._fail()
        return parameters, moments, velocities, max_velocities, t


def load_amsgrad(optimizer, text):
    """Restore an AMSGrad optimizer's full state from a dump_amsgrad string.

    On full validation success, atomically replace every parameter's data
    and requires_grad, clear every grad, and replace m, v, v_max and t;
    lr, beta1, beta2, eps and the parameter identities are left
    unchanged. On any failure no state is touched.
    """
    if not isinstance(optimizer, AMSGrad):
        raise TypeError("optimizer must be an AMSGrad instance")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _check_amsgrad_state(optimizer)
    parameters, moments, velocities, max_velocities, t = (
        _AMSGradParser(text).parse()
    )
    count = len(optimizer.parameters)
    if len(parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if (
        len(moments) != count
        or len(velocities) != count
        or len(max_velocities) != count
    ):
        raise ValueError(
            "state m, v and v_max must match the parameters in length"
        )
    updates = []
    for (data, requires_grad), parameter in zip(
        parameters, optimizer.parameters
    ):
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError("state data shape must match parameter shape")
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
    for name, buffers in (
        ("m", moments),
        ("v", velocities),
        ("v_max", max_velocities),
    ):
        for buffer, parameter in zip(buffers, optimizer.parameters):
            current = parameter.data
            if isinstance(current, list):
                if not isinstance(buffer, list) or len(buffer) != len(current):
                    raise ValueError(
                        "state " + name + " shape must match parameter shape"
                    )
            elif isinstance(buffer, list):
                raise ValueError(
                    "state " + name + " shape must match parameter shape"
                )
    for velocity, maximum in zip(velocities, max_velocities):
        if isinstance(velocity, list):
            v_values = velocity
            h_values = maximum
        else:
            v_values = [velocity]
            h_values = [maximum]
        for value, top in zip(v_values, h_values):
            if value < 0.0:
                raise ValueError("state v must be non-negative")
            if top < 0.0:
                raise ValueError("state v_max must be non-negative")
            if top < value:
                raise ValueError("state v_max must be elementwise >= v")
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.m = moments
    optimizer.v = velocities
    optimizer.v_max = max_velocities
    optimizer.t = t
    return None


_CHECKPOINT_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)")
_CHECKPOINT_DIGEST = re.compile(r',"digest":"([0-9a-f]{64})"\}\Z')


def dump_checkpoint(parameters, optimizer):
    """Serialize named parameters and a supported optimizer.

    Adam, AdamW, Adamax, Adagrad, AMSGrad, MomentumSGD and SGD are
    supported. The parameters mapping is validated with the dump_state
    contract and its values must be the optimizer's parameters itemwise
    in the same order. The output contains no whitespace and no trailing
    newline.

    For AMSGrad the version-1 form is emitted: top-level keys are
    version, names and amsgrad in that order; version is the integer 1,
    names is the parameter-name array and amsgrad is exactly the object
    produced by dump_amsgrad, keeping its six-decimal byte-level format.

    For Adam the version-2 form is emitted: top-level keys are version,
    type, names and state in that order; version is the integer 2, type
    is the string "adam", names is the parameter-name array and state is
    exactly the object produced by dump_adam, keeping its six-decimal
    byte-level format.

    For AdamW the version-3 form is emitted: top-level keys are version,
    type, names and state in that order; version is the integer 3, type
    is the string "adamw", names is the parameter-name array and state
    is exactly the object produced by dump_adamw, keeping its
    six-decimal byte-level format.

    For Adamax the version-4 form is emitted: top-level keys are
    version, type, names and state in that order; version is the integer
    4, type is the string "adamax", names is the parameter-name array
    and state is exactly the object produced by dump_adamax (keys
    parameters, m, u, t; each parameter entry has data then
    requires_grad), with every array ordered by names.

    For SGD the version-5 form is emitted: top-level keys are version,
    type, names and state in that order; version is the integer 5, type
    is the string "sgd", names is the parameter-name array and state is
    exactly the object produced by dump_sgd (its single key parameters;
    each entry has data then requires_grad), ordered by names.

    For MomentumSGD the version-6 form is emitted: top-level keys are
    version, type, names and state in that order; version is the integer
    6, type is the string "momentum_sgd", names is the parameter-name
    array and state is exactly the object produced by dump_momentum_sgd
    (keys parameters, velocity; each parameter entry has data then
    requires_grad), with every array ordered by names.

    For Adagrad the version-7 form is emitted: top-level keys are
    version, type, names and state in that order; version is the integer
    7, type is the string "adagrad", names is the parameter-name array
    and state is exactly the object produced by dump_adagrad (keys
    parameters, sum_sq; each parameter entry has data then
    requires_grad), with every array ordered by names.

    For RMSprop the version-8 form is emitted: a payload object whose
    top-level keys are version, type, names, hyperparameters and state
    in that order; version is the integer 8, type is the string
    "rmsprop", names is the parameter-name array, hyperparameters is an
    object with keys lr, alpha, eps in that order and state is exactly
    the object produced by dump_rmsprop (keys parameters, square_avg;
    each parameter entry has data then requires_grad), with every array
    ordered by names. The payload carries no trailing newline; the final
    text replaces the payload's closing brace with
    ,"digest":"H"} where H is the lowercase 64-character hexadecimal
    SHA-256 of the payload's UTF-8 bytes.

    For Adadelta the version-9 form is emitted: top-level keys are
    version, type, names and state in that order; version is the integer
    9, type is the string "adadelta", names is the parameter-name array
    and state is exactly the object produced by dump_adadelta (keys
    parameters, square_avg, acc_delta; each parameter entry has data
    then requires_grad), with every array ordered by names. rho and eps
    are not serialized.
    """
    if not isinstance(
        optimizer,
        (
            Adam,
            AdamW,
            Adamax,
            Adadelta,
            Adagrad,
            AMSGrad,
            MomentumSGD,
            RMSprop,
            SGD,
        ),
    ):
        raise TypeError(
            "optimizer must be an Adam, AdamW, Adamax, Adadelta, Adagrad,"
            " AMSGrad, MomentumSGD, RMSprop or SGD instance"
        )
    optimizer_parameters = optimizer.parameters
    if not isinstance(optimizer_parameters, list):
        raise TypeError("optimizer parameters must be a list")
    _check_state_parameters(parameters)
    if len(parameters) != len(optimizer_parameters):
        raise ValueError("parameters must match the optimizer parameters")
    for (_, tensor), parameter in zip(
        parameters.items(), optimizer_parameters
    ):
        if tensor is not parameter:
            raise ValueError(
                "parameter values must be the optimizer parameters in order"
            )
    names = (
        "["
        + ",".join('"' + name + '"' for name, _ in parameters.items())
        + "]"
    )
    if isinstance(optimizer, SGD):
        return (
            '{"version":5,"type":"sgd","names":'
            + names
            + ',"state":'
            + dump_sgd(optimizer)
            + "}"
        )
    if isinstance(optimizer, MomentumSGD):
        return (
            '{"version":6,"type":"momentum_sgd","names":'
            + names
            + ',"state":'
            + dump_momentum_sgd(optimizer)
            + "}"
        )
    if isinstance(optimizer, Adagrad):
        return (
            '{"version":7,"type":"adagrad","names":'
            + names
            + ',"state":'
            + dump_adagrad(optimizer)
            + "}"
        )
    if isinstance(optimizer, Adadelta):
        return (
            '{"version":9,"type":"adadelta","names":'
            + names
            + ',"state":'
            + dump_adadelta(optimizer)
            + "}"
        )
    if isinstance(optimizer, RMSprop):
        _check_rmsprop_hyperparameters(optimizer)
        payload = (
            '{"version":8,"type":"rmsprop","names":'
            + names
            + ',"hyperparameters":{"lr":'
            + _format_float(optimizer.lr)
            + ',"alpha":'
            + _format_float(optimizer.alpha)
            + ',"eps":'
            + _format_float(optimizer.eps)
            + '},"state":'
            + dump_rmsprop(optimizer)
            + "}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return payload[:-1] + ',"digest":"' + digest + '"}'
    if isinstance(optimizer, AMSGrad):
        return (
            '{"version":1,"names":'
            + names
            + ',"amsgrad":'
            + dump_amsgrad(optimizer)
            + "}"
        )
    if isinstance(optimizer, AdamW):
        return (
            '{"version":3,"type":"adamw","names":'
            + names
            + ',"state":'
            + dump_adamw(optimizer)
            + "}"
        )
    if isinstance(optimizer, Adamax):
        return (
            '{"version":4,"type":"adamax","names":'
            + names
            + ',"state":'
            + dump_adamax(optimizer)
            + "}"
        )
    return (
        '{"version":2,"type":"adam","names":'
        + names
        + ',"state":'
        + dump_adam(optimizer)
        + "}"
    )


class _CheckpointParser:
    """Strict parser for the exact textual forms produced by dump_checkpoint."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_checkpoint format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_name(self):
        match = _STATE_NAME.match(self._text, self._pos)
        if match is None:
            self._fail()
        name = match.group(0)
        self._pos = match.end()
        return name

    def _parse_names(self):
        self._expect("[")
        names = []
        if not self._text.startswith("]", self._pos):
            self._expect('"')
            names.append(self._parse_name())
            self._expect('"')
            while self._text.startswith(",", self._pos):
                self._pos += 1
                self._expect('"')
                names.append(self._parse_name())
                self._expect('"')
        self._expect("]")
        return names

    def _parse_state(self, state_parser):
        # The state object is the final value, immediately followed by
        # the checkpoint's own single closing brace. Delegate its exact
        # format (including duplicate/missing/extra/reordered inner keys)
        # to the optimizer parser, which rejects malformed or trailing
        # bytes.
        if not self._text.endswith("}"):
            self._fail()
        state_text = self._text[self._pos:-1]
        state_parser(state_text).parse()
        return state_text

    def _parse_number(self):
        match = _STATE_NUMBER.match(self._text, self._pos)
        if match is None:
            self._fail()
        token = match.group(0)
        self._pos = match.end()
        value = float(token)
        if not math.isfinite(value) or (value == 0.0 and token[0] == "-"):
            self._fail()
        return value

    def _parse_state_digest(self, state_parser):
        # Version 8 appends ,"digest":"H"} after the state object, where
        # H is the lowercase hexadecimal SHA-256 of the payload: the
        # full text with the digest member replaced by the payload's own
        # closing brace. The digest must verify before the state is
        # accepted; the state text itself is validated by the optimizer
        # parser.
        match = _CHECKPOINT_DIGEST.search(self._text)
        if match is None:
            self._fail()
        payload = self._text[: match.start()] + "}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if digest != match.group(1):
            self._fail()
        state_text = self._text[self._pos : match.start()]
        state_parser(state_text).parse()
        return state_text

    def parse(self):
        self._expect('{"version":')
        match = _CHECKPOINT_INTEGER.match(self._text, self._pos)
        if match is None:
            self._fail()
        self._pos = match.end()
        version = int(match.group(0))
        hyperparameters = None
        if version == 1:
            self._expect(',"names":')
            names = self._parse_names()
            self._expect(',"amsgrad":')
            kind = "amsgrad"
            state_text = self._parse_state(_AMSGradParser)
        elif version == 2:
            self._expect(',"type":"adam","names":')
            names = self._parse_names()
            self._expect(',"state":')
            kind = "adam"
            state_text = self._parse_state(_AdamParser)
        elif version == 3:
            self._expect(',"type":"adamw","names":')
            names = self._parse_names()
            self._expect(',"state":')
            kind = "adamw"
            state_text = self._parse_state(_AdamWParser)
        elif version == 4:
            self._expect(',"type":"adamax","names":')
            names = self._parse_names()
            self._expect(',"state":')
            kind = "adamax"
            state_text = self._parse_state(_AdamaxParser)
        elif version == 5:
            self._expect(',"type":"sgd","names":')
            names = self._parse_names()
            self._expect(',"state":')
            kind = "sgd"
            state_text = self._parse_state(_SGDParser)
        elif version == 6:
            self._expect(',"type":"momentum_sgd","names":')
            names = self._parse_names()
            self._expect(',"state":')
            kind = "momentum_sgd"
            state_text = self._parse_state(_MomentumSGDParser)
        elif version == 7:
            self._expect(',"type":"adagrad","names":')
            names = self._parse_names()
            self._expect(',"state":')
            kind = "adagrad"
            state_text = self._parse_state(_AdagradParser)
        elif version == 9:
            self._expect(',"type":"adadelta","names":')
            names = self._parse_names()
            self._expect(',"state":')
            kind = "adadelta"
            state_text = self._parse_state(_AdadeltaParser)
        elif version == 8:
            self._expect(',"type":"rmsprop","names":')
            names = self._parse_names()
            self._expect(',"hyperparameters":{"lr":')
            lr = self._parse_number()
            self._expect(',"alpha":')
            alpha = self._parse_number()
            self._expect(',"eps":')
            eps = self._parse_number()
            self._expect('},"state":')
            kind = "rmsprop"
            hyperparameters = (lr, alpha, eps)
            state_text = self._parse_state_digest(_RMSpropParser)
        else:
            raise ValueError("unsupported checkpoint version")
        if not names:
            raise ValueError("checkpoint names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("checkpoint names must not be duplicated")
        return kind, names, state_text, hyperparameters


def load_checkpoint(parameters, optimizer, text):
    """Restore named parameters and a supported optimizer.

    The parameters mapping is validated with the dump_state contract and
    its values must be the optimizer's parameters itemwise in the same
    order. The exact version-1 (AMSGrad), version-2 (Adam),
    version-3 (AdamW), version-4 (Adamax), version-5 (SGD),
    version-6 (MomentumSGD), version-7 (Adagrad), version-8 (RMSprop)
    and version-9 (Adadelta) forms produced by dump_checkpoint are
    accepted, and the checkpoint type must match the target optimizer.

    Versions 1-3 and 5-7 require the checkpoint names to match the
    target parameter names in order and behave exactly like
    load_amsgrad, load_adam, load_adamw, load_sgd, load_momentum_sgd or
    load_adagrad for the embedded state.

    Version 9 requires the checkpoint names to match the target
    parameter names in order and behaves exactly like load_adadelta for
    the embedded state; rho and eps are left unchanged.

    Version 8 requires the checkpoint names to match the target
    parameter names in order and its digest to verify against the
    payload; its hyperparameters must satisfy lr > 0, 0 <= alpha < 1 and
    eps > 0. On full validation success it behaves exactly like
    load_rmsprop for the embedded state and additionally replaces lr,
    alpha and eps with the checkpoint values.

    Version 4 binds state positions by name: its names array must name
    exactly the same parameter set as the target (duplicates rejected),
    but may be in any order; the parameters, m and u entries are aligned
    to the target/optimizer order before being applied. On full
    validation success, atomically replace data, requires_grad, m, u and
    t, clear every grad, and keep object identities and
    hyperparameters; return None. On any failure no state is touched.
    """
    if not isinstance(
        optimizer,
        (
            Adam,
            AdamW,
            Adamax,
            Adadelta,
            Adagrad,
            AMSGrad,
            MomentumSGD,
            RMSprop,
            SGD,
        ),
    ):
        raise TypeError(
            "optimizer must be an Adam, AdamW, Adamax, Adadelta, Adagrad,"
            " AMSGrad, MomentumSGD, RMSprop or SGD instance"
        )
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    optimizer_parameters = optimizer.parameters
    if not isinstance(optimizer_parameters, list):
        raise TypeError("optimizer parameters must be a list")
    _check_state_parameters(parameters)
    if len(parameters) != len(optimizer_parameters):
        raise ValueError("parameters must match the optimizer parameters")
    for (_, tensor), parameter in zip(
        parameters.items(), optimizer_parameters
    ):
        if tensor is not parameter:
            raise ValueError(
                "parameter values must be the optimizer parameters in order"
            )
    kind, names, state_text, hyperparameters = _CheckpointParser(
        text
    ).parse()
    if kind == "amsgrad":
        if not isinstance(optimizer, AMSGrad):
            raise ValueError("checkpoint type must match the optimizer")
    elif kind == "adamw":
        if not isinstance(optimizer, AdamW):
            raise ValueError("checkpoint type must match the optimizer")
    elif kind == "adamax":
        if not isinstance(optimizer, Adamax):
            raise ValueError("checkpoint type must match the optimizer")
    elif kind == "sgd":
        if not isinstance(optimizer, SGD):
            raise ValueError("checkpoint type must match the optimizer")
    elif kind == "momentum_sgd":
        if not isinstance(optimizer, MomentumSGD):
            raise ValueError("checkpoint type must match the optimizer")
    elif kind == "adagrad":
        if not isinstance(optimizer, Adagrad):
            raise ValueError("checkpoint type must match the optimizer")
    elif kind == "adadelta":
        if not isinstance(optimizer, Adadelta):
            raise ValueError("checkpoint type must match the optimizer")
    elif kind == "rmsprop":
        if not isinstance(optimizer, RMSprop):
            raise ValueError("checkpoint type must match the optimizer")
    elif not isinstance(optimizer, Adam):
        raise ValueError("checkpoint type must match the optimizer")
    target_names = list(parameters.keys())
    if kind == "adamax":
        # Version 4 binds by name and allows the checkpoint order to
        # differ from the target parameters/optimizer order.
        if set(names) != set(target_names) or len(names) != len(target_names):
            raise ValueError(
                "checkpoint names must name exactly the target parameters"
            )
    elif names != target_names:
        raise ValueError("checkpoint names must match parameters in order")
    if kind == "amsgrad":
        load_amsgrad(optimizer, state_text)
    elif kind == "adamw":
        load_adamw(optimizer, state_text)
    elif kind == "adamax":
        _load_checkpoint_adamax(parameters, optimizer, state_text, names)
    elif kind == "sgd":
        load_sgd(optimizer, state_text)
    elif kind == "momentum_sgd":
        load_momentum_sgd(optimizer, state_text)
    elif kind == "adagrad":
        load_adagrad(optimizer, state_text)
    elif kind == "adadelta":
        load_adadelta(optimizer, state_text)
    elif kind == "rmsprop":
        lr, alpha, eps = hyperparameters
        if lr <= 0.0:
            raise ValueError("checkpoint lr must be positive")
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError("checkpoint alpha must be in [0.0, 1.0)")
        if eps <= 0.0:
            raise ValueError("checkpoint eps must be positive")
        load_rmsprop(optimizer, state_text)
        optimizer.lr = lr
        optimizer.alpha = alpha
        optimizer.eps = eps
    else:
        load_adam(optimizer, state_text)
    return None


def _load_checkpoint_adamax(parameters, optimizer, text, names):
    """Apply a version-4 Adamax state, binding its entries by name.

    Validates the whole embedded state and its alignment to the target
    parameters before touching anything; the state's arrays follow the
    checkpoint names order and are rearranged into optimizer order.
    """
    _check_adamax_state(optimizer)
    state_parameters, moments, infinity_norms, t = _AdamaxParser(text).parse()
    for buffer in infinity_norms:
        values = buffer if isinstance(buffer, list) else [buffer]
        for value in values:
            if value < 0.0:
                raise ValueError("state u must be non-negative")
    count = len(optimizer.parameters)
    if len(state_parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(moments) != count or len(infinity_norms) != count:
        raise ValueError("state m and u must match the parameters in length")
    by_name = {
        name: (spec, moment, infinity_norm)
        for name, spec, moment, infinity_norm in zip(
            names, state_parameters, moments, infinity_norms
        )
    }
    updates = []
    ordered_moments = []
    ordered_infinity_norms = []
    for name, parameter in zip(parameters.keys(), optimizer.parameters):
        data, requires_grad = by_name[name][0]
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError(
                    "state data shape must match parameter shape"
                )
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
        ordered_moments.append(by_name[name][1])
        ordered_infinity_norms.append(by_name[name][2])
    for slot_name, buffers in (
        ("m", ordered_moments),
        ("u", ordered_infinity_norms),
    ):
        for buffer, parameter in zip(buffers, optimizer.parameters):
            current = parameter.data
            if isinstance(current, list):
                if not isinstance(buffer, list) or len(buffer) != len(current):
                    raise ValueError(
                        "state " + slot_name
                        + " shape must match parameter shape"
                    )
            elif isinstance(buffer, list):
                raise ValueError(
                    "state " + slot_name
                    + " shape must match parameter shape"
                )
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.m = ordered_moments
    optimizer.u = ordered_infinity_norms
    optimizer.t = t
    return None


def dump_training_state(parameters, optimizer, global_step, rng_state):
    """Serialize named parameters, a supported optimizer and loop state.

    RMSprop, Adamax, Adam and AdamW are supported; any other optimizer
    raises TypeError, and so do non-bool loop values of the wrong type.
    The parameters mapping and the optimizer state follow the
    dump_checkpoint contract. For Adamax and Adam, lr, beta1, beta2 and
    eps are validated exactly as the constructor validates them, and for
    AdamW weight_decay is validated the same way: a wrong type raises
    TypeError and an illegal value raises ValueError. global_step must be
    a non-bool non-negative int and rng_state a non-bool int in
    0..4294967295; an out-of-range value raises ValueError.

    The output contains no whitespace and no trailing newline; top-level
    keys are version, global_step, rng_state and checkpoint in that
    order, where global_step and rng_state are the two loop-state
    integers and checkpoint is exactly the JSON object produced by
    dump_checkpoint, keeping its inner bytes unchanged. RMSprop emits
    version 1 with a version-8 checkpoint; Adamax emits version 2 with a
    version-4 checkpoint; both close with a single brace. Adam emits
    version 3 with a version-2 checkpoint and AdamW emits version 4 with
    a version-3 checkpoint; each appends a fifth key digest: the
    lowercase 64-character hexadecimal SHA-256 of the UTF-8 bytes of the
    otherwise identical text containing only the first four keys.
    """
    if isinstance(optimizer, RMSprop):
        version = 1
        checkpoint = dump_checkpoint(parameters, optimizer)
    elif isinstance(optimizer, Adamax):
        version = 2
        checkpoint = dump_checkpoint(parameters, optimizer)
        Adamax._check_hyperparameters(
            optimizer.lr, optimizer.beta1, optimizer.beta2, optimizer.eps
        )
    elif isinstance(optimizer, AdamW):
        version = 4
        checkpoint = dump_checkpoint(parameters, optimizer)
        AdamW._check_hyperparameters(
            optimizer.lr,
            optimizer.beta1,
            optimizer.beta2,
            optimizer.eps,
            optimizer.weight_decay,
        )
    elif isinstance(optimizer, Adam):
        version = 3
        checkpoint = dump_checkpoint(parameters, optimizer)
        Adam._check_hyperparameters(
            optimizer.lr, optimizer.beta1, optimizer.beta2, optimizer.eps
        )
    else:
        raise TypeError(
            "optimizer must be an RMSprop, Adamax, Adam or AdamW instance"
        )
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("global_step must be a non-bool non-negative int")
    if global_step < 0:
        raise ValueError("global_step must be a non-bool non-negative int")
    if isinstance(rng_state, bool) or not isinstance(rng_state, int):
        raise TypeError("rng_state must be a non-bool int in 0..4294967295")
    if rng_state < 0 or rng_state > 4294967295:
        raise ValueError("rng_state must be a non-bool int in 0..4294967295")
    prefix = (
        '{"version":'
        + str(version)
        + ',"global_step":'
        + str(global_step)
        + ',"rng_state":'
        + str(rng_state)
        + ',"checkpoint":'
        + checkpoint
    )
    if version not in (3, 4):
        return prefix + "}"
    # The digest covers the exact bytes of the four-key document; the
    # returned text replaces its closing brace with the digest member.
    payload = prefix + "}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload[:-1] + ',"digest":"' + digest + '"}'


_TRAINING_STATE_DIGEST = re.compile(r',"digest":"([0-9a-f]{64})"\}\Z')


class _TrainingStateParser:
    """Strict parser for the exact textual form of dump_training_state."""

    def __init__(self, text):
        self._text = text
        self._pos = 0

    def _fail(self):
        raise ValueError("text does not match the dump_training_state format")

    def _expect(self, literal):
        if not self._text.startswith(literal, self._pos):
            self._fail()
        self._pos += len(literal)

    def _parse_integer(self):
        match = _CHECKPOINT_INTEGER.match(self._text, self._pos)
        if match is None:
            self._fail()
        self._pos = match.end()
        return int(match.group(0))

    def parse(self):
        self._expect('{"version":')
        version = self._parse_integer()
        if version not in (1, 2, 3, 4):
            raise ValueError("unsupported training state version")
        self._expect(',"global_step":')
        global_step = self._parse_integer()
        self._expect(',"rng_state":')
        rng_state = self._parse_integer()
        if rng_state > 4294967295:
            raise ValueError("rng_state must be in 0..4294967295")
        self._expect(',"checkpoint":')
        # The checkpoint object is followed either by the training state's
        # own single closing brace (versions 1 and 2) or by the version-3
        # /version-4 digest member ,"digest":"H"}. Its exact format
        # (version 8 for v1, version 4 for v2, version 2 for v3, version 3
        # for v4) is validated by the checkpoint parser before any state is
        # applied.
        if version not in (3, 4):
            if not self._text.endswith("}"):
                self._fail()
            checkpoint_text = self._text[self._pos:-1]
            return version, global_step, rng_state, checkpoint_text
        match = _TRAINING_STATE_DIGEST.search(self._text)
        if match is None:
            self._fail()
        # The digest signs the four-key document: every byte up to the
        # digest member's comma, closed again with a single brace.
        payload = self._text[: match.start()] + "}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if digest != match.group(1):
            self._fail()
        checkpoint_text = self._text[self._pos : match.start()]
        return version, global_step, rng_state, checkpoint_text


def _restore_named_adam_family(
    parameters, optimizer, text, names, check_state, state_parser
):
    """Restore an Adam/AdamW state bound by name through one private path.

    This is the single validate -> rearrange -> commit route shared by the
    version-2 Adam state embedded in a version-3 training state and the
    version-3 AdamW state embedded in a version-4 training state. ``names``
    may be in any order but must name exactly the target parameters
    (duplicates rejected); the data, requires_grad, m and v entries are
    bound by name and rearranged into optimizer.parameters order.

    The live optimizer state is validated with ``check_state`` and the
    embedded text with ``state_parser``; counts, name-set equality and the
    scalar/one-dimensional shape of every data, m and v entry are all
    checked (the strict parser already guarantees finite floats, the
    boolean requires_grad tokens and a non-bool non-negative integer t)
    before anything is touched. The commit then happens exactly once:
    each Tensor's data and requires_grad are replaced in place, every grad
    is cleared, and m, v and t are replaced. Tensor identities, the
    optimizer.parameters list and the hyperparameters are preserved.
    """
    check_state(optimizer)
    state_parameters, moments, velocities, t = state_parser(text).parse()
    count = len(optimizer.parameters)
    if len(state_parameters) != count:
        raise ValueError("state must contain exactly one entry per parameter")
    if len(moments) != count or len(velocities) != count:
        raise ValueError("state m and v must match the parameters in length")
    target_names = list(parameters.keys())
    if set(names) != set(target_names) or len(names) != len(target_names):
        raise ValueError(
            "checkpoint names must name exactly the target parameters"
        )
    by_name = {
        name: (spec, moment, velocity)
        for name, spec, moment, velocity in zip(
            names, state_parameters, moments, velocities
        )
    }
    updates = []
    ordered_moments = []
    ordered_velocities = []
    for name, parameter in zip(parameters.keys(), optimizer.parameters):
        data, requires_grad = by_name[name][0]
        current = parameter.data
        if isinstance(current, list):
            if not isinstance(data, list) or len(data) != len(current):
                raise ValueError(
                    "state data shape must match parameter shape"
                )
        elif isinstance(data, list):
            raise ValueError("state data shape must match parameter shape")
        updates.append((parameter, data, requires_grad))
        ordered_moments.append(by_name[name][1])
        ordered_velocities.append(by_name[name][2])
    for slot_name, buffers in (
        ("m", ordered_moments),
        ("v", ordered_velocities),
    ):
        for buffer, parameter in zip(buffers, optimizer.parameters):
            current = parameter.data
            if isinstance(current, list):
                if not isinstance(buffer, list) or len(buffer) != len(current):
                    raise ValueError(
                        "state " + slot_name
                        + " shape must match parameter shape"
                    )
            elif isinstance(buffer, list):
                raise ValueError(
                    "state " + slot_name
                    + " shape must match parameter shape"
                )
    for parameter, data, requires_grad in updates:
        parameter.data = data
        parameter.requires_grad = requires_grad
        parameter.grad = None
    optimizer.m = ordered_moments
    optimizer.v = ordered_velocities
    optimizer.t = t
    return None


def load_training_state(parameters, optimizer, text):
    """Restore named parameters, a supported optimizer and loop state.

    RMSprop, Adamax, Adam and AdamW are supported; any other optimizer
    raises TypeError, and a non-string text raises TypeError. The
    parameters mapping follows the load_checkpoint contract. Only the
    exact forms produced by dump_training_state are accepted: version 1
    must wrap a version-8 RMSprop checkpoint, version 2 a version-4
    Adamax checkpoint, version 3 a version-2 Adam checkpoint and
    version 4 a version-3 AdamW checkpoint, and the checkpoint type must
    match the target optimizer. Any outer parse, key set/order,
    version/target, integer lexical/range or inner checkpoint error
    raises ValueError.

    Versions 3 and 4 additionally require their digest member to verify
    against the SHA-256 of the four-key document and bind the embedded
    checkpoint by name: its names may be in any order but must name
    exactly the target parameters. Any digest, inner version/type or
    name/shape violation raises ValueError.

    The content is fully validated before anything is committed. On
    success the embedded checkpoint restores the optimizer atomically
    (RMSprop: data, requires_grad, square_avg and hyperparameters;
    Adamax: data, requires_grad, m, u and t; Adam: data, requires_grad,
    m, v and t; AdamW: data, requires_grad, m, v and t, keeping object
    identities and hyperparameters), clears every grad and the tuple
    (global_step, rng_state) of two ints is returned. On any failure the
    parameters, grads, slots and hyperparameters are left unchanged.
    """
    if not isinstance(optimizer, (RMSprop, Adamax, Adam, AdamW)):
        raise TypeError(
            "optimizer must be an RMSprop, Adamax, Adam or AdamW instance"
        )
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    optimizer_parameters = optimizer.parameters
    if not isinstance(optimizer_parameters, list):
        raise TypeError("optimizer parameters must be a list")
    _check_state_parameters(parameters)
    if len(parameters) != len(optimizer_parameters):
        raise ValueError("parameters must match the optimizer parameters")
    for (_, tensor), parameter in zip(
        parameters.items(), optimizer_parameters
    ):
        if tensor is not parameter:
            raise ValueError(
                "parameter values must be the optimizer parameters in order"
            )
    version, global_step, rng_state, checkpoint_text = _TrainingStateParser(
        text
    ).parse()
    kind, names, state_text, _ = _CheckpointParser(checkpoint_text).parse()
    if version == 1:
        if kind != "rmsprop" or not isinstance(optimizer, RMSprop):
            raise ValueError(
                "version 1 training state must wrap an RMSprop checkpoint"
            )
    elif version == 2:
        if kind != "adamax" or not isinstance(optimizer, Adamax):
            raise ValueError(
                "version 2 training state must wrap an Adamax checkpoint"
            )
    elif version == 3:
        # The checkpoint parser reaches kind "adam" only through its
        # version-2 branch, so inner version 2 and type "adam" are
        # already enforced byte for byte.
        if kind != "adam" or not isinstance(optimizer, Adam) or isinstance(
            optimizer, AdamW
        ):
            raise ValueError(
                "version 3 training state must wrap an Adam checkpoint"
            )
        _restore_named_adam_family(
            parameters,
            optimizer,
            state_text,
            names,
            _check_adam_state,
            _AdamParser,
        )
        return (global_step, rng_state)
    else:
        # The checkpoint parser reaches kind "adamw" only through its
        # version-3 branch, so inner version 3 and type "adamw" are
        # already enforced byte for byte.
        if kind != "adamw" or not isinstance(optimizer, AdamW):
            raise ValueError(
                "version 4 training state must wrap an AdamW checkpoint"
            )
        _restore_named_adam_family(
            parameters,
            optimizer,
            state_text,
            names,
            _check_adamw_state,
            _AdamWParser,
        )
        return (global_step, rng_state)
    load_checkpoint(parameters, optimizer, checkpoint_text)
    return (global_step, rng_state)


def _format_float(value):
    if not math.isfinite(value):
        raise ValueError("cannot serialize a non-finite float")
    text = f"{value:.6f}"
    if text == "-0.000000":
        text = "0.000000"
    return text


def _json_value(value):
    if isinstance(value, list):
        return "[" + ",".join(_format_float(float(x)) for x in value) + "]"
    return _format_float(float(value))


def _write_stdout_bytes(text):
    """Write text to stdout as UTF-8 bytes, bypassing newline translation."""
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(text.encode("utf-8"))
        buffer.flush()
    else:
        sys.stdout.write(text)


def _cli_tensor():
    line = sys.stdin.readline()
    try:
        payload = json.loads(line)
    except (ValueError, OSError):
        print("invalid JSON input", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("input must be a JSON object", file=sys.stderr)
        return 2
    if set(payload) - {"data", "requires_grad"}:
        print("only data and requires_grad are allowed", file=sys.stderr)
        return 2
    if "data" not in payload:
        print("data is required", file=sys.stderr)
        return 2
    requires_grad = payload.get("requires_grad", False)
    if not isinstance(requires_grad, bool):
        print("requires_grad must be a bool", file=sys.stderr)
        return 2
    try:
        tensor = Tensor(payload["data"], requires_grad)
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(tensor.to_json() + "\n")
    return 0


class _ObjectPairs(list):
    """Marker for a JSON object parsed as ordered (key, value) pairs."""


def _parse_json_pairs(text):
    """Parse JSON preserving object member pairs so duplicates are visible."""
    return json.loads(text, object_pairs_hook=_ObjectPairs)


def _has_duplicate_key(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            return True
        seen.add(key)
    return False


def _read_stdin_line():
    """Read stdin, requiring exactly one line; returns (text, error)."""
    try:
        text = sys.stdin.read()
    except OSError:
        return None, "failed to read stdin"
    if text.endswith("\n"):
        text = text[:-1]
    if "\n" in text:
        return None, "stdin must contain exactly one line of JSON"
    return text, None


def _cli_checkpoint_parameters():
    """Parse the checkpoint stdin payload into an ordered name -> Tensor dict.

    Returns (parameters, error); exactly one of the two is None.
    """
    text, error = _read_stdin_line()
    if error is not None:
        return None, error
    try:
        payload = _parse_json_pairs(text)
    except ValueError:
        return None, "invalid JSON input"
    if not isinstance(payload, _ObjectPairs):
        return None, "input must be an object with only the parameters key"
    if _has_duplicate_key(payload):
        return None, "duplicate key in JSON input"
    if [key for key, _ in payload] != ["parameters"]:
        return None, "input must be an object with only the parameters key"
    members = payload[0][1]
    if not isinstance(members, _ObjectPairs):
        return None, "parameters must be an object"
    if _has_duplicate_key(members):
        return None, "duplicate key in JSON input"
    if len(members) == 0:
        return None, "parameters must be non-empty"
    parameters = {}
    for name, spec in members:
        if not isinstance(spec, _ObjectPairs):
            return (
                None,
                "each parameter must contain only data and requires_grad"
                " in that order",
            )
        if _has_duplicate_key(spec):
            return None, "duplicate key in JSON input"
        if [key for key, _ in spec] != ["data", "requires_grad"]:
            return (
                None,
                "each parameter must contain only data and requires_grad"
                " in that order",
            )
        try:
            parameters[name] = Tensor(spec[0][1], spec[1][1])
        except (TypeError, ValueError) as exc:
            return None, str(exc)
    return parameters, None


def _cli_checkpoint_save(path):
    parameters, error = _cli_checkpoint_parameters()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    try:
        content = dump_state(parameters)
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    error = _atomic_write_checkpoint(path, content)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    return 0


def _cli_checkpoint_load(path):
    try:
        with open(path, "rb") as state_file:
            raw = state_file.read()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parameters, error = _cli_checkpoint_parameters()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    try:
        load_state(parameters, text)
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(dump_state(parameters) + "\n")
    return 0


def _cli_evaluate_samples():
    """Parse the evaluate stdin payload into a list of (x, y) float pairs.

    Returns (samples, error); exactly one of the two is None.
    """
    text, error = _read_stdin_line()
    if error is not None:
        return None, error
    try:
        payload = _parse_json_pairs(text)
    except ValueError:
        return None, "invalid JSON input"
    if not isinstance(payload, _ObjectPairs):
        return None, "input must be an object with only the samples key"
    if _has_duplicate_key(payload):
        return None, "duplicate key in JSON input"
    if [key for key, _ in payload] != ["samples"]:
        return None, "input must be an object with only the samples key"
    members = payload[0][1]
    if not isinstance(members, list) or isinstance(members, _ObjectPairs):
        return None, "samples must be a non-empty array of [x, y] pairs"
    if len(members) == 0:
        return None, "samples must be a non-empty array of [x, y] pairs"
    samples = []
    for item in members:
        if (
            not isinstance(item, list)
            or isinstance(item, _ObjectPairs)
            or len(item) != 2
        ):
            return None, "each sample must be a [x, y] pair"
        x, y = item
        for value in (x, y):
            if isinstance(value, bool) or not isinstance(value, float):
                return None, "sample values must be finite floats"
            if not math.isfinite(value):
                return None, "sample values must be finite floats"
        samples.append((x, y))
    return samples, None


def _cli_evaluate(path):
    try:
        with open(path, "rb") as state_file:
            raw = state_file.read()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    samples, error = _cli_evaluate_samples()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    try:
        entries = _StateParser(text).parse()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if len(entries) == 2:
        if entries[0][0] != "w" or entries[1][0] != "b":
            print("state must contain exactly the parameters w and b in order",
                  file=sys.stderr)
            return 2
        if isinstance(entries[0][1], list) or isinstance(entries[1][1], list):
            print("parameters w and b must be scalar Tensors", file=sys.stderr)
            return 2
    elif len(entries) == 7:
        names = ["w", "b", "mw", "mb", "vw", "vb", "t"]
        requires_flags = (True, True, False, False, False, False, False)
        if (
            [entry[0] for entry in entries] != names
            or any(isinstance(entry[1], list) for entry in entries)
            or any(
                entry[2] is not flag
                for entry, flag in zip(entries, requires_flags)
            )
        ):
            print(
                "state must contain exactly the scalar parameters w, b, mw,"
                " mb, vw, vb and t in order with requires_grad true, true,"
                " false, false, false, false, false",
                file=sys.stderr,
            )
            return 2
        t_value = entries[6][1]
        if (
            not math.isfinite(t_value)
            or t_value < 0.0
            or not t_value.is_integer()
        ):
            print(
                "t must be a non-negative integer-valued float",
                file=sys.stderr,
            )
            return 2
    else:
        print(
            "state must contain exactly the parameters w and b or the"
            " parameters w, b, mw, mb, vw, vb and t",
            file=sys.stderr,
        )
        return 2
    try:
        w_tensor = Tensor(entries[0][1], entries[0][2])
        b_tensor = Tensor(entries[1][1], entries[1][2])
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    w = w_tensor.data
    b = b_tensor.data
    total = 0.0
    for x, y in samples:
        prediction = w * x
        if not math.isfinite(prediction):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
        prediction = prediction + b
        if not math.isfinite(prediction):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
        residual = prediction - y
        if not math.isfinite(residual):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
        squared = residual * residual
        if not math.isfinite(squared):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
        total = total + squared
        if not math.isfinite(total):
            print("evaluation produced a non-finite value", file=sys.stderr)
            return 2
    loss = total / len(samples)
    if not math.isfinite(loss):
        print("evaluation produced a non-finite value", file=sys.stderr)
        return 2
    # Write raw UTF-8 bytes so Windows text-mode newline translation cannot
    # turn the single trailing LF into CRLF.
    _write_stdout_bytes('{"loss":' + _format_float(loss) + "}\n")
    return 0


def _atomic_write_checkpoint(path, content):
    """Atomically replace path with UTF-8 content; None on success, error text."""
    directory = os.path.dirname(os.path.abspath(path))
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
        with os.fdopen(fd, "wb") as tmp_file:
            fd = None  # the file object owns the descriptor now
            tmp_file.write(content.encode("utf-8"))
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        return str(exc)
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return None


def _train_forward(w, b, samples):
    """Mean-squared-error scalar Tensor: sum((w*x+b-y)^2) / n, sample order."""
    total = None
    for x, y in samples:
        residual = w.mul(x).add(b).add(-y)
        squared = residual.mul(residual)
        total = squared if total is None else total.add(squared)
    return total.mul(1.0 / len(samples))


def _cli_train_config(include_momentum=False):
    """Parse the train/momentum stdin payload.

    Returns (config, error); exactly one is None. config is
    (samples, steps, lr, seed) for train, or
    (samples, steps, lr, momentum, seed) when include_momentum is True.
    """
    text, error = _read_stdin_line()
    if error is not None:
        return None, error
    try:
        payload = _parse_json_pairs(text)
    except ValueError:
        return None, "invalid JSON input"
    if not isinstance(payload, _ObjectPairs):
        return None, "input must be an object"
    if _has_duplicate_key(payload):
        return None, "duplicate key in JSON input"
    expected_keys = ["samples", "steps", "lr"]
    if include_momentum:
        expected_keys.append("momentum")
    expected_keys.append("seed")
    if [key for key, _ in payload] != expected_keys:
        if include_momentum:
            return (
                None,
                "input must contain only samples, steps, lr, momentum"
                " and seed in that order",
            )
        return (
            None,
            "input must contain only samples, steps, lr and seed in that order",
        )
    samples_raw = payload[0][1]
    steps = payload[1][1]
    lr = payload[2][1]
    if include_momentum:
        momentum_value = payload[3][1]
        seed = payload[4][1]
    else:
        seed = payload[3][1]
    if not isinstance(samples_raw, list) or isinstance(
        samples_raw, _ObjectPairs
    ):
        return None, "samples must be a non-empty array of [x, y] pairs"
    if len(samples_raw) == 0:
        return None, "samples must be a non-empty array of [x, y] pairs"
    samples = []
    for item in samples_raw:
        if (
            not isinstance(item, list)
            or isinstance(item, _ObjectPairs)
            or len(item) != 2
        ):
            return None, "each sample must be a [x, y] pair"
        x, y = item
        for value in (x, y):
            if isinstance(value, bool) or not isinstance(value, float):
                return None, "sample values must be finite floats"
            if not math.isfinite(value):
                return None, "sample values must be finite floats"
        samples.append((x, y))
    if isinstance(steps, bool) or not isinstance(steps, int):
        return None, "steps must be a positive integer"
    if steps <= 0:
        return None, "steps must be a positive integer"
    if isinstance(lr, bool) or not isinstance(lr, float):
        return None, "lr must be a positive finite float"
    if not math.isfinite(lr) or lr <= 0.0:
        return None, "lr must be a positive finite float"
    if include_momentum:
        momentum_error = (
            "momentum must be a finite float between 0.0 (inclusive)"
            " and 1.0 (exclusive)"
        )
        if isinstance(momentum_value, bool) or not isinstance(
            momentum_value, float
        ):
            return None, momentum_error
        if (
            not math.isfinite(momentum_value)
            or momentum_value < 0.0
            or momentum_value >= 1.0
        ):
            return None, momentum_error
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None, "seed must be an integer between 0 and 4294967295"
    if seed < 0 or seed > 4294967295:
        return None, "seed must be an integer between 0 and 4294967295"
    if include_momentum:
        return (samples, steps, lr, momentum_value, seed), None
    return (samples, steps, lr, seed), None


def _read_state_file(input_path):
    """Read a checkpoint path as strict UTF-8.

    Returns (text, exit_code, error); on success exit_code is 0 and the
    other two are None-ish. Missing/unreadable files exit 1 like other I/O
    failures; non-UTF-8 state is a usage error and exits 2.
    """
    try:
        with open(input_path, "rb") as state_file:
            raw = state_file.read()
    except OSError as exc:
        return None, 1, str(exc)
    try:
        return raw.decode("utf-8"), 0, None
    except UnicodeDecodeError as exc:
        return None, 2, str(exc)


def _cli_train(output_path, input_path):
    if input_path is not None:
        state_text, code, error = _read_state_file(input_path)
        if error is not None:
            print(error, file=sys.stderr)
            return code
    else:
        state_text = None
    config, error = _cli_train_config()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    samples, steps, lr, seed = config
    if state_text is not None:
        try:
            entries = _StateParser(state_text).parse()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if (
            len(entries) != 2
            or entries[0][0] != "w"
            or entries[1][0] != "b"
            or isinstance(entries[0][1], list)
            or isinstance(entries[1][1], list)
            or entries[0][2] is not True
            or entries[1][2] is not True
        ):
            print(
                "state must contain exactly the scalar parameters w and b"
                " with requires_grad true in order",
                file=sys.stderr,
            )
            return 2
        try:
            w = Tensor(entries[0][1], True)
            b = Tensor(entries[1][1], True)
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        w = Tensor(seed / 4294967296.0, True)
        b = Tensor(0.0, True)
    try:
        optimizer = SGD([w, b], lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = _train_forward(w, b, samples)
            loss.backward()
            optimizer.step()
            # Quantize after every step so a reloaded checkpoint resumes the
            # exact same trajectory as uninterrupted training.
            w.data = float(_format_float(w.data))
            b.data = float(_format_float(b.data))
        final_loss = _train_forward(w, b, samples).data
        content = dump_state({"w": w, "b": b})
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    error = _atomic_write_checkpoint(output_path, content)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(
        '{"loss":'
        + _format_float(final_loss)
        + ',"steps":'
        + str(steps)
        + "}\n"
    )
    return 0


def _cli_momentum(output_path, input_path):
    if input_path is not None:
        state_text, code, error = _read_state_file(input_path)
        if error is not None:
            print(error, file=sys.stderr)
            return code
    else:
        state_text = None
    config, error = _cli_train_config(include_momentum=True)
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    samples, steps, lr, momentum, seed = config
    if state_text is not None:
        try:
            entries = _StateParser(state_text).parse()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if len(entries) != 4 or [entry[0] for entry in entries] != [
            "w",
            "b",
            "vw",
            "vb",
        ] or any(isinstance(entry[1], list) for entry in entries):
            print(
                "state must contain exactly the scalar parameters w, b,"
                " vw and vb in order",
                file=sys.stderr,
            )
            return 2
        if (
            entries[0][2] is not True
            or entries[1][2] is not True
            or entries[2][2] is not False
            or entries[3][2] is not False
        ):
            print(
                "w and b must have requires_grad true; vw and vb must have"
                " requires_grad false",
                file=sys.stderr,
            )
            return 2
        try:
            w = Tensor(entries[0][1], True)
            b = Tensor(entries[1][1], True)
            vw = entries[2][1]
            vb = entries[3][1]
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        w = Tensor(seed / 4294967296.0, True)
        b = Tensor(0.0, True)
        vw = 0.0
        vb = 0.0
    try:
        for _ in range(steps):
            w.zero_grad()
            b.zero_grad()
            loss = _train_forward(w, b, samples)
            loss.backward()
            # Compute the velocity buffers and new parameters before
            # committing anything, so a non-finite result aborts the step
            # with all four values untouched.
            new_vw = momentum * vw + w.grad
            new_vb = momentum * vb + b.grad
            if not math.isfinite(new_vw) or not math.isfinite(new_vb):
                raise ValueError("momentum buffer must be finite")
            new_w = w.data - lr * new_vw
            new_b = b.data - lr * new_vb
            if not math.isfinite(new_w) or not math.isfinite(new_b):
                raise ValueError("updated data must be finite")
            # Quantize after every step so a reloaded checkpoint resumes the
            # exact same trajectory as uninterrupted training.
            vw = float(_format_float(new_vw))
            vb = float(_format_float(new_vb))
            w.data = float(_format_float(new_w))
            b.data = float(_format_float(new_b))
        final_loss = _train_forward(w, b, samples).data
        vw_tensor = Tensor(vw, False)
        vb_tensor = Tensor(vb, False)
        content = dump_state(
            {"w": w, "b": b, "vw": vw_tensor, "vb": vb_tensor}
        )
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    error = _atomic_write_checkpoint(output_path, content)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(
        '{"loss":'
        + _format_float(final_loss)
        + ',"steps":'
        + str(steps)
        + "}\n"
    )
    return 0


def _cli_adam_config():
    """Parse the adam stdin payload.

    Returns (config, error); exactly one is None. config is
    (samples, steps, lr, beta1, beta2, epsilon, seed) with samples a list
    of (x, y) float pairs.
    """
    text, error = _read_stdin_line()
    if error is not None:
        return None, error
    try:
        payload = _parse_json_pairs(text)
    except ValueError:
        return None, "invalid JSON input"
    if not isinstance(payload, _ObjectPairs):
        return None, "input must be an object"
    if _has_duplicate_key(payload):
        return None, "duplicate key in JSON input"
    expected_keys = [
        "samples",
        "steps",
        "lr",
        "beta1",
        "beta2",
        "epsilon",
        "seed",
    ]
    if [key for key, _ in payload] != expected_keys:
        return (
            None,
            "input must contain only samples, steps, lr, beta1, beta2,"
            " epsilon and seed in that order",
        )
    samples_raw = payload[0][1]
    steps = payload[1][1]
    lr = payload[2][1]
    beta1 = payload[3][1]
    beta2 = payload[4][1]
    epsilon_value = payload[5][1]
    seed = payload[6][1]
    if not isinstance(samples_raw, list) or isinstance(
        samples_raw, _ObjectPairs
    ):
        return None, "samples must be a non-empty array of [x, y] pairs"
    if len(samples_raw) == 0:
        return None, "samples must be a non-empty array of [x, y] pairs"
    samples = []
    for item in samples_raw:
        if (
            not isinstance(item, list)
            or isinstance(item, _ObjectPairs)
            or len(item) != 2
        ):
            return None, "each sample must be a [x, y] pair"
        x, y = item
        for value in (x, y):
            if isinstance(value, bool) or not isinstance(value, float):
                return None, "sample values must be finite floats"
            if not math.isfinite(value):
                return None, "sample values must be finite floats"
        samples.append((x, y))
    if isinstance(steps, bool) or not isinstance(steps, int):
        return None, "steps must be a positive integer"
    if steps <= 0:
        return None, "steps must be a positive integer"
    if isinstance(lr, bool) or not isinstance(lr, float):
        return None, "lr must be a positive finite float"
    if not math.isfinite(lr) or lr <= 0.0:
        return None, "lr must be a positive finite float"
    for name, value in (("beta1", beta1), ("beta2", beta2)):
        beta_error = (
            name
            + " must be a finite float between 0.0 (inclusive) and 1.0"
            " (exclusive)"
        )
        if isinstance(value, bool) or not isinstance(value, float):
            return None, beta_error
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            return None, beta_error
    if isinstance(epsilon_value, bool) or not isinstance(epsilon_value, float):
        return None, "epsilon must be a positive finite float"
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
        return None, "epsilon must be a positive finite float"
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None, "seed must be an integer between 0 and 4294967295"
    if seed < 0 or seed > 4294967295:
        return None, "seed must be an integer between 0 and 4294967295"
    return (samples, steps, lr, beta1, beta2, epsilon_value, seed), None


def _cli_adam(output_path, input_path):
    if input_path is not None:
        state_text, code, error = _read_state_file(input_path)
        if error is not None:
            print(error, file=sys.stderr)
            return code
    else:
        state_text = None
    config, error = _cli_adam_config()
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    samples, steps, lr, beta1, beta2, epsilon, seed = config
    if state_text is not None:
        try:
            entries = _StateParser(state_text).parse()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        names = ["w", "b", "mw", "mb", "vw", "vb", "t"]
        requires_flags = (True, True, False, False, False, False, False)
        if (
            len(entries) != 7
            or [entry[0] for entry in entries] != names
            or any(isinstance(entry[1], list) for entry in entries)
            or any(
                entry[2] is not flag
                for entry, flag in zip(entries, requires_flags)
            )
        ):
            print(
                "state must contain exactly the scalar parameters w, b, mw,"
                " mb, vw, vb and t in order with requires_grad true, true,"
                " false, false, false, false, false",
                file=sys.stderr,
            )
            return 2
        t_value = entries[6][1]
        if (
            not isinstance(t_value, float)
            or not math.isfinite(t_value)
            or t_value < 0.0
            or not t_value.is_integer()
        ):
            print(
                "t must be a non-negative integer-valued float",
                file=sys.stderr,
            )
            return 2
        try:
            w = Tensor(entries[0][1], True)
            b = Tensor(entries[1][1], True)
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        mw = entries[2][1]
        mb = entries[3][1]
        vw = entries[4][1]
        vb = entries[5][1]
        t = t_value
    else:
        w = Tensor(seed / 4294967296.0, True)
        b = Tensor(0.0, True)
        mw = 0.0
        mb = 0.0
        vw = 0.0
        vb = 0.0
        t = 0.0
    try:
        one_minus_b1 = 1.0 - beta1
        one_minus_b2 = 1.0 - beta2
        for _ in range(steps):
            w.zero_grad()
            b.zero_grad()
            loss = _train_forward(w, b, samples)
            loss.backward()
            # Advance t and derive the bias-correction denominators before
            # touching any parameter or buffer.
            new_t = t + 1.0
            if not math.isfinite(new_t):
                raise ValueError("t must remain finite")
            b1_power = beta1**new_t
            b2_power = beta2**new_t
            if not math.isfinite(b1_power) or not math.isfinite(b2_power):
                raise ValueError("beta power must be finite")
            bias1 = 1.0 - b1_power
            bias2 = 1.0 - b2_power
            if not math.isfinite(bias1) or not math.isfinite(bias2):
                raise ValueError("bias correction must be finite")
            # Compute every new moment and parameter up front, so a
            # non-finite result aborts the step with all seven values
            # untouched.
            updates = []
            for data, grad, m, v in (
                (w.data, w.grad, mw, vw),
                (b.data, b.grad, mb, vb),
            ):
                new_m = beta1 * m + one_minus_b1 * grad
                new_v = beta2 * v + one_minus_b2 * (grad * grad)
                if not math.isfinite(new_m) or not math.isfinite(new_v):
                    raise ValueError("adam buffers must be finite")
                m_hat = new_m / bias1
                v_hat = new_v / bias2
                if not math.isfinite(m_hat) or not math.isfinite(v_hat):
                    raise ValueError(
                        "bias-corrected moments must be finite"
                    )
                denom = math.sqrt(v_hat) + epsilon
                if not math.isfinite(denom):
                    raise ValueError("update denominator must be finite")
                new_data = data - lr * m_hat / denom
                if not math.isfinite(new_data):
                    raise ValueError("updated data must be finite")
                updates.append((new_m, new_v, new_data))
            # Quantize all seven values, then commit, so a reloaded
            # checkpoint resumes the exact same trajectory as uninterrupted
            # training.
            mw = float(_format_float(updates[0][0]))
            vw = float(_format_float(updates[0][1]))
            w_new = float(_format_float(updates[0][2]))
            mb = float(_format_float(updates[1][0]))
            vb = float(_format_float(updates[1][1]))
            b_new = float(_format_float(updates[1][2]))
            t = float(_format_float(new_t))
            w.data = w_new
            b.data = b_new
        final_loss = _train_forward(w, b, samples).data
        content = dump_state(
            {
                "w": w,
                "b": b,
                "mw": Tensor(mw, False),
                "mb": Tensor(mb, False),
                "vw": Tensor(vw, False),
                "vb": Tensor(vb, False),
                "t": Tensor(t, False),
            }
        )
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    error = _atomic_write_checkpoint(output_path, content)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(
        '{"loss":'
        + _format_float(final_loss)
        + ',"steps":'
        + str(steps)
        + "}\n"
    )
    return 0


def main(argv):
    if len(argv) == 1 and argv[0] == "tensor":
        try:
            return _cli_tensor()
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if (
        len(argv) == 3
        and argv[0] == "checkpoint"
        and argv[1] in ("save", "load")
    ):
        try:
            if argv[1] == "save":
                return _cli_checkpoint_save(argv[2])
            return _cli_checkpoint_load(argv[2])
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) == 2 and argv[0] == "evaluate":
        try:
            return _cli_evaluate(argv[1])
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) in (2, 3) and argv[0] == "train":
        try:
            return _cli_train(argv[1], argv[2] if len(argv) == 3 else None)
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) in (2, 3) and argv[0] == "momentum":
        try:
            return _cli_momentum(
                argv[1], argv[2] if len(argv) == 3 else None
            )
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) in (2, 3) and argv[0] == "adam":
        try:
            return _cli_adam(
                argv[1], argv[2] if len(argv) == 3 else None
            )
        except Exception as exc:  # noqa: BLE001 - unexpected CLI failure
            print(str(exc), file=sys.stderr)
            return 1
    print(
        "usage: python autograd.py tensor | checkpoint {save|load} PATH"
        " | evaluate PATH | train OUTPUT [INPUT]"
        " | momentum OUTPUT [INPUT] | adam OUTPUT [INPUT]",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
