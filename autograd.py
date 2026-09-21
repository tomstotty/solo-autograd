#!/usr/bin/env python3
"""Minimal autograd framework: finite scalar / 1D float tensors."""

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

    def conv_transpose1d(self, kernel, stride=1, padding=0):
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
        n = len(data)
        k = len(weights)
        out_len = (n - 1) * stride - 2 * padding + k
        if out_len <= 0:
            raise ValueError("conv_transpose1d output length must be positive")
        # Full transposed cross-correlation: each input element scatters a
        # scaled copy of the kernel onto the output, accumulating
        # out[i*stride - padding + r] in ascending i then r order and
        # skipping out-of-range positions; a non-finite product or partial
        # sum aborts before a result tensor exists, so no state can change
        # on failure.
        out_data = [0.0] * out_len
        for i in range(n):
            for r in range(k):
                j = i * stride - padding + r
                if not 0 <= j < out_len:
                    continue
                product = data[i] * weights[r]
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
        # Snapshot both operands so later caller-side mutation of either
        # input list cannot change what a pending backward pass uses.
        snapshot_a = list(data)
        snapshot_b = list(weights)

        def backward_fn(grad):
            dx = [0.0] * n
            dk = [0.0] * k
            for i in range(n):
                for r in range(k):
                    j = i * stride - padding + r
                    if not 0 <= j < out_len:
                        continue
                    g = grad[j]
                    contrib_x = g * snapshot_b[r]
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
                    contrib_k = g * snapshot_a[i]
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

    def max_pool1d(self, kernel_size, stride=None, padding=0):
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
        n = len(data)
        k = kernel_size
        out_len = (n + 2 * padding - k) // stride + 1
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
                j = o * stride + i - padding
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
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [self._zeros_like(parameter.data) for parameter in parameters]
        self.v = [self._zeros_like(parameter.data) for parameter in parameters]
        self.t = 0

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
