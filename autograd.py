#!/usr/bin/env python3
"""从零实现的最小自动微分框架。

提供 Tensor（有限 float 标量或非空一维 float 列表）、add/mul 运算、
反向传播、固定小数位 JSON 序列化，以及 ``python autograd.py tensor`` 命令行入口。
仅使用 Python 标准库。
"""

import json
import math
import sys


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _is_number(value):
    """bool 虽是 int 的子类，但这里严格视为非数字。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_finite(value):
    if not math.isfinite(value):
        raise ValueError("数值必须有限（不能为 NaN 或 Infinity）")


def _validate_data(data):
    """校验 data，返回 (存储值, 是否标量)。

    类型错误抛 TypeError，非有限 / 空列表抛 ValueError。
    返回的标量与列表元素统一为 float。
    """
    if _is_number(data):
        value = float(data)
        _check_finite(value)
        return value, True
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("data 列表必须非空")
        values = []
        for item in data:
            if not _is_number(item):
                raise TypeError("data 列表元素必须为有限数字")
            value = float(item)
            _check_finite(value)
            values.append(value)
        return values, False
    raise TypeError("data 必须为有限数字标量或非空一维数字列表")


# ---------------------------------------------------------------------------
# 内部工具：广播运算与梯度归约
# ---------------------------------------------------------------------------

def _broadcast_combine(a, a_scalar, b, b_scalar, combine, op_name):
    """标量/向量逐元素广播；两个向量长度不一致抛 ValueError。"""
    if a_scalar and b_scalar:
        return combine(a, b), True
    if a_scalar:
        return [combine(a, y) for y in b], False
    if b_scalar:
        return [combine(x, b) for x in a], False
    if len(a) != len(b):
        raise ValueError("%s 运算中两个向量长度必须一致" % op_name)
    return [combine(x, y) for x, y in zip(a, b)], False


def _sum_to(grad, target_scalar):
    """把上游梯度按广播规则归约到目标形状。

    目标是标量时对向量求和；目标是向量时梯度本身形状已与输出一致。
    """
    if target_scalar:
        return grad if isinstance(grad, float) else float(sum(grad))
    return grad


def _mul_const(grad, const, const_scalar):
    """输出形状的梯度与某个操作数（标量或同形向量）逐元素相乘。"""
    if isinstance(grad, float):
        return grad * const
    if const_scalar:
        return [g * const for g in grad]
    return [g * c for g, c in zip(grad, const)]


def _add_like(a, b):
    """相同形状梯度的累加（标量或等长向量）。"""
    if isinstance(a, float):
        return a + b
    return [x + y for x, y in zip(a, b)]


class _Node:
    """计算图中的一个内部节点：需求梯度的父节点 + 局部梯度函数。"""

    __slots__ = ("parents", "local")

    def __init__(self, parents, local):
        self.parents = parents
        self.local = local


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad 必须为 bool")
        value, is_scalar = _validate_data(data)
        self.data = value
        self.grad = None
        self.requires_grad = requires_grad
        self._is_scalar = is_scalar
        self._node = None  # 叶子节点为 None；运算结果挂 _Node

    # -- 基本属性 --------------------------------------------------------

    @property
    def shape(self):
        return () if self._is_scalar else (len(self.data),)

    def zero_grad(self):
        """清空梯度，返回 None。"""
        self.grad = None
        return None

    # -- 运算 ------------------------------------------------------------

    def _coerce_other(self, other):
        """返回 (other_data, other_scalar, other_tensor_or_None)。"""
        if isinstance(other, Tensor):
            return other.data, other._is_scalar, other
        if not _is_number(other):
            raise TypeError("运算对象必须为 Tensor 或有限数字")
        value = float(other)
        _check_finite(value)
        return value, True, None

    def _binary(self, other, is_mul):
        other_data, other_scalar, other_tensor = self._coerce_other(other)
        combine = (lambda x, y: x * y) if is_mul else (lambda x, y: x + y)
        op_name = "mul" if is_mul else "add"
        out_data, out_scalar = _broadcast_combine(
            self.data, self._is_scalar,
            other_data, other_scalar,
            combine, op_name,
        )

        other_requires = other_tensor is not None and other_tensor.requires_grad
        result = Tensor(out_data, self.requires_grad or other_requires)

        parents = []
        if self.requires_grad:
            parents.append(self)
        if other_requires:
            parents.append(other_tensor)

        def local(grad):
            """给定输出梯度，按 parents 顺序返回各自的局部梯度。"""
            grads = []
            if self.requires_grad:
                if is_mul:
                    g = _mul_const(grad, other_data, other_scalar)
                    grads.append(_sum_to(g, self._is_scalar))
                else:
                    grads.append(_sum_to(grad, self._is_scalar))
            if other_requires:
                if is_mul:
                    g = _mul_const(grad, self.data, self._is_scalar)
                    grads.append(_sum_to(g, other_scalar))
                else:
                    grads.append(_sum_to(grad, other_scalar))
            return grads

        if result.requires_grad:
            result._node = _Node(parents, local)
        return result

    def add(self, other):
        return self._binary(other, is_mul=False)

    def mul(self, other):
        return self._binary(other, is_mul=True)

    # -- 反向传播 --------------------------------------------------------

    def _validate_grad(self, grad):
        """归一化 backward 的初始梯度，并做类型 / 有限性 / 形状校验。"""
        if grad is None:
            if self._is_scalar:
                return 1.0
            raise ValueError("非标量张量 backward 必须传入同形梯度")
        if isinstance(grad, Tensor):
            value, is_scalar = grad.data, grad._is_scalar
        else:
            value, is_scalar = _validate_data(grad)  # TypeError / ValueError
        if is_scalar != self._is_scalar:
            raise ValueError("梯度形状必须与张量形状一致")
        if not is_scalar and len(value) != len(self.data):
            raise ValueError("梯度形状必须与张量形状一致")
        return value

    def backward(self, grad=None):
        """拓扑反向传播并累加叶子梯度；无图（不需求梯度）抛 ValueError。"""
        if not self.requires_grad:
            raise ValueError("无法对不需求梯度的张量执行 backward")
        root_grad = self._validate_grad(grad)

        order = []
        visited = set()

        def visit(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            if tensor._node is not None:
                for parent in tensor._node.parents:
                    visit(parent)
            order.append(tensor)

        visit(self)

        accumulated = {id(self): root_grad}
        for tensor in reversed(order):
            current = accumulated.get(id(tensor))
            node = tensor._node
            if node is None:
                # 叶子：累加进 .grad；从未连入图的叶子保持 None
                if current is not None:
                    if tensor.grad is None:
                        tensor.grad = current
                    else:
                        tensor.grad = _add_like(tensor.grad, current)
                continue
            if current is None:
                continue
            for parent, parent_grad in zip(node.parents, node.local(current)):
                old = accumulated.get(id(parent))
                if old is None:
                    accumulated[id(parent)] = parent_grad
                else:
                    accumulated[id(parent)] = _add_like(old, parent_grad)
        return None

    # -- 序列化 ----------------------------------------------------------

    def to_json(self):
        """紧凑单行 JSON，键序固定为 data, grad, requires_grad。"""
        grad_json = "null" if self.grad is None else _json_number(self.grad)
        return '{"data":%s,"grad":%s,"requires_grad":%s}' % (
            _json_number(self.data),
            grad_json,
            "true" if self.requires_grad else "false",
        )


def _format_fixed(value):
    """固定 6 位小数；负零归一化为 0.000000；禁止 NaN/Infinity。"""
    if not math.isfinite(value):
        raise ValueError("无法将非有限值序列化为 JSON")
    if value == 0.0:
        value = 0.0  # -0.0 -> 0.0
    return format(value, ".6f")


def _json_number(value):
    if isinstance(value, list):
        return "[" + ",".join(_format_fixed(x) for x in value) + "]"
    return _format_fixed(value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_tensor():
    """从 stdin 读一行 JSON（仅含 data / requires_grad），输出 Tensor JSON。"""
    line = sys.stdin.readline()
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("输入必须为 JSON 对象")
        extra = set(obj.keys()) - {"data", "requires_grad"}
        if extra:
            raise ValueError("存在未知字段: %s" % ", ".join(sorted(extra)))
        if "data" not in obj:
            raise ValueError("缺少字段: data")
        requires_grad = obj.get("requires_grad", False)
        tensor = Tensor(obj["data"], requires_grad)
    except (TypeError, ValueError) as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - 其他意外错误归为退出码 1
        print("错误: %s" % exc, file=sys.stderr)
        return 1

    print(tensor.to_json())
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "tensor":
        return _cli_tensor()
    print("用法: python autograd.py tensor", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
