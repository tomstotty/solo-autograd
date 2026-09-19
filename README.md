# solo-autograd

从零实现的自动微分与神经网络框架，仅用 Python 标准库、不联网。

- 入口：`python autograd.py <子命令>`
- 所有随机来源必须由显式随机种子决定；相同种子与输入必须产生逐字节相同的输出。
- 数值结果统一写成 JSON，浮点数按固定小数位格式化。

## 测试

    python -m unittest discover
