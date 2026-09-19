# solo-ledger

复式记账命令行工具，数据保存在工作目录下的本地文件中。

- 仅使用 Python 标准库，不联网。
- 入口：`python ledger.py <子命令>`
- 金额以整数「分」存储，累加不使用浮点数。

## 测试

    python -m unittest discover
