# pairwise-metric

只依赖 Python 标准库的滑动窗口指标统计内核：样本时间戳由调用方注入，
不使用真实时钟、线程、网络或随机数，同一输入序列永远得到同样的结果。

- `metric/core.py` — 统计内核：定长环形窗口、过期与超容量丢弃、计数/求和/均值/增量/速率/分位数/去重计数。
- `tests/test_core.py` — 验收用例。

## 运行测试

在项目根目录执行：

```
python3 -m unittest discover -s tests -v
```

Windows 上如果没有 `python3`，可用：

```
python -m unittest discover -s tests -v
```
