"""滑动窗口指标统计内核（纯标准库，行为完全确定）。

样本的时间戳由调用方注入；内核不使用真实时钟、线程、网络或随机数，
同一输入序列永远得到相同的统计结果。

对外接口：

* Sample        —— 一条样本：时间戳、数值、可选去重键与到达序号；
* MetricWindow  —— 定长滑动窗口：样本始终按时间戳有序保存，过期与超容量
                   的样本都会被丢弃，对外提供计数、求和、均值、增量、速率、
                   分位数与去重计数；
* quantile_of() —— 数值序列的分位数估计（线性插值）。
"""

from bisect import bisect_right

__all__ = ["MetricWindow", "Sample", "quantile_of"]


def _require_int(value, label):
    """确认参数是非布尔的整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s 必须是整数: %r" % (label, value))
    return value


def _require_number(value, label):
    """确认参数是非布尔的数值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s 必须是数值: %r" % (label, value))
    return value


def quantile_of(values, q):
    """对数值序列做分位数估计；q 必须落在 0..1 之间。

    采用线性插值：位置为 q*(n-1)，q=0 取最小值、q=1 取最大值，
    结果随 q 单调不减；空序列返回 None。
    """
    _require_number(q, "分位数")
    if q < 0 or q > 1:
        raise ValueError("分位数必须落在 0..1: %r" % (q,))
    ordered = sorted(values)
    if not ordered:
        return None
    position = q * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    if fraction == 0:
        return ordered[lower_index]
    return (ordered[lower_index]
            + (ordered[upper_index] - ordered[lower_index]) * fraction)


class Sample:
    """窗口里的一条样本。"""

    __slots__ = ("timestamp", "value", "key", "sequence")

    def __init__(self, timestamp, value, key, sequence):
        self.timestamp = timestamp
        self.value = value
        self.key = key
        self.sequence = sequence

    def __repr__(self):
        return "Sample(ts=%r, value=%r, key=%r)" % (
            self.timestamp, self.value, self.key)


class MetricWindow:
    """定长滑动窗口。

    不变量：样本按时间戳非递减有序，且时间戳都落在 (now-span, now] 内；
    样本数不超过 capacity，超出时丢弃时间戳最旧的样本；now 只增不减。
    """

    __slots__ = ("_span", "_capacity", "_items", "_sequence", "_now")

    def __init__(self, span, capacity):
        _require_int(span, "窗口长度")
        _require_int(capacity, "窗口容量")
        if span < 1:
            raise ValueError("窗口长度必须为正: %r" % (span,))
        if capacity < 1:
            raise ValueError("窗口容量必须为正: %r" % (capacity,))
        self._span = span
        self._capacity = capacity
        self._items = []
        self._sequence = 0
        self._now = None

    # ---- 只读视图 -------------------------------------------------

    @property
    def span(self):
        return self._span

    @property
    def capacity(self):
        return self._capacity

    @property
    def now(self):
        return self._now

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return "MetricWindow(span=%d, capacity=%d, size=%d, now=%r)" % (
            self._span, self._capacity, len(self._items), self._now)

    def is_empty(self):
        """窗口里是否已经没有样本。"""
        return not self._items

    def samples(self):
        """按时间戳顺序返回当前保留的样本。"""
        return list(self._items)

    def timestamps(self):
        """当前保留样本的时间戳。"""
        return [sample.timestamp for sample in self._items]

    def values(self):
        """当前保留样本的数值。"""
        return [sample.value for sample in self._items]

    # ---- 写入 -----------------------------------------------------

    def add(self, timestamp, value, key=None):
        """写入一条样本，返回是否被接受。

        时间戳落在窗口左边界 (now-span] 之外（即 timestamp <= now-span）
        的样本会被直接拒绝；迟到但仍在窗口内的样本会按时间戳插入到
        正确位置，窗口始终保持有序。
        """
        _require_int(timestamp, "时间戳")
        _require_number(value, "样本值")
        if self._now is not None and timestamp <= self._now - self._span:
            return False
        if self._now is None or timestamp > self._now:
            self._now = timestamp
            self._expire()
        sample = Sample(timestamp, value, key, self._sequence)
        index = bisect_right([item.timestamp for item in self._items],
                             timestamp)
        self._items.insert(index, sample)
        if len(self._items) > self._capacity:
            del self._items[0]
        self._sequence += 1
        return True

    def advance(self, now):
        """把窗口推进到 now，返回因此被丢弃的样本数。

        时间只能向前推进；回退（now 小于当前时间）抛出 ValueError。
        """
        _require_int(now, "时间")
        if self._now is not None and now < self._now:
            raise ValueError("时间只能向前推进: %r < %r" % (now, self._now))
        if self._now is None or now > self._now:
            self._now = now
            return self._expire()
        return 0

    # ---- 查询 -----------------------------------------------------

    def count(self):
        """窗口内当前保留的样本数。"""
        return len(self._items)

    def total(self):
        """窗口内样本值之和。"""
        return sum(sample.value for sample in self._items)

    def mean(self):
        """窗口内样本的平均值；空窗口无样本可平均，返回 None。"""
        if not self._items:
            return None
        return self.total() / len(self._items)

    def delta(self):
        """最新样本与最旧样本的数值差。"""
        if len(self._items) < 2:
            return 0
        return self._items[-1].value - self._items[0].value

    def rate(self):
        """窗口内数值的单位时间增量。

        分母取最新与最旧样本自身的时间戳之差，与当前窗口时间无关；
        样本少于两条或首尾时间戳相同时返回 0.0。
        """
        if len(self._items) < 2:
            return 0.0
        oldest = self._items[0]
        newest = self._items[-1]
        elapsed = newest.timestamp - oldest.timestamp
        if elapsed <= 0:
            return 0.0
        return (newest.value - oldest.value) / elapsed

    def quantile(self, q):
        """窗口内数值的分位数。"""
        return quantile_of((sample.value for sample in self._items), q)

    def distinct(self):
        """窗口内样本的去重计数。

        带键样本按键去重；没有去重键的样本各自独立计数，不互相合并。
        """
        seen = set()
        keyless = 0
        for sample in self._items:
            if sample.key is None:
                keyless += 1
            else:
                seen.add(sample.key)
        return len(seen) + keyless

    # ---- 内部结构 -------------------------------------------------

    def _expire(self):
        """丢弃已经落在窗口之外（timestamp <= now-span）的样本。"""
        boundary = self._now - self._span
        keep = 0
        for sample in self._items:
            if sample.timestamp <= boundary:
                keep += 1
            else:
                break
        if keep:
            del self._items[:keep]
        return keep
