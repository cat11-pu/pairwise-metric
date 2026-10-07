"""滑动窗口指标统计内核（纯标准库，行为完全确定）。

样本的时间戳由调用方注入；内核不使用真实时钟、线程、网络或随机数，
同一输入序列永远得到相同的统计结果。

对外接口：

* Sample        —— 一条样本：时间戳、数值、可选去重键与到达序号；
* MetricWindow  —— 定长滑动窗口：环形槽位保存样本，过期与超容量的样本
                   都会被丢弃，对外提供计数、求和、均值、增量、速率、
                   分位数与去重计数；
* quantile_of() —— 数值序列的分位数估计。
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
    """对数值序列做分位数估计；q 必须落在 0..1 之间。"""
    _require_number(q, "分位数")
    if q < 0 or q > 1:
        raise ValueError("分位数必须落在 0..1: %r" % (q,))
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


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
    """定长滑动窗口。"""

    __slots__ = ("_span", "_capacity", "_slots", "_head", "_size",
                 "_sequence", "_now")

    def __init__(self, span, capacity):
        _require_int(span, "窗口长度")
        _require_int(capacity, "窗口容量")
        if span < 1:
            raise ValueError("窗口长度必须为正: %r" % (span,))
        if capacity < 1:
            raise ValueError("窗口容量必须为正: %r" % (capacity,))
        self._span = span
        self._capacity = capacity
        self._slots = [None] * capacity
        self._head = 0
        self._size = 0
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
        return self._size

    def __repr__(self):
        return "MetricWindow(span=%d, capacity=%d, size=%d, now=%r)" % (
            self._span, self._capacity, self._size, self._now)

    def is_empty(self):
        """窗口里是否已经没有样本。"""
        return self._size == 0

    def samples(self):
        """按窗口顺序返回当前保留的样本。"""
        return [self._slots[(self._head + index) % self._capacity]
                for index in range(self._size)]

    def timestamps(self):
        """当前保留样本的时间戳。"""
        return [sample.timestamp for sample in self.samples()]

    def values(self):
        """当前保留样本的数值。"""
        return [sample.value for sample in self.samples()]

    # ---- 写入 -----------------------------------------------------

    def add(self, timestamp, value, key=None):
        """写入一条样本，返回是否被接受。

        时间戳已经落在窗口之外的样本会被直接丢弃。
        """
        _require_int(timestamp, "时间戳")
        _require_number(value, "样本值")
        if self._now is not None and timestamp <= self._now - self._span:
            return False
        if self._now is None or timestamp > self._now:
            self._now = timestamp
            self._expire()
        if self._size >= self._capacity:
            self._drop_oldest()
        self._insert(Sample(timestamp, value, key, self._sequence))
        self._sequence += 1
        return True

    def advance(self, now):
        """把窗口推进到 now，返回因此被丢弃的样本数。"""
        _require_int(now, "时间")
        if self._now is not None and now < self._now:
            raise ValueError("时间不能回退: %r < %r" % (now, self._now))
        self._now = now
        return self._expire()

    # ---- 查询 -----------------------------------------------------

    def count(self):
        """窗口内当前保留的样本数。"""
        return self._size

    def total(self):
        """窗口内样本值之和。"""
        return sum(self.values())

    def mean(self):
        """窗口内样本的平均值。"""
        if self._size == 0:
            return None
        return self.total() / self._size

    def delta(self):
        """最新样本与最旧样本的数值差。"""
        if self._size == 0:
            return 0
        return self._newest().value - self._oldest().value

    def rate(self):
        """窗口内数值的单位时间增量。

        样本少于两条或首尾时间戳相同时返回 0.0。
        """
        if self._size < 2:
            return 0.0
        oldest = self._oldest()
        newest = self._newest()
        elapsed = newest.timestamp - oldest.timestamp
        if elapsed <= 0:
            return 0.0
        return (newest.value - oldest.value) / elapsed

    def quantile(self, q):
        """窗口内数值的分位数。"""
        return quantile_of(self.values(), q)

    def distinct(self):
        """窗口内样本的去重计数。"""
        keyed = {sample.key for sample in self.samples()
                 if sample.key is not None}
        unkeyed = sum(1 for sample in self.samples() if sample.key is None)
        return len(keyed) + unkeyed

    # ---- 内部结构 -------------------------------------------------

    def _oldest(self):
        return self._slots[self._head]

    def _newest(self):
        return self._slots[(self._head + self._size - 1) % self._capacity]

    def _drop_oldest(self):
        if self._size == 0:
            return None
        sample = self._slots[self._head]
        self._slots[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._size -= 1
        return sample

    def _expire(self):
        """丢弃已经落在窗口之外的样本，返回丢弃条数。"""
        if self._now is None:
            return 0
        dropped = 0
        while (self._size > 0
               and self._oldest().timestamp <= self._now - self._span):
            self._drop_oldest()
            dropped += 1
        return dropped

    def _insert(self, sample):
        """把样本放进环形槽位，保持窗口按时间戳有序。"""
        if self._size == 0 or sample.timestamp >= self._newest().timestamp:
            slot = (self._head + self._size) % self._capacity
            self._slots[slot] = sample
            self._size += 1
            return
        ordered = self.samples()
        index = bisect_right([item.timestamp for item in ordered],
                             sample.timestamp)
        ordered.insert(index, sample)
        self._rebuild(ordered)

    def _rebuild(self, ordered):
        """按给定顺序重建环形槽位。"""
        self._slots = [None] * self._capacity
        for index, sample in enumerate(ordered):
            self._slots[index] = sample
        self._head = 0
        self._size = len(ordered)
