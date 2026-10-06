"""metric.core 的验收测试。

只断言期望的统计结果与不变量，覆盖正常窗口、过期与容量、乱序时间戳、
分位数、去重计数、速率、空窗口语义与异常输入。
"""

import unittest

from metric import MetricWindow, quantile_of


def feed(window, samples):
    """按顺序向窗口写入 (时间戳, 数值) 样本。"""
    for timestamp, value in samples:
        window.add(timestamp, value)


class AggregateTests(unittest.TestCase):

    def test_basic_aggregates_over_a_window(self):
        window = MetricWindow(span=100, capacity=10)
        feed(window, [(10, 100.0), (20, 140.0), (30, 180.0)])
        self.assertEqual(window.count(), 3)
        self.assertEqual(window.timestamps(), [10, 20, 30])
        self.assertEqual(window.values(), [100.0, 140.0, 180.0])
        self.assertEqual(window.total(), 420.0)
        self.assertEqual(window.mean(), 140.0)
        self.assertEqual(window.delta(), 80.0)
        self.assertEqual(window.rate(), 4.0)
        self.assertEqual(window.now, 30)
        self.assertFalse(window.is_empty())

    def test_empty_window_semantics(self):
        window = MetricWindow(span=5, capacity=4)
        self.assertTrue(window.is_empty())
        self.assertEqual(window.count(), 0)
        self.assertEqual(window.samples(), [])
        self.assertEqual(window.timestamps(), [])
        self.assertEqual(window.values(), [])
        self.assertEqual(window.total(), 0)
        self.assertIsNone(window.mean())
        self.assertEqual(window.delta(), 0)
        self.assertEqual(window.rate(), 0.0)
        self.assertIsNone(window.quantile(0.5))
        self.assertEqual(window.distinct(), 0)
        self.assertEqual(window.advance(9), 0)
        self.assertTrue(window.is_empty())


class EvictionTests(unittest.TestCase):

    def test_expiry_boundary_drops_the_sample_and_counts_it(self):
        window = MetricWindow(span=10, capacity=100)
        self.assertTrue(window.add(5, 1.0))
        self.assertTrue(window.add(12, 2.0))
        self.assertFalse(window.add(2, 9.0))
        self.assertEqual(window.count(), 2)
        self.assertEqual(window.advance(15), 1)
        self.assertEqual(window.timestamps(), [12])
        self.assertEqual(window.count(), 1)
        self.assertEqual(window.total(), 2.0)
        self.assertEqual(window.advance(15), 0)

    def test_capacity_keeps_the_newest_samples(self):
        window = MetricWindow(span=100, capacity=3)
        feed(window, [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
        self.assertEqual(window.count(), 3)
        self.assertEqual(window.timestamps(), [3, 4, 5])
        self.assertEqual(window.values(), [30, 40, 50])
        self.assertEqual(window.total(), 120)
        self.assertEqual(window.advance(101), 0)

    def test_window_rolls_forward_without_regrowing(self):
        window = MetricWindow(span=10, capacity=3)
        feed(window, [(5, 50), (9, 90), (12, 120)])
        window.advance(15)
        self.assertEqual(window.count(), 2)
        self.assertEqual(window.timestamps(), [9, 12])
        self.assertEqual(window.total(), 210)
        self.assertEqual(window.mean(), 105.0)
        self.assertEqual(window.delta(), 30)
        self.assertTrue(window.add(20, 200))
        self.assertEqual(window.count(), 2)
        self.assertEqual(window.timestamps(), [12, 20])
        self.assertEqual(window.total(), 320)
        self.assertEqual(window.mean(), 160.0)
        self.assertEqual(window.delta(), 80)
        self.assertEqual(window.rate(), 10.0)

        rolling = MetricWindow(span=10, capacity=4)
        feed(rolling, [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
        previous = rolling.count()
        for now in (8, 12, 20, 40):
            rolling.advance(now)
            self.assertLessEqual(rolling.count(), previous)
            previous = rolling.count()
            for timestamp in rolling.timestamps():
                self.assertGreater(timestamp, rolling.now - rolling.span)
                self.assertLessEqual(timestamp, rolling.now)


class RateTests(unittest.TestCase):

    def test_rate_uses_sample_endpoints(self):
        window = MetricWindow(span=1000, capacity=10)
        feed(window, [(10, 100.0), (20, 140.0), (30, 180.0)])
        self.assertEqual(window.delta(), 80.0)
        self.assertEqual(window.rate(), 4.0)
        window.advance(80)
        self.assertEqual(window.count(), 3)
        self.assertEqual(window.timestamps(), [10, 20, 30])
        self.assertEqual(window.rate(), 4.0)

        flat = MetricWindow(span=100, capacity=10)
        flat.add(1, 5.0)
        self.assertEqual(flat.rate(), 0.0)
        flat.add(1, 9.0)
        self.assertEqual(flat.count(), 2)
        self.assertEqual(flat.rate(), 0.0)

    def test_out_of_order_samples_keep_endpoint_semantics(self):
        window = MetricWindow(span=100, capacity=10)
        window.add(10, 100.0)
        window.add(30, 250.0)
        self.assertTrue(window.add(20, 300.0))
        window.advance(60)
        self.assertEqual(window.count(), 3)
        self.assertEqual(window.timestamps(), [10, 20, 30])
        self.assertEqual(window.values(), [100.0, 300.0, 250.0])
        self.assertEqual(window.total(), 650.0)
        self.assertEqual(window.mean(), 650.0 / 3)
        self.assertEqual(window.delta(), 150.0)
        self.assertEqual(window.rate(), 7.5)
        self.assertTrue(window.add(5, 50.0))
        self.assertEqual(window.timestamps(), [5, 10, 20, 30])


class DistributionTests(unittest.TestCase):

    def test_quantiles_interpolate_linearly(self):
        window = MetricWindow(span=100, capacity=10)
        feed(window, [(1, 10), (2, 20), (3, 30), (4, 40)])
        self.assertEqual(window.quantile(0.0), 10.0)
        self.assertEqual(window.quantile(0.25), 17.5)
        self.assertEqual(window.quantile(0.5), 25.0)
        self.assertEqual(window.quantile(0.75), 32.5)
        self.assertEqual(window.quantile(1.0), 40.0)
        self.assertEqual(quantile_of([10, 20, 30, 40], 0.5), 25.0)
        self.assertEqual(quantile_of([7], 0.9), 7.0)
        self.assertIsNone(quantile_of([], 0.5))

        trend = MetricWindow(span=100, capacity=10)
        feed(trend, [(1, 5), (2, 3), (3, 9), (4, 1), (5, 7)])
        quantiles = [trend.quantile(step / 10.0) for step in range(11)]
        self.assertEqual(quantiles, sorted(quantiles))

    def test_distinct_count_uses_keys(self):
        window = MetricWindow(span=10, capacity=10)
        window.add(1, 1.0, key="a")
        window.add(2, 2.0, key="a")
        window.add(4, 4.0, key="b")
        window.add(5, 5.0)
        window.add(6, 6.0)
        self.assertEqual(window.count(), 5)
        self.assertEqual(window.distinct(), 4)
        self.assertEqual(window.advance(13), 2)
        self.assertEqual(window.count(), 3)
        self.assertEqual(window.timestamps(), [4, 5, 6])
        self.assertEqual(window.distinct(), 3)


class InputValidationTests(unittest.TestCase):

    def test_invalid_inputs_and_backward_time_are_rejected(self):
        self.assertRaises(ValueError, MetricWindow, 0, 4)
        self.assertRaises(ValueError, MetricWindow, -3, 4)
        self.assertRaises(ValueError, MetricWindow, 5, 0)
        self.assertRaises(ValueError, MetricWindow, 5, -1)
        self.assertRaises(TypeError, MetricWindow, 5.5, 4)
        self.assertRaises(TypeError, MetricWindow, 5, True)

        window = MetricWindow(span=5, capacity=4)
        self.assertRaises(TypeError, window.add, 1.5, 2)
        self.assertRaises(TypeError, window.add, True, 2)
        self.assertRaises(TypeError, window.add, 1, True)
        self.assertRaises(TypeError, window.add, 1, "oops")
        self.assertRaises(ValueError, window.quantile, 1.5)
        self.assertRaises(ValueError, window.quantile, -0.1)
        self.assertRaises(TypeError, window.quantile, "half")
        self.assertRaises(TypeError, window.advance, 2.5)

        self.assertTrue(window.add(9, 3.0))
        self.assertEqual(window.now, 9)
        self.assertRaises(ValueError, window.advance, 4)


if __name__ == "__main__":
    unittest.main()
