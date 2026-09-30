# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.metric.utils import Metric
from verl.workers.engine_workers import _aggregate_mini_batch_metric_value


def test_scalar_dp_metrics_are_preserved_for_later_reduction():
    assert _aggregate_mini_batch_metric_value([3, 3]) == [3, 3]


def test_nested_micro_batch_metrics_are_flattened():
    assert _aggregate_mini_batch_metric_value([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_dp_metric_objects_keep_metric_aggregation():
    first = Metric("mean", 1.0)
    first.append(3.0)
    second = Metric("mean", 3.0)
    second.append(5.0)
    metrics = [first, second]

    assert _aggregate_mini_batch_metric_value(metrics) == 3.0


def test_empty_metric_list_is_preserved():
    assert _aggregate_mini_batch_metric_value([]) == []
