"""GC 前后内存对比：旧实现（tag/tombstone 只增不减）vs 新实现（自动回收）。

跑法：python benchmarks/memory_compare.py
"""

import sys
import tracemalloc

sys.path.insert(0, ".")

from orset import ORSet  # noqa: E402

N = 20_000


def state_size(obj, seen=None):
    """递归统计内部 adds/removes 状态占用的字节。"""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += state_size(k, seen) + state_size(v, seen)
    elif isinstance(obj, (set, frozenset, list, tuple)):
        for item in obj:
            total += state_size(item, seen)
    return total


class OldORSet:
    """GC 引入前的实现：adds/removes 只增不减。"""

    def __init__(self, replica_id):
        self._replica_id = replica_id
        self._counter = 0
        self._adds = {}
        self._removes = {}

    def _tag(self):
        import uuid

        self._counter += 1
        return f"{self._replica_id}:{self._counter}:{uuid.uuid4().hex}"

    def add(self, x):
        self._adds.setdefault(x, set()).add(self._tag())

    def remove(self, x):
        visible = self._adds.get(x, set()) - self._removes.get(x, set())
        if visible:
            self._removes.setdefault(x, set()).update(visible)

    def elements(self):
        return {
            x for x, t in self._adds.items() if t - self._removes.get(x, set())
        }


def workload(cls, distinct_keys):
    s = cls("r1")
    for i in range(N):
        x = i if distinct_keys else "hot"
        s.add(x)
        s.remove(x)
    return s


def measure(label, factory):
    tracemalloc.start()
    s = factory()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    deep = state_size(s._adds) + state_size(s._removes)
    print(
        f"{label:<34} 逻辑元素={len(s.elements()):>6}  "
        f"内部状态深度字节={deep:>10,}  "
        f"tracemalloc当前={current:>10,}  峰值={peak:>10,}"
    )
    return deep


def two_replica_gossip():
    """两副本：r1 增删 N 个不同 key，每个操作都双向同步后墓碑应被回收。"""
    a, b = ORSet("A"), ORSet("B")
    old_a, old_b = OldORSet("A"), OldORSet("B")
    for i in range(N):
        a.add(i)
        a.remove(i)
        b.merge(a)
        a.merge(b)
        old_a.add(i)
        old_a.remove(i)
        old_b._adds.setdefault(i, set()).update(old_a._adds[i])
        old_b._removes.setdefault(i, set()).update(old_a._removes[i])
    new_bytes = state_size(a._adds) + state_size(a._removes)
    old_bytes = state_size(old_a._adds) + state_size(old_a._removes)
    print(
        f"{'双副本 gossip 后 r1（新+GC）':<34} 逻辑元素={len(a.elements()):>6}  "
        f"内部状态深度字节={new_bytes:>10,}"
    )
    print(
        f"{'双副本 gossip 后 r1（旧无GC）':<34} 逻辑元素={len(old_a.elements()):>6}  "
        f"内部状态深度字节={old_bytes:>10,}"
    )
    print(f"  r2 同样已回收: {b._adds == {} and b._removes == {}}")


if __name__ == "__main__":
    print(f"每副本 {N:,} 次 add+remove，结束后逻辑集合为空\n")
    old1 = measure("单副本 不同key（旧无GC）", lambda: workload(OldORSet, True))
    new1 = measure("单副本 不同key（新+GC）", lambda: workload(ORSet, True))
    old2 = measure("单副本 同一热点key（旧无GC）", lambda: workload(OldORSet, False))
    new2 = measure("单副本 同一热点key（新+GC）", lambda: workload(ORSet, False))
    print()
    two_replica_gossip()
    print()
    print(f"不同 key 场景缩减：{old1:,} -> {new1:,} 字节（{old1 / max(new1,1):.0f}x）")
    print(f"热点 key 场景缩减：{old2:,} -> {new2:,} 字节（{old2 / max(new2,1):.0f}x）")
