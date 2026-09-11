"""Add-wins OR-Set（Observed-Remove Set）CRDT。

只依赖标准库。

语义：
- 每次 add 生成一个全局唯一 tag；元素在场当且仅当它存在未被删除的 tag。
- remove 只会删除"自己当前见过"的 tag，因此并发的 add 赢：
  没见过的 add tag 不会被顺手删掉。
- merge 是两个副本状态（add tag 集合 / remove tag 集合）的逐项并集，
  因此满足交换律、结合律、幂等律，且不会修改传入的 other 副本。
- 所有公开方法均线程安全。
"""

from __future__ import annotations

import threading
import uuid

__all__ = ["ORSet"]


class ORSet:
    """状态型 add-wins OR-Set。

    元素必须是可哈希的。每个副本构造时传入一个 replica_id（仅用于调试，
    tag 的全局唯一性同时由进程内计数器和 uuid4 保证）。
    """

    def __init__(self, replica_id):
        self._replica_id = replica_id
        # 每个副本内自增的 add 序号
        self._counter = 0
        # x -> 所有见过的 add tag
        self._adds: dict = {}
        # x -> 所有已删除的 add tag（tombstone）
        self._removes: dict = {}
        self._lock = threading.RLock()

    def _new_tag(self) -> str:
        """生成全局唯一 tag（调用方须已持锁）。"""
        self._counter += 1
        return f"{self._replica_id}:{self._counter}:{uuid.uuid4().hex}"

    def add(self, x) -> None:
        """加入元素 x。每次调用都会打一个全新的全局唯一 tag。"""
        with self._lock:
            self._adds.setdefault(x, set()).add(self._new_tag())

    def remove(self, x) -> None:
        """删除元素 x：只删除当前可见（自己见过且未被删除）的 tag。

        若 x 不在集合中（包括从未见过的元素），这是一个 no-op，
        不会记录任何 tombstone，因此不会挡住之后 merge 进来的并发 add。
        """
        with self._lock:
            visible = self._adds.get(x, set()) - self._removes.get(x, set())
            if visible:
                self._removes.setdefault(x, set()).update(visible)

    def lookup(self, x) -> bool:
        """x 是否在集合中。"""
        with self._lock:
            return bool(self._adds.get(x, set()) - self._removes.get(x, set()))

    def elements(self) -> set:
        """返回当前所有元素的快照集合。

        返回的是新建的 set，调用方随意修改，不会影响副本内部状态。
        """
        with self._lock:
            return {
                x
                for x, tags in self._adds.items()
                if tags - self._removes.get(x, set())
            }

    def merge(self, other) -> "ORSet":
        """把 other 的状态并入本副本（in-place），返回 self。

        - 不修改 other 的状态；
        - 双副本按锁对象 id 排序加锁，避免双向并发 merge 时死锁。
        """
        if not isinstance(other, ORSet):
            raise TypeError(f"只能合并 ORSet，收到 {type(other).__name__}")
        if other is self:
            # 幂等：自己 merge 自己什么都不做
            return self

        first, second = sorted((self._lock, other._lock), key=id)
        with first:
            with second:
                # 直接 update：set.update 只读取 other 集合并逐元素拷贝，
                # 不共享内部集合对象；全程只写 self，other 只读。
                for x, tags in other._adds.items():
                    self._adds.setdefault(x, set()).update(tags)
                for x, tags in other._removes.items():
                    self._removes.setdefault(x, set()).update(tags)
        return self
