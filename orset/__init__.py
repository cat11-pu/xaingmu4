"""Add-wins OR-Set（Observed-Remove Set）CRDT。

只依赖标准库，线程安全。

语义：
- 每次 add 生成一个全局唯一 tag；元素在场当且仅当它存在未被删除的 tag。
- remove 只会删除"自己当前见过"的 tag，因此并发的 add 赢：
  没见过的 add tag 不会被顺手删掉。
- merge 是两个副本状态（add tag 集合 / remove tag 集合）的逐项并集，
  因此满足交换律、结合律、幂等律，且不会修改传入的 other 副本。

墓碑回收（GC）：
tag 和墓碑天然只增不减，长时间增删会留下垃圾。本实现做两层回收，
add-wins 语义与 merge 收敛性不变：

1. 一个 tag 一旦进入本副本的 removes，它对本地成员判定永久失效
   （removes 单调增长），立即从 adds 中摘除；merge 收到墓碑时同样
   摘除本地 adds 中对应的死 tag。墓碑保留用于向其他副本传播。
2. 墓碑 t 只有在"进程内不存在任何仍在 adds 中持有 t 的存活副本"时
   才会被彻底清除。tag 全局唯一且只随并集传播：一旦没有任何副本还
   持有 add(t)，t 就不可能被重新引入，墓碑已无对象可中和。
   所有 ORSet 实例登记在模块级 WeakSet 中——已被解释器回收的副本
   不可能再把旧 tag 传播回来，新建副本的初始状态为空也不持有古旧 tag。

跨进程/分布式部署不在此保证内：那边的墓碑回收需要显式的组成员与
ack 协议（CRDT log cleanup 的标准前提）。
"""

from __future__ import annotations

import contextlib
import threading
import uuid
import weakref

__all__ = ["ORSet"]

# 进程内所有存活的 ORSet 副本。用 WeakSet：副本本身被回收后，
# 它不可能再持有任何旧 tag，自动退出 GC 的确认集合。
_LIVE_REPLICAS: "weakref.WeakSet[ORSet]" = weakref.WeakSet()


class ORSet:
    """状态型 add-wins OR-Set。

    元素必须是可哈希的。每个副本构造时传入一个 replica_id（仅用于调试，
    tag 的全局唯一性同时由进程内计数器和 uuid4 保证）。
    """

    def __init__(self, replica_id):
        self._replica_id = replica_id
        # 每个副本内自增的 add 序号
        self._counter = 0
        # x -> 所有"仍可能有效"的 add tag（已被本地墓碑覆盖的 tag 立即摘除）
        self._adds: dict = {}
        # x -> 所有已删除的 add tag（tombstone），等待全副本确认后回收
        self._removes: dict = {}
        self._lock = threading.RLock()
        _LIVE_REPLICAS.add(self)

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
            if not visible:
                return
            removes = self._removes.setdefault(x, set())
            removes.update(visible)
            # 这些 tag 对本地成员判定已永久失效，立即从 adds 摘除。
            self._adds[x].difference_update(visible)
            if not self._adds[x]:
                del self._adds[x]
        # 放锁后再跑 GC：GC 需要按全局顺序获取多个副本的锁，
        # 不能在持有本副本锁时嵌套获取（否则与并发 merge 构成锁序倒置）。
        # 只检查刚埋下的 tag：若全球已经没人持有 add(t) 立即回收。
        self._gc(visible)

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
                # set.update 只读取 other 集合并逐元素拷贝，不共享内部
                # 集合对象；全程只写 self，other 只读。
                for x, tags in other._adds.items():
                    bucket = self._adds.setdefault(x, set())
                    bucket.update(tags)
                    # 自己已有墓碑的 tag 即使随对端 adds 到达也立即抵消，
                    # 否则它会重新进入本地 adds，让该 tag 的墓碑永远无法回收。
                    if x in self._removes:
                        bucket.difference_update(self._removes[x])
                    if not bucket:
                        del self._adds[x]
                for x, tags in other._removes.items():
                    self._removes.setdefault(x, set()).update(tags)
                    # 新到的墓碑立刻中和本地 adds 中的死 tag，
                    # 否则本副本会一直被误判为 add(t) 的持有者。
                    if x in self._adds:
                        self._adds[x].difference_update(tags)
                        if not self._adds[x]:
                            del self._adds[x]
        # other 刚确认过一批墓碑，可能满足回收条件（放锁后再 GC）。
        self._gc()
        return self

    def _gc(self, only_tags=None) -> None:
        """回收已无任何存活副本在 adds 中持有的 tag 及其墓碑。

        判据（调用时不持有任何锁）：tag 全局唯一、只随 adds 的并集传播；
        一旦所有存活副本的 adds 里都没有 t，t 就再也不可能被重新引入，
        墓碑已无对象可中和，可以彻底删除。实例登记在 WeakSet 中，已被
        解释器回收的副本自动退出确认集合。

        only_tags 给定时只检查这批 tag（remove 刚埋下的），避免全表扫描；
        merge 后不传参，做一次全表回收。只清自己，不清别的副本。
        """
        # 先拍一张存活副本的强引用快照（WeakSet 迭代期间可能变动）。
        peers = [r for r in _LIVE_REPLICAS if r is not None]
        locks = sorted((r._lock for r in peers), key=id)
        with contextlib.ExitStack() as stack:
            for lock in locks:
                stack.enter_context(lock)
            # 持全部锁期间没有任何副本能改动状态，判定结果一致。
            # 先算出全球仍被持有的 tag 集合（只关注本副本有墓碑的元素）。
            for x in list(self._removes):
                tombstones = self._removes.get(x)
                if not tombstones:
                    continue
                candidates = tombstones if only_tags is None else tombstones & set(only_tags)
                if not candidates:
                    continue
                held_anywhere = set()
                for r in peers:
                    live = r._adds.get(x)
                    if live:
                        held_anywhere |= live
                dead = candidates - held_anywhere
                if not dead:
                    continue
                tombstones.difference_update(dead)
                if x in self._adds:
                    self._adds[x].difference_update(dead)
                    if not self._adds[x]:
                        del self._adds[x]
                if not tombstones:
                    del self._removes[x]
