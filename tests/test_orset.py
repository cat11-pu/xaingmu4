"""orset 测试：add-wins 语义、merge 三大律、快照隔离、并发安全。"""

import random
import sys
import threading

import pytest

from orset import ORSet


# ---------- 基础操作 ----------

def test_add_and_lookup():
    s = ORSet("r1")
    assert s.lookup("a") is False
    s.add("a")
    assert s.lookup("a") is True
    assert s.elements() == {"a"}


def test_lookup_returns_bool():
    s = ORSet("r1")
    assert isinstance(s.lookup("a"), bool)
    s.add(1)
    assert isinstance(s.lookup(1), bool)


def test_remove_present_element():
    s = ORSet("r1")
    s.add("a")
    s.add("b")
    s.remove("a")
    assert not s.lookup("a")
    assert s.lookup("b")


def test_remove_unknown_element_is_noop():
    s = ORSet("r1")
    # 没见过的元素：不报错、不影响后续 add
    s.remove("ghost")
    assert s.elements() == set()
    s.add("ghost")
    assert s.lookup("ghost")


def test_readd_after_remove():
    s = ORSet("r1")
    s.add("a")
    s.remove("a")
    s.add("a")
    assert s.lookup("a")
    s.remove("a")
    assert not s.lookup("a")


def test_elements_returns_snapshot_not_internal_ref():
    s = ORSet("r1")
    s.add("a")
    snap = s.elements()
    snap.add("evil")
    snap.discard("a")
    # 内部状态不受影响
    assert s.elements() == {"a"}
    assert s.lookup("evil") is False


# ---------- add-wins 核心语义 ----------

def test_concurrent_add_wins_over_remove():
    a = ORSet("A")
    b = ORSet("B")
    # 共同历史：两边都见过 x
    a.add("x")
    b.merge(a)
    a.merge(b)
    # 分叉：A 删 x（只删自己见过的 tag），B 并发重新 add x（全新 tag）
    a.remove("x")
    b.add("x")
    # 双向同步后 x 必须在场
    a.merge(b)
    b.merge(a)
    assert a.lookup("x")
    assert b.lookup("x")
    assert a.elements() == b.elements() == {"x"}


def test_remove_does_not_kill_unobserved_add():
    a = ORSet("A")
    b = ORSet("B")
    b.add("x")          # A 从未见过这个 add
    a.remove("x")       # 只能 no-op
    a.merge(b)
    assert a.lookup("x")  # add 赢


# ---------- merge：交换律 / 结合律 / 幂等 / 不改对方 ----------

def _build_forked_replicas():
    """构造三个各有操作的副本，全量合并的期望结果是 {a,b,c,d,e}。"""
    r1 = ORSet("r1")
    r1.add("a")
    r1.add("b")

    r2 = ORSet("r2")
    r2.add("b")
    r2.remove("b")     # 只埋掉自己那个 b tag
    r2.add("c")

    r3 = ORSet("r3")
    r3.add("a")
    r3.remove("a")     # 只埋掉自己那个 a tag
    r3.add("d")

    r4 = ORSet("r4")
    r4.add("e")
    return r1, r2, r3, r4


def test_empty_merge_empty():
    assert ORSet("a").merge(ORSet("b")).elements() == set()


def test_merge_with_empty_either_direction():
    full = ORSet("full")
    full.add("x")
    empty = ORSet("empty")

    assert full.merge(ORSet("e")).elements() == {"x"}
    assert ORSet("e").merge(full).elements() == {"x"}
    # 空副本本身仍是空的
    assert empty.elements() == set()


def test_merge_commutative():
    r1, r2, r3, r4 = _build_forked_replicas()
    s1, s2, s3, s4 = _build_forked_replicas()

    left = ORSet("L")
    for r in (r1, r2, r3, r4):
        left.merge(r)

    right = ORSet("R")
    for r in (s4, s3, s2, s1):
        right.merge(r)

    assert left.elements() == right.elements() == {"a", "b", "c", "d", "e"}


def test_merge_associative():
    # (r1 ⋈ r2) ⋈ r3  ==  r1 ⋈ (r2 ⋈ r3)
    r1a, r2a, r3a, _ = _build_forked_replicas()
    r1b, r2b, r3b, _ = _build_forked_replicas()

    left = r1a.merge(r2a).merge(r3a)
    inner = r2b.merge(r3b)
    right = r1b.merge(inner)

    assert left.elements() == {"a", "b", "c", "d"}
    assert left.elements() == right.elements()


def test_merge_order_shuffle_is_stable():
    expected = {"a", "b", "c", "d", "e"}
    rng = random.Random(20260911)
    reference = None

    for trial in range(20):
        replicas = list(_build_forked_replicas())
        rng.shuffle(replicas)
        acc = ORSet(f"acc-{trial}")
        for r in replicas:
            acc.merge(r)
        if reference is None:
            reference = acc.elements()
        else:
            assert acc.elements() == reference
    assert reference == expected


def test_merge_self_is_noop():
    s = ORSet("self")
    s.add("a")
    s.add("b")
    before = s.elements()
    result = s.merge(s)
    assert result is s
    assert s.elements() == before


def test_merge_twice_equals_once():
    r1, r2, r3, r4 = _build_forked_replicas()
    once = ORSet("once")
    twice = ORSet("twice")

    for r in (r1, r2, r3, r4):
        once.merge(r)

    for r in (r1, r2, r3, r4):
        twice.merge(r)
    for r in (r1, r2, r3, r4):
        twice.merge(r)  # 重复 merge

    assert twice.elements() == once.elements()
    # 源副本状态不被 merge 改动
    assert r2.lookup("b") is False  # r2 自己仍只看到被删后的状态
    assert r3.lookup("a") is False


def test_merge_does_not_mutate_other():
    a = ORSet("A")
    b = ORSet("B")
    a.add("x")
    b.add("y")

    a.merge(b)
    # b 没有被写入：不含 x
    assert b.elements() == {"y"}
    # b 的内部集合与 a 不共享引用：之后改 b 不会回流到 a
    b.add("z")
    assert a.elements() == {"x", "y"}
    assert "z" not in a.elements()


def test_merge_returns_self():
    a = ORSet("A")
    b = ORSet("B")
    assert a.merge(b) is a


def test_merge_rejects_non_orset():
    a = ORSet("A")
    with pytest.raises(TypeError):
        a.merge({"x"})


# ---------- 多线程并发 ----------

def test_concurrent_adds_single_replica():
    s = ORSet("shared")
    n_threads, per_thread = 8, 1000

    def worker(tid):
        for j in range(per_thread):
            s.add((tid, j))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(s.elements()) == n_threads * per_thread


def test_concurrent_add_remove_gossip_converges():
    """4 个副本并发 add/remove/互相 merge，结束后全量同步必须收敛且不崩。"""
    n_replicas = 4
    per_replica = 50
    removed_per_replica = 5
    replicas = [ORSet(f"r{i}") for i in range(n_replicas)]
    errors = []
    start = threading.Barrier(n_replicas)
    done = threading.Barrier(n_replicas)

    def worker(i):
        try:
            start.wait()
            me = replicas[i]
            # 本地 add 自己的 50 个元素
            for j in range(per_replica):
                me.add((i, j))
            # 埋掉自己前 5 个（只可能看到自己的 tag）
            for j in range(removed_per_replica):
                me.remove((i, j))
            # 并发 gossip：多轮随机挑对面 merge，穿插 remove
            for rnd in range(30):
                peer = replicas[(i + 1 + (rnd % (n_replicas - 1))) % n_replicas]
                me.merge(peer)
                if rnd % 5 == 0:
                    me.remove((i, rnd % per_replica))
            done.wait()
        except Exception as exc:  # noqa: BLE001 - 任何异常都算测试失败
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_replicas)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发过程出现异常: {errors!r}"

    # 全量同步（多轮 all-to-all），之后所有副本视图必须一致
    for _ in range(n_replicas):
        for i, ri in enumerate(replicas):
            for rj in replicas:
                ri.merge(rj)

    views = {tuple(sorted(r.elements())) for r in replicas}
    assert len(views) == 1, f"副本未收敛: {[r.elements() for r in replicas]}"

    # 被确定性删掉的键（每个键只有 owner 自己的一个 tag，删了就是全网删）：
    # gossip 前删 0..4，gossip 中穿插删 rnd%50 = 0,5,10,15,20,25
    removed_keys = set(range(removed_per_replica)) | {0, 5, 10, 15, 20, 25}
    expected_count = n_replicas * (per_replica - len(removed_keys))
    assert len(replicas[0].elements()) == expected_count


def test_concurrent_bidirectional_merge_no_deadlock():
    """两个线程方向相反地反复 merge，必须能在超时时间内全部结束（无死锁）。"""
    a, b = ORSet("A"), ORSet("B")
    a.add("x")
    b.add("y")
    errors = []

    def left():
        try:
            for _ in range(200):
                a.add("a-tag")
                a.merge(b)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def right():
        try:
            for _ in range(200):
                b.add("b-tag")
                b.merge(a)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=left)
    t2 = threading.Thread(target=right)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not t1.is_alive() and not t2.is_alive(), "双向 merge 发生死锁"
    assert not errors
    a.merge(b)
    b.merge(a)
    assert a.elements() == b.elements()


# ---------- 墓碑回收（GC） ----------

def _state_size(r):
    """统计副本内部 adds/removes 状态占用的字节（dict/set/键/字符串）。"""
    seen = set()

    def sizeof(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        total = sys.getsizeof(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                total += sizeof(k) + sizeof(v)
        elif isinstance(obj, (set, frozenset, list, tuple)):
            for item in obj:
                total += sizeof(item)
        return total

    return sizeof(r._adds) + sizeof(r._removes)


def test_gc_single_replica_frees_all_garbage():
    s = ORSet("solo")
    for i in range(500):
        s.add(i)
        s.remove(i)
    assert s.elements() == set()
    # 只有自己一个存活副本：tag 一进 removes 即被全副本确认，彻底回收
    assert s._adds == {}
    assert s._removes == {}


def test_gc_keeps_tombstone_while_peer_holds_add():
    a = ORSet("A")
    b = ORSet("B")
    a.add("x")
    a.merge(b)            # b 的 adds 里也持有该 tag
    b.merge(a)
    a.remove("x")         # b 还没收到墓碑，仍持有 add(tag)
    assert "x" in a._removes   # 不能回收，否则 b 的 add 会让 x 复活
    b.merge(a)                # b 收到墓碑，摘除本地死 tag
    a.merge(b)                # 全球再无 add(tag)，触发全量回收
    assert a._removes == {}
    assert b._removes == {}
    assert a.elements() == b.elements() == set()


def test_gc_immediate_when_no_peer_ever_held_add():
    a = ORSet("A")
    b = ORSet("B")          # 一直活着，但从没见过 x
    a.add("x")
    a.remove("x")           # 全球只有 a 持有过 tag，已随墓碑抵消
    assert a._removes == {}  # 立即回收是安全的：b 无 add 可传播
    assert a.elements() == b.elements() == set()


def test_gc_safe_with_concurrent_add_wins():
    """GC 不能破坏 add-wins：未被全部确认的墓碑不能挡住后来的新 tag。"""
    a = ORSet("A")
    b = ORSet("B")
    a.add("x")
    a.merge(b)             # b 见过初始 tag t1
    b.merge(a)
    a.remove("x")          # a 删 t1；b 仍持有 add(t1)，墓碑必须保留
    assert "x" in a._removes
    b.add("x")            # b 并发 add 全新 tag t2
    a.merge(b)            # a 拿到 t2（add 赢）；t1 被本地墓碑立即抵消
    assert a.lookup("x")
    b.merge(a)            # b 收到 t1 墓碑并摘除 t1；t1 从此全球灭绝
    a.merge(b)            # 全量 GC：t1 墓碑回收，t2 保留
    assert a.elements() == b.elements() == {"x"}
    assert a._removes == {} and b._removes == {}
    assert a.lookup("x") and b.lookup("x")
    # 再删 t2：两边同步摘除、全网无持有者后，t2 墓碑同样被回收
    a.remove("x")
    assert a.lookup("x") is False
    assert b.lookup("x")          # b 还持有 t2（merge 是单向的）
    b.merge(a)
    b._gc()
    a.merge(b)
    assert not a.lookup("x") and not b.lookup("x")
    assert a._removes == {} and b._removes == {}


def test_gc_offline_peer_then_reconnect():
    """离线副本仍持有 add(tag) 期间墓碑不回收；重连同步后再回收。"""
    a = ORSet("A")
    offline = ORSet("offline")
    a.add("x")
    a.merge(offline)          # offline 持有 add(tag) 后离线
    offline.merge(a)
    a.remove("x")
    assert "x" in a._removes  # offline 还活着且持有 tag，不能回收
    assert offline.lookup("x")
    # offline 重连：收到墓碑，x 消失；随后双向同步使墓碑被安全回收
    offline.merge(a)
    assert not offline.lookup("x")
    a.merge(offline)
    assert a._removes == {} and offline._removes == {}


def test_gc_after_peer_dies():
    """副本被解释器回收后自动退出确认集合，不再阻碍 GC。"""
    import gc as gc_mod
    import weakref

    a = ORSet("A")
    ephemeral = ORSet("ep")
    a.add("x")
    a.merge(ephemeral)
    ephemeral.merge(a)
    a.remove("x")             # ephemeral 还持有 tag，先保留
    assert "x" in a._removes
    ref = weakref.ref(ephemeral)
    del ephemeral
    gc_mod.collect()
    assert ref() is None
    a._gc()                   # 持有 tag 的副本已不存在 -> 回收
    assert a._removes == {}
    assert a.elements() == set()
    # 此后新增/删除/回收依旧正常
    a.add("y")
    a.remove("y")
    assert a._removes == {}
    assert a.elements() == set()


def test_gc_concurrent_workload_reclaims_memory():
    """20k 次增删（单副本），结束后内部状态应被回收到接近空集水平。"""
    s = ORSet("bulk")
    for i in range(20_000):
        s.add(i)
        s.remove(i)
    assert s.elements() == set()
    assert s._adds == {} and s._removes == {}
    # 空状态只剩 dict 自身的固定容量（CPython 大量增删后保留小哈希表，
    # 但仍是与 20k 无关的常量），而不是 MB 级垃圾
    assert _state_size(s) < 1024
