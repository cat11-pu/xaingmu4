# orset

Add-wins OR-Set（Observed-Remove Set）CRDT。仅使用 Python 标准库，线程安全。

## 用法

```python
from orset import ORSet

a = ORSet("replica-A")
b = ORSet("replica-B")

a.add("x")
a.lookup("x")      # True
a.elements()       # {"x"}
a.remove("x")      # 只删除自己见过的 add tag
a.lookup("x")      # False
a.remove("never")  # 没见过的元素：no-op，不会挡住并发 add

a.merge(b)         # 把 b 的状态并入 a，返回 a；b 不被修改
```

### add-wins 语义

两个副本并发，一边 add 一边 remove 同一个元素，merge 之后该元素在场：
remove 只埋掉自己见过的 tag，没见过的全新 add tag 不会被顺手删掉。

### merge 性质

merge 是两个副本内部 add/tombstone 集合的逐项并集，因此满足交换律、
结合律、幂等律：任意 merge 顺序、重复 merge、自己 merge 自己，
结果都一致；merge 不会修改传入的 other 副本。

### 墓碑回收

tag 与墓碑只增不减，长时间增删会积累垃圾。本实现做两层回收，
不改变 add-wins 语义与收敛性：

1. tag 一旦进入本副本墓碑集合，立即从有效 add 集合中摘除
   （merge 收到墓碑时同样摘除）；
2. 墓碑只有在**进程内不存在任何仍持有对应 add tag 的存活副本**时
   才被彻底清除（tag 全局唯一、只随并集传播；已被解释器回收的副本
   不可能再把旧 tag 传播回来）。

跨进程/分布式部署的墓碑回收需要显式的组成员与 ack 协议，
不在本库范围内。

## 测试

    pytest -q

或：

    python -m pytest
