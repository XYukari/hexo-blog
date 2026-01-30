---
title: pbds学习笔记
date: 2025-12-08 17:26:20
category:
  - 技术
tags:
  - C++
  - 数据结构
---

pbds 是 GNU 扩展库的一部分，在 g++ 环境下可以直接使用，clang 下不能使用。

```cpp
#include <bits/extc++.h> // pbds 万能头
using namespace __gnu_cxx;
using namespace __gnu_pbds;
```

### 堆

```cpp
#include <ext/pb_ds/priority_queue.hpp>
using namespace __gnu_pbds;

using heap = __gnu_pbds::priority_queue<int>; // 默认大根堆
heap q;
using small_heap = __gnu_pbds::priority_queue<int, less<int>>; // 小根堆
struct myCmp { bool operator()(node x, node y) { return x.b < y.b; } }; // 也可以在自定义类型里重载运算符
using my_heap = __gnu_pbds::priority_queue<node, myCmp>;

q.top(); q.pop(); q.size(); q.empty(); q.clear(); // 和 STL 完全一样
id = q.push(10); // 效果和 STL 一样，会多返回一个指向插入的元素的迭代器（可以不接收）复杂度 O(1)
q.modify(id, 5); // 直接修改迭代器位置的元素，复杂度均摊 O(log n)
q.erase(id); // 直接删除迭代器位置的元素，均摊 O(log n)
q.join(p); // 把堆p合并到堆q，p清空，复杂度 O(1)
```

modify 的一个应用是在 Dijkstra 中可以直接修改堆内元素，不需要重复插入+ $vis$ 数组记录。

### 哈希表

```cpp
#include <ext/pb_ds/assoc_container.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace __gnu_pbds;
gp_hash_table<int, int> gp_table;
cc_hash_table<int, int> cc_table;

// 用时间设置哈希值防卡，codeforces 常用
const int RANDOM = chrono::high_resolution_clock::now().time_since_epoch().count();
struct chash {
    int operator()(int x) const { return x ^ RANDOM; }
    // 如果要用 string 之类的类型作为 key，可以调用 std::hash
    hash<string> hasher;
    int operator()(string s) const { return hasher(s) ^ RANDOM; }
    // std::pair 没有默认的哈希，需要自定义
    int operator()(pair<int, int> x) const { return x.first* 31 + x.second; }
};
cc_hash_table<int, int, chash> table1;
cc_hash_table<string, int, chash> table2;
```
