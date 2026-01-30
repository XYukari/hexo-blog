---
title: CF1879D Sum of XOR Functions
date: 2025-12-08 17:40:50
categories:
  - 做题
tags:
  - 算法
  - 题解
  - 位运算
  - Codeforces
---

[异或和按位处理的典型例题。](https://codeforces.com/problemset/problem/1879/D "异或和按位处理的典型例题。")要求所有子区间异或和乘区间长度的总和，朴素的方法是 $O(n^2)$ 地枚举区间，显然无法通过。

因为涉及异或和，而异或运算不进位，故自然地想到**把 $a_i$ 写成二进制形式，单独研究每一位的贡献**，最后再合并。这是处理此类问题的一般思路。

### 1. 二进制拆分

比方说，对于如下样例，我们把 $a_i$ 写成二进制形式：

```
a[1] = 5  = 0101
a[2] = 11 = 1011
a[3] = 7  = 0111
a[4] = 4  = 0100
```

第 $0$ 位可以提出来，变成一个 01 串 `1110`，第 $1$ 位提出来，得到 `0110`……

### 2. 对每一位求贡献

二进制拆分后，问题也就转化成了：对每个 01 串求所有含有奇数个 $1$ 的区间的区间长度和 $res$（因为只有含奇数个 $1$ 的区间，异或和才为 $1$，长度才会被计入贡献）。

下面用 dp 求 $res$：

要研究区间，往往通过前缀来转化。有哪些情况能使得区间 $[l,r]$ 有奇数个 $1$ 呢？

- 如果区间 $[1,r]$ 有奇数个 $1$，区间 $[1,l]$ 有偶数个 $1$，则区间 $[l,r]$ 有奇数个 $1$；
- 如果区间 $[1,r]$ 有偶数个 $1$，区间 $[1,l]$ 有奇数个 $1$，则区间 $[l,r]$ 有奇数个 $1$。

由此，我们

- 设 $cnt_{i,0/1}$ 表示在 $[1,i]$ 中，有多少个前缀含有偶数/奇数个 $1$；
- 设 $sum_{i,0/1}$ 表示在 $[1,i]$ 中，含有偶数/奇数个 $1$ 的前缀总长度是多少；
- 设 $len_i$ 表示恰好以 $i$ 结尾的含有奇数个 $1$ 的区间的总长度，显然 $res=\sum len_i$。

于是有

$len_i = cnt_{i-1,0}\times i-sum_{i-1,0}$，若 $[1,i]$ 有偶数个 $1$;
$len_i = cnt_{i-1,1}\times i-sum_{i-1,1}$，若 $[1,i]$ 有奇数个 $1$。

这实质上是把所有前缀和作差求区间和的操作放在一起做。

### 3. 合并每一位统计答案

最终的答案 $ans=\sum\limits_{i=0}^{30} res_i\times 2^i$。

```cpp
void solve() {
    int n, ans = 0;
    cin >> n;
    vector<int> a(n + 1), b(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }
    for (int t = 0; t <= 30; t++) {
        for (int i = 1; i <= n; i++) {
            b[i] = (a[i] >> t) & 1;
        }
        int x = 0, res = 0;   // the number of 1s
        int cnt[2] = {1, 0};  // the number of the intervals
        int sum[2] = {0, 0};  // the sum of the intervals' lengthes
        for (int i = 1; i <= n; i++) {
            x = (x + b[i]) % 2;
            res = (res + (LL)cnt[1 - x] * i % mod - sum[1 - x] + mod) % mod;
            cnt[x]++, sum[x] = (sum[x] + i) % mod;
        }
        ans = (ans + (LL)res * ((1 << t) % mod) % mod) % mod;
    }
    cout << ans << endl;
}
```

<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>
