---
title: CF1878D Reverse Madness
date: 2025-12-08 18:04:06
category:
  - 做题
tags:
  - 算法
  - 题解
  - 差分
  - Codeforces
---

[传送门。](https://codeforces.com/contest/1878/problem/D "观察式子发现结论。")有这样一个结论，由 $x$ 得到的反转区间 $[a,b]$ 的对称轴就是 $x$ 所在的题给区间 $[l,r]$ 的对称轴，且 $[a,b]\subset [l,r]$。

这个结论有什么用？如果没有这个结论，我们离线 $q$ 次询问得到的是一系列散乱的反转区间。因为反转区间可能有覆盖、重叠，要得到答案只能暴力地 $O(qn)$ 去反转 $s$，没法整合在一起来降低复杂度。但是一旦有了这个结论，我们就知道：反转区间要么不重叠，要么关于同一个轴对称，我们可以把同轴对称（$x$ 属于同一个区间）的反转区间整合在一起。

如何整合？朴素的想法是把每个反转区间在原序列上标记 +1。如果一个位置被奇数个反转区间覆盖，则与其对称位置交换；否则不交换。因为只有一次查询，所以可用差分+前缀和优化。

```cpp
void solve() {
    int n, k; string s;
    cin >> n >> k >> s;
    vector<int> l(k + 1), r(k + 1), c(n + 1), p(n + 2);
    // c记录x所属区间，p为差分数组
    for (int i = 1; i <= k; i++) cin >> l[i];
    for (int i = 1; i <= k; i++) {
        cin >> r[i];
        for (int j = l[i]; j <= r[i]; j++) c[j] = i;
    }
    int q; cin >> q;
    while (q--) {
        int x; cin >> x;
        int a = min(x, r[c[x]] + l[c[x]] - x);
        int b = max(x, r[c[x]] + l[c[x]] - x);
        p[a]++, p[b + 1]--;
    }
    for (int i = 1; i <= n; i++) p[i] += p[i - 1];
    for (int i = 1; i <= k; i++) {
        for (int j = l[i]; j <= (l[i] + r[i]) / 2; j++)
            if (p[j] & 1) swap(s[j - 1], s[r[i] + l[i] - j - 1]); // 如果被奇数个区间覆盖即交换
    cout << s << endl;
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
