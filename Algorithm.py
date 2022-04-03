git config --global user.email katonyonko@yahoo.co.jp
git config --global user.name "katonyonko"

#再帰関数を書くとき
import sys
sys.setrecursionlimit(1000000)

#n以下の素数を列挙(10**6くらいまでは高速に動く)
import math
def Eratosthenes(n):
  prime=[]
  furui=list(range(2,n+1))
  while furui[0]<math.sqrt(n):
    prime.append(furui[0])
    furui=[i for i in furui if i%furui[0]!=0]
  return prime+furui

#素因数分解はここからコピーするのが多分速い
def gcd(a, b):
  while b: a, b = b, a % b
  return a
def lcm(a, b):
  return a // gcd(a, b) * b
def isPrimeMR(n):
  d = n - 1
  d = d // (d & -d)
  L = [2, 7, 61] if n < 1<<32 else [2, 3, 5, 7, 11, 13, 17] if n < 1<<48 else [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
  for a in L:
    t = d
    y = pow(a, t, n)
    if y == 1: continue
    while y != n - 1:
      y = y * y % n
      if y == 1 or t == n - 1: return 0
      t <<= 1
  return 1
def findFactorRho(n):
  m = 1 << n.bit_length() // 8
  for c in range(1, 99):
      f = lambda x: (x * x + c) % n
      y, r, q, g = 2, 1, 1, 1
      while g == 1:
          x = y
          for i in range(r):
              y = f(y)
          k = 0
          while k < r and g == 1:
              ys = y
              for i in range(min(m, r - k)):
                  y = f(y)
                  q = q * abs(x - y) % n
              g = gcd(q, n)
              k += m
          r <<= 1
      if g == n:
          g = 1
          while g == 1:
              ys = f(ys)
              g = gcd(abs(x - ys), n)
      if g < n:
          if isPrimeMR(g): return g
          elif isPrimeMR(n // g): return n // g
          return findFactorRho(g)
def primeFactor(n):
  i = 2
  ret = {}
  rhoFlg = 0
  while i * i <= n:
      k = 0
      while n % i == 0:
          n //= i
          k += 1
      if k: ret[i] = k
      i += i % 2 + (3 if i % 3 == 1 else 1)
      if i == 101 and n >= 2 ** 20:
          while n > 1:
              if isPrimeMR(n):
                  ret[n], n = 1, 1
              else:
                  rhoFlg = 1
                  j = findFactorRho(n)
                  k = 0
                  while n % j == 0:
                      n //= j
                      k += 1
                  ret[j] = k
  if n > 1: ret[n] = 1
  if rhoFlg: ret = {x: ret[x] for x in sorted(ret)}
  return ret
def divisors(N):
  pf = primeFactor(N)
  ret = [1]
  for p in pf:
      ret_prev = ret
      ret = []
      for i in range(pf[p]+1):
          for r in ret_prev:
              ret.append(r * (p ** i))
  return sorted(ret)

#約数全列挙
def make_divisors(n):
  lower_divisors , upper_divisors = [], []
  i = 1
  while i*i <= n:
      if n % i == 0:
          lower_divisors.append(i)
          if i != n // i:
              upper_divisors.append(n//i)
      i += 1
  return lower_divisors + upper_divisors[::-1]

#numpyの高速化
from numba import njit
@njit(cache=True)

#累積和
A=np.array([1,2,3,4,5])
np.cumsum(A) #[1,3,6,10]
np.sum(A)-np.cumsum(np.append(0, A[:len(A)-1])) #逆累積和[15, 14, 12,  9,  5]

#複数行の読み込み
import sys
s = sys.stdin.readlines()

#dequeueの方が若干早いらしい
from collections import deque
dq = deque()
# 後ろへデータを挿入
dq.append(データ)
# 前へデータを挿入
dq.appendleft(データ)
# 後ろのデータの取り出し
dq.pop()
# 前のデータを取り出し
dq.popleft()
# dequeが空になるまで繰り返し
while dq:

#bit の性質a + b − (a xor b) = 2 × (a and b)
#bit全探索をやる場合はitertools.product()を使いましょう
import itertools
import pprint
l1 = ['a', 'b', 'c']
l2 = ['X', 'Y', 'Z']
p = itertools.product(l1, l2)
for v in p:
  print(v)
p2 = itertools.product(l1, repeat=2) #こんな指定もできる

#Union Find
class UnionFind():
  def __init__(self, n):
    self.n = n
    self.parents = [-1] * n
  def find(self, x):
    if self.parents[x] < 0:
      return x
    else:
      self.parents[x] = self.find(self.parents[x])
      return self.parents[x]
  def union(self, x, y):
    x = self.find(x)
    y = self.find(y)
    if x == y:
      return
    if self.parents[x] > self.parents[y]:
       x, y = y, x
    self.parents[x] += self.parents[y]
    self.parents[y] = x
  def size(self, x):
    return -self.parents[self.find(x)]
  def same(self, x, y):
    return self.find(x) == self.find(y)
  def members(self, x):
    root = self.find(x)
    return [i for i in range(self.n) if self.find(i) == root]
  def roots(self):
    return [i for i, x in enumerate(self.parents) if x < 0]
  def group_count(self):
    return len(self.roots())
  def all_group_members(self):
    return {r: self.members(r) for r in self.roots()}
  def __str__(self):
    return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())

#Segment Tree
class SegTree:
    X_unit = 1 << 30
    X_f = min
 
    def __init__(self, N):
        self.N = N
        self.X = [self.X_unit] * (N + N)
 
    def build(self, seq):
        for i, x in enumerate(seq, self.N):
            self.X[i] = x
        for i in range(self.N - 1, 0, -1):
            self.X[i] = self.X_f(self.X[i << 1], self.X[i << 1 | 1])
 
    def set_val(self, i, x):
        i += self.N
        self.X[i] = x
        while i > 1:
            i >>= 1
            self.X[i] = self.X_f(self.X[i << 1], self.X[i << 1 | 1])
 
    def fold(self, L, R):
        L += self.N
        R += self.N
        vL = self.X_unit
        vR = self.X_unit
        while L < R:
            if L & 1:
                vL = self.X_f(vL, self.X[L])
                L += 1
            if R & 1:
                R -= 1
                vR = self.X_f(self.X[R], vR)
            L >>= 1
            R >>= 1
        return self.X_f(vL, vR)

#遅延セグ木 0-indexed、演算はmaxの例を書いているが、変更する場合はunitとclassmethodの部分を変える
#https://maspypy.com/segment-tree-%e3%81%ae%e3%81%8a%e5%8b%89%e5%bc%b72 1-indexedと書かれているが、以下のコードは0-indexed
class LazySegTree:
    X_unit = 0
    A_unit = 0

    @classmethod
    def X_f(cls, x, y):
        return max(x,y)

    @classmethod
    def A_f(cls, x, y):
        return max(x,y)

    @classmethod
    def operate(cls, x, y):
        return max(x,y)

    def __init__(self, N):
        self.N = N
        self.X = [self.X_unit] * (N + N)
        self.A = [self.A_unit] * (N + N)

    def build(self, seq):
        for i, x in enumerate(seq, self.N):
            self.X[i] = x
        for i in range(self.N - 1, 0, -1):
            self.X[i] = self.X_f(self.X[i << 1], self.X[i << 1 | 1])

    def _eval_at(self, i):
        return self.operate(self.X[i], self.A[i])

    def _propagate_at(self, i):
        self.X[i] = self._eval_at(i)
        self.A[i << 1] = self.A_f(self.A[i << 1], self.A[i])
        self.A[i << 1 | 1] = self.A_f(self.A[i << 1 | 1], self.A[i])
        self.A[i] = self.A_unit

    def _propagate_above(self, i):
        H = i.bit_length() - 1
        for h in range(H, 0, -1):
            self._propagate_at(i >> h)

    def _recalc_above(self, i):
        while i > 1:
            i >>= 1
            self.X[i] = self.X_f(self._eval_at(i << 1), self._eval_at(i << 1 | 1))

    def set_val(self, i, x):
        i += self.N
        self._propagate_above(i)
        self.X[i] = x
        self.A[i] = self.A_unit
        self._recalc_above(i)

    def fold(self, L, R):
        L += self.N
        R += self.N
        self._propagate_above(L // (L & -L))
        self._propagate_above(R // (R & -R) - 1)
        vL = self.X_unit
        vR = self.X_unit
        while L < R:
            if L & 1:
                vL = self.X_f(vL, self._eval_at(L))
                L += 1
            if R & 1:
                R -= 1
                vR = self.X_f(self._eval_at(R), vR)
            L >>= 1
            R >>= 1
        return self.X_f(vL, vR)

    def operate_range(self, L, R, x):
        L += self.N
        R += self.N
        L0 = L // (L & -L)
        R0 = R // (R & -R) - 1
        self._propagate_above(L0)
        self._propagate_above(R0)
        while L < R:
            if L & 1:
                self.A[L] = self.A_f(self.A[L], x)
                L += 1
            if R & 1:
                R -= 1
                self.A[R] = self.A_f(self.A[R], x)
            L >>= 1
            R >>= 1
        self._recalc_above(L0)
        self._recalc_above(R0)

#累積和
import itertools
list(itertools.accumulate(l))

#BIT(Binary Indexed Tree, Fenwick Treeとも, 添え字が0から始まる形で管理していることに注意)
class BIT:
    def __init__(self, n):
        self._n = n
        self.data = [0] * n
    def add(self, p, x):
        assert 0 <= p < self._n
        p += 1
        while p <= self._n:
            self.data[p - 1] += x
            p += p & -p
    #合計にはrを含む
    def sum(self, l, r):
        assert 0 <= l <= r <= self._n
        return self._sum(r) - self._sum(l)
    def _sum(self, r):
        s = 0
        while r > 0:
            s += self.data[r - 1]
            r -= r & -r
        return s

#mod 3などの小さい素数でのコンビネーションの求め方
https://atcoder.jp/contests/arc117/editorial/1113

#Dijkstra
from heapq import heappop,heappush
def Dijkstra(G,s):
  done=[False]*len(G)
  inf=10**20
  C=[inf]*len(G)
  C[s]=0
  h=[]
  heappush(h,(0,s))
  while h:
    x,y=heappop(h)
    if done[y]:
      continue
    done[y]=True
    for v in G[y]:
      if C[v[1]]>C[y]+v[0]:
        C[v[1]]=C[y]+v[0]
        heappush(h,(C[v[1]],v[1]))
  return C

#Warshall-Floyd
# cost[i][j]: 頂点v_iから頂点v_jへ到達するための辺コストの和
def WF(cost):
    for k in range(len(cost)):
        for i in range(len(cost)):
            for j in range(len(cost)):
                if cost[i][k]!=INF and cost[k][j]!=INF:
                    cost[i][j] = min(cost[i][j], cost[i][k] + cost[k][j])
    return cost

#オイラーツアー
def EulerTour(G, s):
    depth=[-1]*len(G)
    depth[s]=0
    done = [0]*len(G)
    Q = [~s, s] # 根をスタックに追加
    parent=[-1]*len(G)
    ET = []
    left=[-1]*len(G)
    while Q:
        i = Q.pop()
        if i >= 0: # 行きがけの処理
            done[i] = 1
            if left[i]==-1: left[i]=len(ET)
            ET.append(i)
            for a in G[i][::-1]:
                if done[a]: continue
                depth[a]=depth[i]+1
                parent[a]=i
                Q.append(~a) # 帰りがけの処理をスタックに追加
                Q.append(a) # 行きがけの処理をスタックに追加
        else: # 帰りがけの処理
            ET.append(parent[~i])
    return ET[:-1], left, depth, parent

#LCA(最小共通祖先)ここは準備
S,F,depth,parent=EulerTour(G,0)
INF = (len(G), None)
M = 2*len(G)
M0 = 2**(M-1).bit_length()
data = [INF]*(2*M0)
for i, v in enumerate(S):
    data[M0-1+i] = (depth[v], i)
for i in range(M0-2, -1, -1):
    data[i] = min(data[2*i+1], data[2*i+2])
#LCAの計算 (generatorで最小値を求める)
def _query(a, b):
    yield INF
    a += M0; b += M0
    while a < b:
        if b & 1:
            b -= 1
            yield data[b-1]
        if a & 1:
            yield data[a-1]
            a += 1
        a >>= 1; b >>= 1
# LCAの計算 (外から呼び出す関数)
def LCA(u, v):
    fu = F[u]; fv = F[v]
    if fu > fv:
        fu, fv = fv, fu
    return S[min(_query(fu, fv+1))[1]]

#強連結成分分解
#labels:強連結成分のラベル番号, lb_cnt:全強連結成分数, build:graphから強連結成分分解を実行, construct:強連結成分によるDAGと各成分の頂点リストのリストをこの順に抽出
class SCC:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.rev_graph = [[] for _ in range(n)]
        self.labels = [-1] * n
        self.lb_cnt = 0

    def add_edge(self, v, nxt_v):
        self.graph[v].append(nxt_v)
        self.rev_graph[nxt_v].append(v)

    def build(self):
        self.post_order = []
        self.used = [False] * self.n
        for v in range(self.n):
            if not self.used[v]:
                self._dfs(v)
        for v in reversed(self.post_order):
            if self.labels[v] == -1:
                self._rev_dfs(v)
                self.lb_cnt += 1

    def _dfs(self, v):
        stack = [v, 0]
        while stack:
            v, idx = stack[-2:]
            if not idx and self.used[v]:
                stack.pop()
                stack.pop()
            else:
                self.used[v] = True
                if idx < len(self.graph[v]):
                    stack[-1] += 1
                    stack.append(self.graph[v][idx])
                    stack.append(0)
                else:
                    stack.pop()
                    self.post_order.append(stack.pop())

    def _rev_dfs(self, v):
        stack = [v]
        self.labels[v] = self.lb_cnt
        while stack:
            v = stack.pop()
            for nxt_v in self.rev_graph[v]:
                if self.labels[nxt_v] != -1:
                    continue
                stack.append(nxt_v)
                self.labels[nxt_v] = self.lb_cnt

    def construct(self):
        self.dag = [[] for i in range(self.lb_cnt)]
        self.groups = [[] for i in range(self.lb_cnt)]
        for v, lb in enumerate(self.labels):
            for nxt_v in self.graph[v]:
                nxt_lb = self.labels[nxt_v]
                if lb == nxt_lb:
                    continue
                self.dag[lb].append(nxt_lb)
            self.groups[lb].append(v)
        return self.dag, self.groups

#popcount
def Popcount(n):
    c = (n & 0x5555555555555555) + ((n>>1) & 0x5555555555555555)
    c = (c & 0x3333333333333333) + ((c>>2) & 0x3333333333333333)
    c = (c & 0x0f0f0f0f0f0f0f0f) + ((c>>4) & 0x0f0f0f0f0f0f0f0f)
    c = (c & 0x00ff00ff00ff00ff) + ((c>>8) & 0x00ff00ff00ff00ff)
    c = (c & 0x0000ffff0000ffff) + ((c>>16) & 0x0000ffff0000ffff)
    c = (c & 0x00000000ffffffff) + ((c>>32) & 0x00000000ffffffff)
    return c

#拡張ユークリッド ax+by=gcd(a,b)となるようなx,yを求める。同時にgcdも求める
#aとbが互いに素な時、xはmod bにおいてのaの逆元
def ExtGCD(a, b):
    if b:
        g, y, x = ExtGCD(b, a % b)
        y -= (a // b)*x
        return g, x, y
    return a, 1, 0

# 円同士の交点 円1: 中心(x1,y1) 半径r1 と 円2: 中心(x2,y2) 半径r2 の2つの円の交点
def CirclesCrossPoints(x1, y1, r1, x2, y2, r2):
    eps=10**(-6)
    r=((x1-x2)**2+(y1-y2)**2)**.5
    if r<eps or r+eps>r1+r2 or r1+eps>r+r2 or r2+eps>r+r1:
        return ()
    else:
        rr0 = (x2 - x1)**2 + (y2 - y1)**2
        xd = x2 - x1
        yd = y2 - y1
        rr1 = r1**2; rr2 = r2**2
        cv = (rr0 + rr1 - rr2)
        sv = (4*rr0*rr1 - cv**2)**.5
        return (
            (x1 + (cv*xd - sv*yd)/(2.*rr0), y1 + (cv*yd + sv*xd)/(2.*rr0)),
            (x1 + (cv*xd + sv*yd)/(2.*rr0), y1 + (cv*yd - sv*xd)/(2.*rr0)),
        )

# 頂点から円への接点 中心 (x1,y1) 半径 r1 に対する、点 (x2,y2) から引いた接線がなす接点
def circle_tangent_points(x1, y1, r1, x2, y2):
    dd = (x1 - x2)**2 + (y1 - y2)**2
    r2 = (dd - r1**2)**.5
    return circles_cross_points(x1, y1, r1, x2, y2, r2)

#円の外心を求める
def Gaishin(x1,y1,x2,y2,x3,y3):
    eps=10**(-6)
    if abs((x2-x1)*(y3-y2)-(x3-x2)*(y2-y1))<eps:
      return (max(x1,x2,x3)-min(x1,x2,x3))/2, (max(y1,y2,y3)-min(y1,y2,y3))/2, 0
    py = ((x3 - x1) * (x1**2 + y1**2 - x2**2 - y2**2) - (x2 - x1) * (x1**2 + y1**2 - x3**2- y3**2)) / (2 * (x3 - x1)*(y1 - y2) - 2 * (x2 - x1) * (y1 - y3))
    if x2 == x1:
        px = (2 * (y1 - y3) * py - x1**2 - y1**2 + x3**2 + y3**2) / (2 * (x3 - x1))
    else:
        px = (2 * (y1 - y2) * py - x1**2 - y1**2 + x2**2 + y2**2) / (2 * (x2 - x1))
    r = ((px - x1)**2+(py - y1)**2)**.5
    return px,py,r

#高速フーリエ変換による畳み込み 詳しくはmaspyさんのページhttps://maspypy.com/%E6%95%B0%E5%AD%A6%E3%83%BBnumpy-%E9%AB%98%E9%80%9F%E3%83%95%E3%83%BC%E3%83%AA%E3%82%A8%E5%A4%89%E6%8F%9Bfft%E3%81%AB%E3%82%88%E3%82%8B%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF
# (1 + 2x + 3x^2)(4 + 5x + 6x^2) = 4 + 13x + 28x^2 + 27x^3 + 18x^4
# f = np.array([1, 2, 3], np.int64)
# g = np.array([4, 5, 6], np.int64)
# h = convolve(f, g)
#print(h)[ 4 13 28 27 18]
def Convolve(f, g):
  fft_len = 1
  while 2 * fft_len < len(f) + len(g) - 1:
      fft_len *= 2
  fft_len *= 2
  Ff = np.fft.rfft(f, fft_len)
  Fg = np.fft.rfft(g, fft_len)
  Fh = Ff * Fg
  h = np.fft.irfft(Fh, fft_len)
  h = np.rint(h).astype(np.int64)
  return h[:len(f) + len(g) - 1]

#平衡二分木 nは追加する要素の数字が2**n以下になるような数字として設定。それなりに大きくても構築はO(1)なので遅くはならない。
class BalancingTree:
    def __init__(self, n):
        self.N = n
        self.root = self.node(1<<n, 1<<n)

    def append(self, v):# v を追加（その時点で v はない前提）
        v += 1
        nd = self.root
        while True:
            if v == nd.value:
                # v がすでに存在する場合に何か処理が必要ならここに書く
                return 0
            else:
                mi, ma = min(v, nd.value), max(v, nd.value)
                if mi < nd.pivot:
                    nd.value = ma
                    if nd.left:
                        nd = nd.left
                        v = mi
                    else:
                        p = nd.pivot
                        nd.left = self.node(mi, p - (p&-p)//2)
                        break
                else:
                    nd.value = mi
                    if nd.right:
                        nd = nd.right
                        v = ma
                    else:
                        p = nd.pivot
                        nd.right = self.node(ma, p + (p&-p)//2)
                        break

    def leftmost(self, nd):
        if nd.left: return self.leftmost(nd.left)
        return nd

    def rightmost(self, nd):
        if nd.right: return self.rightmost(nd.right)
        return nd

    def find_l(self, v): # vより真に小さいやつの中での最大値（なければ-1）
        v += 1
        nd = self.root
        prev = 0
        if nd.value < v: prev = nd.value
        while True:
            if v <= nd.value:
                if nd.left:
                    nd = nd.left
                else:
                    return prev - 1
            else:
                prev = nd.value
                if nd.right:
                    nd = nd.right
                else:
                    return prev - 1

    def find_r(self, v): # vより真に大きいやつの中での最小値（なければRoot）
        v += 1
        nd = self.root
        prev = 0
        if nd.value > v: prev = nd.value
        while True:
            if v < nd.value:
                prev = nd.value
                if nd.left:
                    nd = nd.left
                else:
                    return prev - 1
            else:
                if nd.right:
                    nd = nd.right
                else:
                    return prev - 1

    @property
    def max(self):
        return self.find_l((1<<self.N)-1)

    @property
    def min(self):
        return self.find_r(-1)

    def delete(self, v, nd = None, prev = None): # 値がvのノードがあれば削除（なければ何もしない）
        v += 1
        if not nd: nd = self.root
        if not prev: prev = nd
        while v != nd.value:
            prev = nd
            if v <= nd.value:
                if nd.left:
                    nd = nd.left
                else:
                    #####
                    return
            else:
                if nd.right:
                    nd = nd.right
                else:
                    #####
                    return
        if (not nd.left) and (not nd.right):
            if not prev.left:
                prev.right = None
            elif not prev.right:
                prev.left = None
            else:
                if nd.pivot == prev.left.pivot:
                    prev.left = None
                else:
                    prev.right = None

        elif nd.right:
            # print("type A", v)
            nd.value = self.leftmost(nd.right).value
            self.delete(nd.value - 1, nd.right, nd)    
        else:
            # print("type B", v)
            nd.value = self.rightmost(nd.left).value
            self.delete(nd.value - 1, nd.left, nd)

    def __contains__(self, v: int) -> bool:
        return self.find_r(v - 1) == v

    class node:
        def __init__(self, v, p):
            self.value = v
            self.pivot = p
            self.left = None
            self.right = None

    def debug(self):
        def debug_info(nd_):
            return (nd_.value - 1, nd_.pivot - 1, nd_.left.value - 1 if nd_.left else -1, nd_.right.value - 1 if nd_.right else -1)

        def debug_node(nd):
            re = []
            if nd.left:
                re += debug_node(nd.left)
            if nd.value: re.append(debug_info(nd))
            if nd.right:
                re += debug_node(nd.right)
            return re
        print("Debug - root =", self.root.value - 1, debug_node(self.root)[:50])

    def debug_list(self):
        def debug_node(nd):
            re = []
            if nd.left:
                re += debug_node(nd.left)
            if nd.value: re.append(nd.value - 1)
            if nd.right:
                re += debug_node(nd.right)
            return re
        return debug_node(self.root)[:-1]

#LIS(最長増加部分列の長さと具体的な最長増加部分列を求める)
from bisect import bisect_left
def LIS(A: list):
    L = [A[0]]
    ID=[0]*len(A)
    for i in range(1,len(A)):
        if A[i] > L[-1]:
            L.append(A[i])
            ID[i]=len(L)-1
        else:
            tmp=bisect_left(L, A[i])
            L[tmp] = A[i]
            ID[i]=tmp
    L2=[]
    L3=[]
    m=len(L)-1
    for i in range(len(A)-1,-1,-1):
    if ID[i]==m:
        L2.append(A[i])
        L3.append(i)
        m-=1
    return len(L), L2[::-1], L3[::-1] #それぞれ最長増加部分列の長さ、復元した部分列、そのインデックス

#WLIS(広義の最長増加部分列の長さと具体的な広義最長増加部分列を求める)
from bisect import bisect_right
def WLIS(A: list):
    L = [A[0]]
    ID=[0]*len(A)
    for i in range(1,len(A)):
      if A[i] >= L[-1]:
        L.append(A[i])
        ID[i]=len(L)-1
      else:
        tmp=bisect_right(L, A[i])
        L[tmp] = A[i]
        ID[i]=tmp
    L2=[]
    L3=[]
    m=len(L)-1
    for i in range(len(A)-1,-1,-1):
      if ID[i]==m:
        L2.append(A[i])
        L3.append(i)
        m-=1
    return len(L), L2[::-1], L3[::-1]

#Kruskal法 max_vに頂点数,edgesに辺(重み、結んでいる２つの頂点（順不同）で入れる)
class UnionFind():
  def __init__(self, n):
    self.n = n
    self.parents = [-1] * n
  def find(self, x):
    if self.parents[x] < 0:
      return x
    else:
      self.parents[x] = self.find(self.parents[x])
      return self.parents[x]
  def union(self, x, y):
    x = self.find(x)
    y = self.find(y)
    if x == y:
      return
    if self.parents[x] > self.parents[y]:
       x, y = y, x
    self.parents[x] += self.parents[y]
    self.parents[y] = x
  def size(self, x):
    return -self.parents[self.find(x)]
  def same(self, x, y):
    return self.find(x) == self.find(y)
  def members(self, x):
    root = self.find(x)
    return [i for i in range(self.n) if self.find(i) == root]
  def roots(self):
    return [i for i, x in enumerate(self.parents) if x < 0]
  def group_count(self):
    return len(self.roots())
  def all_group_members(self):
    return {r: self.members(r) for r in self.roots()}
  def __str__(self):
    return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())
def Kruskal(G):
  edges=set()
  for i in range(len(G)):
    for j in range(len(G[i])):
      c,k=G[i][j]
      l,m=min(i,k),max(i,k)
      edges.add((c,l,m))
  edges=list(edges)
  edges.sort()
  uf = UnionFind(len(G))
  mst = [] #最小全域木の辺すべて
  weight=0 #最小全域木の重さ
  for edge in edges:
    if not uf.same(edge[1], edge[2]):
      uf.union(edge[1], edge[2])
      mst.append(edge)
      weight+=edge[0]
  return mst,weight

#Prim法
from heapq import heappop,heappush
def Prim(G):
  N=len(G)
  m=set()
  m.add(0)
  mst = [] #最小全域木の辺すべて
  weight=0 #最小全域木の重さ
  h=[]
  for c,v in G[0]:
    if v not in m:
      heappush(h,(c,0,v))
  while len(h)>0:
    c,u,v=heappop(h)
    if v not in m:
      mst.append((c,u,v))
      weight+=c
      m.add(v)
      for c2,v2 in G[v]:
        if v2 not in m:
          heappush(h,(c2,v,v2))
  return mst,weight

#DFS
def dfs(G,r=0):
    used=[False]*len(G)
    parent=[-1]*len(G)
    st=[]
    st.append(r)
    while st:
        x=st.pop()
        if used[x]==True:
            continue
        used[x]=True
        for v in G[x]:
            if v==parent[x]:
                continue
            parent[v]=x
            st.append(v)
    return parent

#BFS
from collections import deque
def bfs(G,s):
  inf=10**30
  D=[inf]*len(G)
  D[s]=0
  dq=deque()
  dq.append(s)
  while dq:
    x=dq.popleft()
    for y in G[x]:
      if D[y]>D[x]+1:
        D[y]=D[x]+1
        dq.append(y)
  return D

#01BFS Gは隣接頂点リストで、(隣接頂点,重み(0or1))が入っている
from collections import deque
def bfs01(G,s):
  inf=10**20
  dist = [inf]*len(G)
  dist[s] = 0
  que = deque()
  que.append(s)
  while que:
    i = que.popleft()
    for j, c in G[i]:
        d = dist[i] + c
        if d < dist[j]:
            dist[j] = d
            if c == 1:
                que.append(j)
            else:
                que.appendleft(j)
  return dist

# 複数の区間が与えられた時の和集合(Xは区間のリストのリスト⇨例えばX=[[1,3],[10,12],[2,8]]なら[[1,8],[10,12]]が返る。1以上3以下か10以上12以下か2以上8以下の区間は1以上8以下か10以上12以下として表せる)
def Union(X):
    tmp=[]
    ans=[]
    for l,r in X:
        tmp.append([l,1])
        tmp.append([r+1,-1])
    tmp.sort()
    i=0
    l=tmp[0][0]
    flag=tmp[0][1]
    while i<len(tmp):
        if i<len(tmp)-1:
            while i<len(tmp)-1 and tmp[i+1][0]==tmp[i][0]:
                flag+=tmp[i+1][1]
                i+=1
        if i<len(tmp):
            if flag==0:
                ans.append([l,tmp[i][0]-1])
                if i<len(tmp)-1:
                    l=tmp[i+1][0]
            if i<len(tmp)-1:
                flag+=tmp[i+1][1]
        i+=1
    size=0
    for x,y in ans:
        size+=y-x+1
    return ans,size

#Combination
F=[1]
for i in range(N):
    F.append(F[-1]*(i+1)%mod)
I=[pow(F[i],mod-2,mod) for i in range(N+1)]

#行列掃き出し法(Aはリスト、Nは桁数（Aのサイズではない）で、Aの要素を2進数表示したときに上の桁から掃き出しでまとめていく関数)
#A=[2,3]であれば、2進表示で10, 11だが、10は残して2の位の1が被っている11については10とxorを取って01に変更する。
def elimination(A,N=60):
  res=[]
  for i in reversed(range(N)):
    f=0
    for j in range(len(A)):
      if (A[j]>>i)&1==1:
        if f==0:
          x=A[j]
          t=j
          f=1
        else:
          A[j]^=x
    if f==1:
      res.append(x)
      A.pop(t)
    else:
      res.append(0)
  return res

#z algorithm
def z_algorithm(s):
    N = len(s)
    Z_alg = [0]*N

    Z_alg[0] = N
    i = 1
    j = 0
    while i < N:
        while i+j < N and s[j] == s[i+j]:
            j += 1
        Z_alg[i] = j
        if j == 0:
            i += 1
            continue
        k = 1
        while i+k < N and k + Z_alg[k]<j:
            Z_alg[i+k] = Z_alg[k]
            k += 1
        i += k
        j -= k
    return Z_alg

#10進数をn進数にする関数
def Base_10_to_n(X, n):
    if (int(X/n)):
        return Base_10_to_n(int(X/n), n)+str(X%n)
    return str(X%n)

#Topological Sort DAGでない入力でも良いが、その場合は-1が出力される Gは行先を入れたグラフ
from heapq import heappop,heappush
def TopologicalSort(G):
  G2=[set() for _ in range(len(G))]
  for i in range(len(G)):
    for v in G[i]:
      G2[v].add(i)
  res=[]
  h=[]
  for i in range(len(G)):
    if len(G2[i])==0:
      heappush(h,i)
  while len(h):
    x=heappop(h)
    res.append(x)
    for y in G[x]:
      G2[y].remove(x)
      if len(G2[y])==0:
        heappush(h,y)
  if len(res)==len(G):
    return res
  else:
    return -1

#サイクルの検出(Topological SortしてDAGじゃなかった場合、rから始まるサイクルを構成できる)
def dfs(G,r=0):
    used=[False]*len(G) #行きがけ
    finished=[False]*len(G) #帰りがけ
    parent=[-1]*len(G)
    st=[]
    st.append([r,1])
    st.append([r,0])
    cycle=[]
    while st:
        x,y=st.pop()
        if y==0:
            cycle.append(x)
            used[x]=True
            for v in G[x]:
                if used[v]==True and finished[v]==False:
                    return 0,cycle #サイクルありの場合
                parent[v]=x
                st.append([v,1])
                st.append([v,0])
        else:
            cycle.pop()
            finished[x]=True
    return None

#multiset
import math
from bisect import bisect_left, bisect_right, insort
from typing import Generic, Iterable, Iterator, TypeVar, Union, List
T = TypeVar('T')
class SortedMultiset(Generic[T]):
  BUCKET_RATIO = 50
  REBUILD_RATIO = 170
 
  def _build(self, a=None) -> None:
      "Evenly divide `a` into buckets."
      if a is None: a = list(self)
      size = self.size = len(a)
      bucket_size = int(math.ceil(math.sqrt(size / self.BUCKET_RATIO)))
      self.a = [a[size * i // bucket_size : size * (i + 1) // bucket_size] for i in range(bucket_size)]
 
  def __init__(self, a: Iterable[T] = []) -> None:
      "Make a new SortedMultiset from iterable. / O(N) if sorted / O(N log N)"
      a = list(a)
      if not all(a[i] <= a[i + 1] for i in range(len(a) - 1)):
          a = sorted(a)
      self._build(a)
 
  def __iter__(self) -> Iterator[T]:
      for i in self.a:
          for j in i: yield j
 
  def __reversed__(self) -> Iterator[T]:
      for i in reversed(self.a):
          for j in reversed(i): yield j
 
  def __len__(self) -> int:
      return self.size
 
  def __repr__(self) -> str:
      return "SortedMultiset" + str(self.a)
 
  def __str__(self) -> str:
      s = str(list(self))
      return "{" + s[1 : len(s) - 1] + "}"
 
  def _find_bucket(self, x: T) -> List[T]:
      "Find the bucket which should contain x. self must not be empty."
      for a in self.a:
          if x <= a[-1]: return a
      return a
 
  def __contains__(self, x: T) -> bool:
      if self.size == 0: return False
      a = self._find_bucket(x)
      i = bisect_left(a, x)
      return i != len(a) and a[i] == x
 
  def count(self, x: T) -> int:
      "Count the number of x."
      return self.index_right(x) - self.index(x)
 
  def add(self, x: T) -> None:
      "Add an element. / O(√N)"
      if self.size == 0:
          self.a = [[x]]
          self.size = 1
          return
      a = self._find_bucket(x)
      insort(a, x)
      self.size += 1
      if len(a) > len(self.a) * self.REBUILD_RATIO:
          self._build()
 
  def discard(self, x: T) -> bool:
      "Remove an element and return True if removed. / O(√N)"
      if self.size == 0: return False
      a = self._find_bucket(x)
      i = bisect_left(a, x)
      if i == len(a) or a[i] != x: return False
      a.pop(i)
      self.size -= 1
      if len(a) == 0: self._build()
      return True
 
  def lt(self, x: T) -> Union[T, None]:
      "Find the largest element < x, or None if it doesn't exist."
      for a in reversed(self.a):
          if a[0] < x:
              return a[bisect_left(a, x) - 1]
 
  def le(self, x: T) -> Union[T, None]:
      "Find the largest element <= x, or None if it doesn't exist."
      for a in reversed(self.a):
          if a[0] <= x:
              return a[bisect_right(a, x) - 1]
 
  def gt(self, x: T) -> Union[T, None]:
      "Find the smallest element > x, or None if it doesn't exist."
      for a in self.a:
          if a[-1] > x:
              return a[bisect_right(a, x)]
 
  def ge(self, x: T) -> Union[T, None]:
      "Find the smallest element >= x, or None if it doesn't exist."
      for a in self.a:
          if a[-1] >= x:
              return a[bisect_left(a, x)]
 
  def __getitem__(self, x: int) -> T:
      "Return the x-th element, or IndexError if it doesn't exist."
      if x < 0: x += self.size
      if x < 0: raise IndexError
      for a in self.a:
          if x < len(a): return a[x]
          x -= len(a)
      raise IndexError
 
  def index(self, x: T) -> int:
      "Count the number of elements < x."
      ans = 0
      for a in self.a:
          if a[-1] >= x:
              return ans + bisect_left(a, x)
          ans += len(a)
      return ans
 
  def index_right(self, x: T) -> int:
      "Count the number of elements <= x."
      ans = 0
      for a in self.a:
          if a[-1] > x:
              return ans + bisect_right(a, x)
          ans += len(a)
      return ans

#BellmanFord
def BellmanFord(G,s=0):
    inf=10**20
    D=[inf]*len(G)
    D[0]=0
    for i in range(len(G)-1):
      for j in range(len(G)):
        for c,v in G[j]:
          if D[j]+c<D[v]:
            D[v]=D[j]+c
    cycle=[0]*len(G)
    for j in range(len(G)):
      for c,v in G[j]:
        if D[j]+c<D[v]: cycle[v]=1
    for i in range(len(G)-1):
      for j in range(len(G)):
        if cycle[j]==1:
          for c,v in G[j]:
            cycle[v]=1
    for i in range(len(G)):
      if cycle[i]==1: D[i]='-inf'
    return D