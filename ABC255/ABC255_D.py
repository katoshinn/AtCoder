import io
import sys

_INPUT = """\
6
5 3
6 11 2 5 5
5
20
0
10 5
1000000000 314159265 271828182 141421356 161803398 0 777777777 255255255 536870912 998244353
555555555
321654987
1000000000
789456123
0
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  from bisect import bisect_left
  from itertools import accumulate
  N,Q=map(int,input().split())
  A=list(map(int,input().split()))
  A.sort()
  B=list(accumulate(A))
  for _ in range(Q):
    X=int(input())
    idx=bisect_left(A,X)
    if idx==N: print(X*N-B[-1])
    elif idx==0: print(B[-1]-X*N)
    else: print(-B[idx-1]+idx*X-(N-idx)*X+B[-1]-B[idx-1])