import io
import sys

_INPUT = """\
6
4 2
2 3
0 0
0 1
1 2
2 0
2 1
2
-100000 -100000
100000 100000
8 3
2 6 8
-17683 17993
93038 47074
58079 -57520
-41515 -89802
-72739 68805
24324 -73073
71049 72103
47863 19268
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  N,K=map(int,input().split())
  A=list(map(int,input().split()))
  A=[A[i]-1 for i in range(K)]
  l=set(A)
  z=[list(map(int,input().split())) for _ in range(N)]
  ans=0
  for i in range(N):
    if i not in l:
      tmp=10**100
      for j in range(K):
        tmp=min(tmp,((z[i][0]-z[A[j]][0])**2+(z[i][1]-z[A[j]][1])**2)**.5)
      ans=max(ans,tmp)
  print(ans)