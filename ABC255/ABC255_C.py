import io
import sys

_INPUT = """\
6
6 2 3 3
0 0 0 1
998244353 -10 -20 30
-555555555555555555 -1000000000000000000 1000000 1000000000000
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  X,A,D,N=map(int,input().split())
  if D==0: print(abs(X-A))
  else:
    if D<0:
      A,D=A+D*(N-1),-D
    if X<=A: print(A-X)
    elif X>=A+D*(N-1): print(X-A-D*(N-1))
    else:
      k=(X-A)//D
      print(min(A+D*(k+1)-X,X-A-D*k))