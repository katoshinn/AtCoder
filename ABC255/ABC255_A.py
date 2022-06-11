import io
import sys

_INPUT = """\
6
1 2
1 0
0 1
2 2
1 2
3 4
2 1
90 80
70 60
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  R,C=map(int,input().split())
  A=[list(map(int,input().split())) for _ in range(2)]
  print(A[R-1][C-1])