import io
import sys

_INPUT = """\
6
999 434
255 15
9999999999 1
"""
sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  from itertools import product
  N,B=map(int,input().split())
  ans=0
  factor=[0]*4
  prime=[2,3,5,7]
  for i in range(4):
    p=prime[i]
    tmp=N
    while tmp>1:
      factor[i]+=1
      tmp//=p
  iter=product(*[list(range(factor[i]+1)) for i in range(4)])
  for v in iter:
    f=1
    for i in range(4):
      f*=pow(prime[i],v[i])
    tmp=str(B+f)
    tmp2=1
    for i in range(len(tmp)):
      tmp2*=int(tmp[i])
    if tmp2==f: ans+=1
  if '0' in str(B): ans+=1
  print(ans)