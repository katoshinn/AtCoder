import io
import sys

_INPUT = """\
6
12 4 3
3 1
6 5
4 3
10 3 4
7 3
3 1
5 4
6 3
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  from heapq import heappop, heappush
  X,F,N=map(int,input().split())
  camel=[[0,0]]
  for i in range(N):
    x=list(map(int,input().split()))
    camel.append(x)
  camel.append([X,0])
  camel.sort()
  h=[]
  nxt=F
  ans=0
  for i in range(N+1):
    nxt+=-camel[i+1][0]+camel[i][0]
    while nxt<0 and len(h)>0:
      nxt-=heappop(h)
      ans+=1
    if nxt<0: ans=-1; break
    heappush(h,-camel[i+1][1])
  print(ans)