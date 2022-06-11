import io
import sys

_INPUT = """\
6
6
1 3 5 6 4 2
3 5 1 4 6 2
2
2 1
1 2
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  import sys
  sys.setrecursionlimit(1000000)
  N=int(input())
  P=list(map(int,input().split()))
  I=list(map(int,input().split()))
  P=[P[i]-1 for i in range(N)]
  I=[I[i]-1 for i in range(N)]
  pd={P[i]:i for i in range(N)}
  id={I[i]:i for i in range(N)}
  ans=[[-1,-1] for i in range(N)]
  flg=0
  def rec(ps,pe,ist,ie):
    if pe-ps!=ie-ist:
      flg=1
    if id[P[ps]]==ist: ans[P[ps]][0]=-1
    else:
      if pe>ps:
        ans[P[ps]][0]=P[ps+1]
        if id[P[ps]]<ist or id[P[ps]]>ie: flg=1
        rec(ps+1,ps+id[P[ps]]-ist,ist,id[P[ps]]-1)
    if ps+id[P[ps]]-ist<pe:
      ans[P[ps]][1]=P[ps+id[P[ps]]-ist+1]
      rec(ps+id[P[ps]]-ist+1,pe,id[P[ps]]+1,ie)
  rec(0,N-1,0,N-1)
  if flg==1 or P[0]!=0: print(-1)
  else:
    for i in range(N):
      print(ans[i][0]+1, ans[i][1]+1)