import io
import sys

_INPUT = """\
6
9 2
2 3 3 4 -4 -7 -4 -1
-1 5
20 10
-183260318 206417795 409343217 238245886 138964265 -415224774 -499400499 -313180261 283784093 498751662 668946791 965735441 382033304 177367159 31017484 27914238 757966050 878978971 73210901
-470019195 -379631053 -287722161 -231146414 -84796739 328710269 355719851 416979387 431167199 498905398
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  from collections import defaultdict
  N,M=map(int,input().split())
  S=list(map(int,input().split()))
  X=list(map(int,input().split()))
  add=[0]
  for i in range(N-1):
    add.append(S[i]-add[-1])
  plus,minus=[add[i] for i in range(N) if i%2==0],[add[i] for i in range(N) if i%2==1]
  dp,dm=defaultdict(int),defaultdict(int)
  for i in range(len(plus)):
    dp[plus[i]]+=1
  for i in range(len(minus)):
    dm[minus[i]]+=1
  ans=0
  for k in dp:
    for i in range(M):
      a=X[i]-k
      tmp=0
      for j in range(M):
        if X[j]-a in dp: tmp+=dp[X[j]-a]
        if X[j]+a in dm: tmp+=dm[X[j]+a]
      ans=max(ans,tmp)
  for k in dm:
    for i in range(M):
      a=X[i]+k
      tmp=0
      for j in range(M):
        if X[j]-a in dp: tmp+=dp[X[j]-a]
        if X[j]+a in dm: tmp+=dm[X[j]+a]
      ans=max(ans,tmp)
  print(ans)