import io
import sys

_INPUT = """\
6
1 2 19 16 0.10
0010000100110011100
1000001000001000001
0000000000000000010
0100000000000110100
0000000000000000100
1000101100000101010
0010001011000110000
0000001001000000000
0000000100010001001
0010010000100000001
0001000010000100000
0011010000000001000
0000000101010100000
0000001000000100010
0110100010000000000
0010011101000101000
0000100110010000000
0010000101101000010
1001000000000000000
1000110000000000000
00000001000000000100
00001000010001000000
00010001010000010000
01110010101000010100
00000000000001100000
00001000010000000100
00101000000010110011
01010100000000000000
00001101010010010010
10000000000000010100
01011010000001100100
00000000000000010011
00001100111000110100
00000010000000000000
00010000100111000000
11010000001001010100
01100010011001011001
00000101000010101010
00100000000000000001
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  import copy
  def evaluation(L,si,sj):
    score=0
    prob=[[0]*20 for _ in range(20)]
    prob[si][sj]=1
    for k in range(len(L)):
      tmp=[[0]*20 for _ in range(20)]
      for i in range(20):
        for j in range(20):
          if L[k]=='U' and i>0 and V[i-1][j]=='0': tmp[i-1][j]+=(1-p)*prob[i][j]
          elif L[k]=='D' and i<19 and V[i][j]=='0': tmp[i+1][j]+=(1-p)*prob[i][j]
          elif L[k]=='R' and j<19 and H[i][j]=='0': tmp[i][j+1]+=(1-p)*prob[i][j]
          elif  L[k]=='L' and j>0 and H[i][j-1]=='0': tmp[i][j-1]+=(1-p)*prob[i][j]
          else: tmp[i][j]+=(1-p)*prob[i][j]
          tmp[i][j]+=p*prob[i][j]
      score+=250000*(400-k)*tmp[ti][tj]
      tmp[ti][tj]=0
      prob=copy.deepcopy(tmp)
      if k==1:
        print(prob)
    return score

  from heapq import heappop,heappush
  def Dijkstra(G,s):
    done=[False]*len(G)
    inf=500
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

  inp=input().split()
  si,sj,ti,tj=map(int,inp[:4])
  p=float(inp[4])
  H=[input() for _ in range(20)]
  V=[input() for _ in range(19)]
  G=[[] for _ in range(20**2)] #Dijkstra用
  G2=[[] for _ in range(20**2)] #Dijkstra用逆グラフ
  G3=[[] for _ in range(20**2)] #BFS用

  #Dijkstra用グラフの構成
  for i in range(20):
    s,t=0,0
    while s<20:
      while t<19 and H[i][t]=='0': t+=1
      if s<t:
        for j in range(s,t): G[i*20+j].append((abs(t-j),i*20+t)); G2[i*20+t].append((abs(t-j),i*20+j))
      s,t=t+1,t+1
  for i in range(20):
    s,t=19,19
    while s>=0:
      while t>0 and H[i][t-1]=='0': t-=1
      if s>t:
        for j in range(t+1,s+1): G[i*20+j].append((abs(t-j),i*20+t)); G2[i*20+t].append((abs(t-j),i*20+j))
      s,t=t-1,t-1
  for j in range(20):
    s,t=0,0
    while s<20:
      while t<19 and V[t][j]=='0': t+=1
      if s<t:
        for i in range(s,t): G[i*20+j].append((abs(t-i),t*20+j)); G2[t*20+j].append((abs(t-i),i*20+j))
      s,t=t+1,t+1
  for j in range(20):
    s,t=19,19
    while s>=0:
      while t>0 and V[t-1][j]=='0': t-=1
      if s>t:
        for i in range(t+1,s+1): G[i*20+j].append((abs(t-i),t*20+j)); G2[t*20+j].append((abs(t-i),i*20+j))
      s,t=t-1,t-1
  t=tj
  while t<19 and H[ti][t]=='0': t+=1
  if t>tj:
    for j in range(tj+1,t+1): G[ti*20+j].append((abs(tj-j),ti*20+tj)); G2[ti*20+tj].append((abs(tj-j),ti*20+j))
  t=tj
  while t>0 and H[ti][t-1]=='0': t-=1
  if t<tj:
    for j in range(t,tj): G[ti*20+j].append((abs(tj-j),ti*20+tj)); G2[ti*20+tj].append((abs(tj-j),ti*20+j))
  t=ti
  while t<19 and V[t][tj]=='0': t+=1
  if t>ti:
    for i in range(ti+1,t+1): G[i*20+tj].append((abs(ti-i),ti*20+tj)); G2[ti*20+tj].append((abs(ti-i),i*20+tj))
  t=ti
  while t>0 and V[t-1][tj]=='0': t-=1
  if t<ti:
    for i in range(t,ti): G[i*20+tj].append((abs(ti-i),ti*20+tj)); G2[ti*20+tj].append((abs(ti-i),i*20+tj))

  #BFS用グラフの構成
  for i in range(20):
    for j in range(20):
      if i>0 and V[i-1][j]=='0': G3[i*20+j].append((i-1)*20+j)
      if i<19 and V[i][j]=='0': G3[i*20+j].append((i+1)*20+j)
      if j>0 and H[i][j-1]=='0': G3[i*20+j].append(i*20+j-1)
      if j<19 and H[i][j]=='0': G3[i*20+j].append(i*20+j+1)

  D=Dijkstra(G,si*20+sj)
  B=bfs(G3,ti*20+tj)
  ans=[]

  #Dijkstraで進めるところまで
  l=0
  d=500
  now=si*20+sj
  for i in range(20):
    for j in range(20):
      if B[i*20+j]<d and D[i*20+j]<500:
        now=i*20+j
        d=B[i*20+j]
  now2=now
  while now!=si*20+sj:
    for c,nxt in G2[now]:
      if D[nxt]==D[now]-c:
        i,j=now//20,now%20
        ii,jj=nxt//20,nxt%20
        if i>ii: ans.append('D'*(i-ii)); l+=i-ii
        elif i<ii: ans.append('U'*(ii-i)); l+=ii-i
        elif j>jj: ans.append('R'*(j-jj)); l+=j-jj
        else: ans.append('L'*(jj-j)); l+=jj-j
        now=nxt
        break
  ans=ans[::-1]

  #BFSでにじり寄る部分
  while now2!=ti*20+tj:
    for nxt2 in G3[now2]:
      if B[nxt2]<B[now2]:
        i,j=now2//20,now2%20
        ii,jj=nxt2//20,nxt2%20
        if i>ii: ans.append('U'); l+=1
        elif i<ii: ans.append('D'); l+=1
        elif j>jj: ans.append('L'); l+=1
        else: ans.append('R'); l+=1
        now2=nxt2

  r=min(200//l,3)
  ans=''.join([ans[i//r] for i in range(len(ans)*r)])+'RDUL'*((200-r*l)//4)
  print(ans,evaluation(ans,si,sj))