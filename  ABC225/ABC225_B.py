import io
import sys

_INPUT = """\
6
5
1 4
2 4
3 4
4 5
4
2 4
1 4
2 3
10
9 10
3 10
4 10
8 10
1 10
2 10
7 10
6 10
5 10
3
1 2
2 3
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
    N=int(input())
    G=[[] for _ in range(N)]
    for i in range(N-1):
        a,b=map(int,input().split())
        a-=1; b-=1
        G[a].append(b)
        G[b].append(a)
    ans='No'
    for i in range(N):
        if len(G[i])==N-1: ans='Yes'
    print(ans)