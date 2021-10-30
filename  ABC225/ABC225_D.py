import io
import sys

_INPUT = """\
6
7 14
1 6 3
1 4 1
1 5 2
1 2 7
1 3 5
3 2
3 4
3 6
2 3 5
2 4 1
1 1 5
3 2
3 4
3 6
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
    N,Q=map(int,input().split())
    prev=[-1]*N
    post=[-1]*N
    for _ in range(Q):
        query=input()
        if query[0]=='1':
            d,x,y=map(int,query.split())
            x-=1; y-=1
            prev[y]=x
            post[x]=y
        elif query[0]=='2':
            d,x,y=map(int,query.split())
            x-=1; y-=1
            post[x]=-1
            prev[y]=-1
        else:
            # print(prev,post)
            d,x=map(int,query.split())
            x-=1
            cur=x
            ans=[]
            while prev[cur]!=-1:
                cur=prev[cur]
            # print(cur)
            while post[cur]!=-1:
                ans.append(cur+1)
                cur=post[cur]
            ans.append(cur+1)
            print(len(ans),*ans)