import io
import sys

_INPUT = """\
6
3
1 1
2 1
1 2
10
414598724 87552841
252911401 309688555
623249116 421714323
605059493 227199170
410455266 373748111
861647548 916369023
527772558 682124751
356101507 249887028
292258775 110762985
850583108 796044319
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
    import math
    class Item:
        def  __init__(self,x,y):
            self.x = x
            self.y = y
        def __lt__(self, other):
            if not isinstance(other, Item):
                return NotImplemented
            return self.x*other.y<self.y*other.x
        def __eq__(self, other):
            if not isinstance(other, Item):
                return NotImplemented
            return self.x*other.y==self.y*other.x 
    inf=10**30
    N=int(input())
    ans=0
    task=[]
    for i in range(N):
        x,y=map(int,input().split())
        if y>1:
            if x>1:
                z=math.gcd(x-1,y)
            else: z=1
            z2=math.gcd(x,y-1)
            task.append([Item((x-1)//z,y//z),Item(x//z2,(y-1)//z2)])
        else: task.append([Item(x-1,1),Item(inf,1)])
    task.sort(key=lambda x:x[1])
    cur=Item(0,inf)
    # for i in range(N): print(i,task[i][0].x,task[i][0].y)
    for i in range(N):
        if cur<task[i][0] or task[i][0]==cur:
            ans+=1
            # print(i)
            cur=task[i][1]
    print(ans)