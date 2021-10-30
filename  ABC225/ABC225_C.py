import io
import sys

_INPUT = """\
6
2 3
1 2 3
8 9 10
2 1
1
2
10 4
1346 1347 1348 1349
1353 1354 1355 1356
1360 1361 1362 1363
1367 1368 1369 1370
1374 1375 1376 1377
1381 1382 1383 1384
1388 1389 1390 1391
1395 1396 1397 1398
1402 1403 1404 1405
1409 1410 1411 1412
2 1
2
9
1 7
7 8 9 10 11 12 13
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
    N,M=map(int,input().split())
    ans='Yes'
    B=[list(map(int,input().split())) for _ in range(N)]
    for i in range(M-1):
        if B[0][i]%7==0: ans='No'
    tmp=B[0][0]-7
    for i in range(N):
        if B[i][0]!=tmp+7: ans='No'
        tmp=B[i][0]
        for j in range(M-1):
            if B[i][j+1]!=B[i][j]+1: ans='No'
    print(ans)