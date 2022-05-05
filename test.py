import io
import sys

_INPUT = """\
6
8
-690377870 113265092 309910313
-237317502 226530184 111095495
-428244446 20449344 86035324
-39706910 368088192 781991
-719554585 184918601 393468779
-164798782 369837202 30446065
-676521103 63096981 401281747
-171745255 252387924 62120322
-521613854 56777512 428978993
-180948782 170332536 141234994
-831388448 111561549 52842458
172665493 334684647 8107233
-521749554 82172757 18542931
-110885769 328691028 4319665
-852824253 182038292 292554027
57367207 364076584 139043300
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
  import math
  mod=10**9+7
  t=int(input())
  for _ in range(t):
    b,q,y=map(int,input().split())
    c,r,z=map(int,input().split())
    if r%q!=0 or c<b or (c-b)%q!=0 or c+(z-1)*r>b+(y-1)*q: print(0)
    elif c-r<b or c+z*r>b+(y-1)*q: print(-1)
    else:
      ans=0
      i=1
      while i*i<=r:
        if r%i==0:
          if i*q//math.gcd(i,q)==r: ans=(ans+(r//i)**2)%mod
          if i**2!=r and r//i*q//math.gcd(r//i,q)==r: ans=(ans+i**2)%mod
        i+=1
      print(ans)