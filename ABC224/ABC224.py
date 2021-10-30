import io
import sys

_INPUT = """\
6
1234
1
31415926535897932384626433832795
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
    mod=998244353
    S=input()
    ans=0
    if len(S)>1:
        t=pow(2,len(S)-2,mod)
    f=[pow(5,len(S)-1-i,mod) for i in range(len(S))]
    tt=[pow(10,len(S)-1-i,mod) for i in range(len(S))]
    ttt=[pow(2,i,mod) for i in range(len(S))]
    y=pow(4,mod-2,mod)
    for i in range(len(S)):
        ans=(ans+int(S[i])*tt[i]*ttt[i])%mod
        if len(S)>1:
            ans=(ans+int(S[i])*t*(f[i]-1)*y)%mod
    print(ans)