import io
import sys

_INPUT = """\
6
aba
ccc
xyz
"""

sys.stdin = io.StringIO(_INPUT)
case_no=int(input())
for __ in range(case_no):
    S=len(set(list(input())))
    if S==1: print(1)
    elif S==2: print(3)
    else: print(6)