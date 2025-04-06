import io
import sys
from collections import defaultdict, deque, Counter
from itertools import permutations, combinations, accumulate
from heapq import heappush, heappop
from bisect import bisect_right, bisect_left
from math import gcd
import math

_INPUT = """\
"""

def input():
  return sys.stdin.readline()[:-1]

def solve():
  pass

def main():
  if sys.stdin.isatty():
    sys.stdin = io.StringIO(_INPUT)
    while True:
      solve()
  else:
    solve()

main()