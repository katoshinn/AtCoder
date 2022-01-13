from urllib.request import urlopen
from bs4 import BeautifulSoup
import io
import sys

#ABCの回数
times=234
#問題
problem="a"

 # 1. Get a html.
with urlopen("https://atcoder.jp/contests/abc{0}/tasks/abc{0}_{1}".format(times, problem)) as res:
  html = res.read().decode("utf-8")
# 2. Load a html by BeautifulSoup.
soup = BeautifulSoup(html, "html.parser")
# 3. Get items you want.
titles = soup.select(".lang-ja pre")
titles =[t.text for t in titles[1:]]

for __ in range(0,len(titles),2):
  sys.stdin = io.StringIO(titles[__])

  """ここから下にコードを記述"""
  
  """ここから上にコードを記述"""

  print(titles[__+1])