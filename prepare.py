#以下に設定するABCの回数を指定する
times = "235"

import os
import shutil
import textfile
 
dir_path = 'C:/Users/katonyonko/OneDrive/デスクトップ/ABC{}'.format(times)
os.mkdir(dir_path)
problems = ["a", "b", "c", "d"]
if int(times) > 125:
  problems += ["e", "f"]
if int(times) > 211:
  problems += ["g", "h"]
for i in range(len(problems)):
  shutil.copy('./code.py', dir_path+'/ABC'+times+'_'+problems[i].upper()+'.py')
  textfile.replace(dir_path+'/ABC'+times+'_'+problems[i].upper()+'.py', 'times=""', 'times="{}"'.format(times))
  textfile.replace(dir_path+'/ABC'+times+'_'+problems[i].upper()+'.py', 'problem=""', 'problem="{}"'.format(problems[i]))