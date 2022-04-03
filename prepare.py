#以下に設定するABCの回数を指定する
times = "246"

import os
import shutil
import textfile
 
real_time='Yes'
dir_path = 'C:/Users/katonyonko/OneDrive/デスクトップ/ABC{}'.format(times)
os.mkdir(dir_path)
problems = ["a", "b", "c", "d"]
if int(times) > 125:
  problems += ["e", "f"]
if int(times) > 211:
  problems += ["g", "h"]
if real_time=='Yes':
  for i in range(len(problems)):
      shutil.copy('./code_realtime.py', dir_path+'/ABC'+times+'_'+problems[i].upper()+'.py')
else:
  for i in range(len(problems)):
      shutil.copy('./code.py', dir_path+'/ABC'+times+'_'+problems[i].upper()+'.py')
      textfile.replace(dir_path+'/ABC'+times+'_'+problems[i].upper()+'.py', 'times=""', 'times="{}"'.format(times))
      textfile.replace(dir_path+'/ABC'+times+'_'+problems[i].upper()+'.py', 'problem=""', 'problem="{}"'.format(problems[i]))