#以下に設定するABCの回数を指定する
<<<<<<< HEAD
times = "076"
=======
<<<<<<< HEAD
times = "243"
=======
times = "078"
>>>>>>> 553b59ae5d316d0b5c3047c1a9ed90ac32cddc42
>>>>>>> 463640a80645a6217ceef2a21b5ab858ded55f10

import os
import shutil
import textfile
 
<<<<<<< HEAD
real_time='Yes'
dir_path = 'C:/Users/katonyonko/OneDrive/デスクトップ/ABC{}'.format(times)
=======
dir_path = 'C:\\Users\\shinykato\\ABC{}'.format(times)
>>>>>>> 553b59ae5d316d0b5c3047c1a9ed90ac32cddc42
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