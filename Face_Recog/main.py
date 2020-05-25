import sys
import os
import glob
import numpy as np 
from Deep_Learning_Face import *

video_list = glob.glob(r'F:\Face_Recog\videos\*.mp4')
video_list.sort()
faces_count = []
for in_file in (video_list):
    out_arr = deep_face_count(in_file)
    f_c = np.max(out_arr)
    if f_c==1:
        f_c==np.sum(out_arr)
    faces_count=np.append(faces_count,f_c)
dir = 'F:\Face_Recog'
out_fp = os.path.join(dir,'People_Count.csv')
np.savetxt(out_fp,faces_count)
