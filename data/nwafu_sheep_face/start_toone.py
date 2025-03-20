import os
import collections
import random

regular_txt_path='/data/user-tjlzwx/Code/mmdetection/data/nwafu_sheep_face/one.txt'
hard_txt_path='/data/user-tjlzwx/Code/mmdetection/data/nwafu_sheep_face/two.txt'
val_txt_path='/data/user-tjlzwx/Code/mmdetection/data/nwafu_sheep_face/val.txt'
train_txt_path='/data/user-tjlzwx/Code/mmdetection/data/nwafu_sheep_face/train.txt'

val=[]




f=open(hard_txt_path,mode='r')
content=f.readlines()
index=[0]*5373
index[0]=1

for idx in content:
    id=int(idx[:-1])
    index[id]=1
f.close()


f=open(regular_txt_path,mode='w')
for idx in range(1,5373):
    if index[idx] ==0:
        f.write(f'{idx}\n')

f.close()

