import os
import collections
import random

regular_txt_path='/data/user-tjlzwx/Code/mmdetection/data/nwafu_sheep_face/one.txt'
hard_txt_path='/data/user-tjlzwx/Code/mmdetection/data/nwafu_sheep_face/two.txt'
val_txt_path='/data/user-tjlzwx/Code/mmdetection/data/nwafu_sheep_face/val.txt'
train_txt_path='/data/user-tjlzwx/Code/mmdetection/data/nwafu_sheep_face/train.txt'

val=[]



f=open(regular_txt_path,mode='r')
content=f.readlines()
index=[0]*(len(content))
for i in range(708):
    idx=random.randrange(0,len(content),1)
    while index[idx]!=0:
        idx=(idx+1)%(len(content))
    try:
        val.append(int(content[idx][:-1]))
        index[idx]=1
    except IndexError :
        print(f'IndexError {idx}')
f.close()

f=open(hard_txt_path,mode='r')
content=f.readlines()
index=[0]*(len(content))
for i in range(367):
    idx=random.randrange(0,len(content),1)
    while index[idx]!=0:
        idx=(idx+1)%(len(content))
    try:
        val.append(int(content[idx][:-1]))
        index[idx]=1
    except IndexError:
        print(f'IndexError {idx}')
f.close()


print(val)

#打开val
# f=open(val_txt_path,mode='r')
# content=f.readlines()
# val=[]
# for i in content:
#     val.append(int(i[:-1]))
# f.close()

f=open(val_txt_path,mode='w')
# content=f.readlines()
for idx in val:
    f.write(f'{idx}\n')
f.close()

#判断重复
c = dict(collections.Counter(val))
s = [k for k,v in c.items() if v > 1]
print(f'repeated id: {s}')
# for idx in range(1,601):
#     f.write(f'{idx}\n')
# f.close()

f=open(train_txt_path,mode='w')
for idx in range(1,5373):
    if idx in val:
        pass
    else:
        f.write(f'{idx}\n')
        # print(f' val {idx} write successfully')
f.close()


