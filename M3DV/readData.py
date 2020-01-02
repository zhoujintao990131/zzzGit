import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
data_train_path='data/train_val/'
data_test_path='data/test/'
tag_train_path='data/train_val.csv'
tag_test_path='data/sampleSubmission.csv'

tmp=pd.read_csv(tag_train_path)
train_tag=tmp['lable']
train_tag=train_tag.tolist()#记录训练数据对应的标签
train_name=tmp['name']
train_name=train_name.tolist()

tmp=pd.read_csv(tag_test_path)
test_name=tmp['name']
test_name=test_name.tolist()

train_filelist=os.listdir(data_train_path)
train_data=[]#记录几百组训练数据
# train_data_voxel=[]#记录结节
# train_data_seg=[]#记录mask
noftrain=len(train_filelist)
for item in train_name:
    tmp=np.load(data_train_path+item+'.npz')
    train_data.append(tmp)
    # train_data_voxel.append(tmp['voxel']/255)
    # train_data_seg.append(tmp['seg'])

test_filelist=os.listdir(data_test_path)
test_data=[]#记录几百组的测试数据
# test_data_voxel=[]#记录结节
# test_data_seg=[]#记录mask
for item in test_name:
    tmp=np.load(data_test_path+item+'.npz')
    test_data.append(tmp)
    # test_data_voxel.append(tmp['voxel']/255)
    # test_data_seg.append(tmp['seg'])
print('load finished')
