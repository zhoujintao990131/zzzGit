from readData import *
import GCLM as WL
import math
import cv2
train_voxel=[]
train_seg=[]
for data in train_data:
    seg=data['seg']
    # voxel=data['voxel']/255
    voxel=data['voxel']
    voxel=np.multiply(seg,voxel)
    train_voxel.append(voxel)
    train_seg.append(seg)

test_voxel=[]
test_seg=[]
for data in test_data:
    seg=data['seg']
    voxel=data['voxel']
    voxel=np.multiply(seg,voxel)
    test_voxel.append(voxel)
    test_seg.append(seg)

print('antiplication finished')

def grayFeature(img):#灰度特征
    tmp=np.nonzero(img)
    im=img[tmp]
    total=im.size
    sum=img.sum()
    mean=np.mean(im)#灰度均值
    var=np.var(im)#灰度方差
    maxP=im.max()#最大灰度值
    minP=im.min()#最小灰度值
    sum=0
    energy=0#能量
    skewness=0#偏离度
    kurtosis=0#峰度
    h=[0]*256
    for k in range(1,256):
        # tmp=np.sum(img==k/255)
        tmp=np.sum(img==k)
        h[k]=tmp/total#灰度概率密度函数
        energy=energy+h[k]**2
        # skewness=skewness+((k/255-mean)**3*h[k])/(var**(3/2))
        # kurtosis=kurtosis+((k/255-mean)**4*h[k])/(var**2)
        skewness=skewness+((k-mean)**3*h[k])/(var**(3/2))
        kurtosis=kurtosis+((k-mean)**4*h[k])/(var**2)
        if h[k]!=0:
            sum=sum+math.log(h[k],2)*h[k]
    entropy=-sum#熵
    T=180#钙化阈值
    # gateT=T/255
    gateT=T
    Ca=np.sum(img>gateT)/total#钙化度
    feature_gray=[mean,var,maxP,minP,energy,skewness,kurtosis,entropy,Ca]#9维灰度特征分别是——均值、方差、最大、最小、能量、偏离度、峰度、熵、钙化度
    return(feature_gray)


def wlFeature(img):#纹理特征
    energy_z=0#能量
    entropy_z=0#熵
    SP_z=0#均匀性
    SI_z=0#惯量
    for z in range(100):
        tmp=img[:,:,z]
        S=WL.getGlcm(tmp,0,1)#横向方向
        S=np.mat(S)
        energy_z=energy_z+np.square(S).sum()
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i,j]!=0:
                    entropy_z=entropy_z+S[i,j]*math.log(S[i,j],2)
                    SP_z=SP_z+1/(1+(i-j)**2)*S[i,j]
                    SI_z=SI_z+(i-j)**2*S[i,j]
    energy_z=energy_z/100
    entropy_z=entropy_z/100
    SP_z=SP_z/100
    SI_z=SI_z/100

    energy_y=0#能量
    entropy_y=0#熵
    SP_y=0#均匀性
    SI_y=0#惯量
    for y in range(100):
        tmp=img[:,y,:]
        S=WL.getGlcm(tmp,0,1)#横向方向
        S=np.mat(S)
        energy_y=energy_y+np.square(S).sum()
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i,j]!=0:
                    entropy_y=entropy_y+S[i,j]*math.log(S[i,j],2)
                    SP_y=SP_y+1/(1+(i-j)**2)*S[i,j]
                    SI_y=SI_y+(i-j)**2*S[i,j]
    energy_y=energy_y/100
    entropy_y=entropy_y/100
    SP_y=SP_y/100
    SI_y=SI_y/100

    energy_x=0#能量
    entropy_x=0#熵
    SP_x=0#均匀性
    SI_x=0#惯量
    for x in range(100):
        tmp=img[x,:,:]
        S=WL.getGlcm(tmp,0,1)#横向方向
        S=np.mat(S)
        energy_x=energy_x+np.square(S).sum()
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i,j]!=0:
                    entropy_x=entropy_x+S[i,j]*math.log(S[i,j],2)
                    SP_x=SP_x+1/(1+(i-j)**2)*S[i,j]
                    SI_x=SI_x+(i-j)**2*S[i,j]
    energy_x=energy_x/100
    entropy_x=entropy_x/100
    SP_x=SP_x/100
    SI_x=SI_x/100

    feature_wl=[entropy_x,energy_y,energy_z,entropy_x,entropy_y,entropy_z,SP_x,SP_y,SP_z,SI_x,SI_y,SI_z]#12维纹理特征分别是——能量、熵、均匀性、惯量
    return(feature_wl)

p=1#层厚1mm
q=1#每像素代表实际大小
n=p/q
mi=1#假设肺结节质量均匀，每个点质量为1

def edgedetect(x,y,z,mask):
    m=mask[x-1:x+2,y-1:y+2,z-1:z+2]
    s=m.sum()
    if s<27:
        flag=True
    else: 
        flag=False
    return(flag)

def shapeFeature(mask,img):#形态特征
    area=0
    tmp=np.nonzero(mask)
    volume=mask[tmp].size
    centerx=tmp[0].sum()/volume
    centery=tmp[1].sum()/volume
    centerz=tmp[2].sum()/volume
    center=np.mat([centerx,centery,centerz])#质心
    for i in range(len(tmp[0])):
        if edgedetect(tmp[0][i],tmp[1][i],tmp[2][i],mask):
            area=area+1#注意这个表面积更倾向于人为定义的，不一定有参考依据
    area=area*1.414#由于表面积计算并不严谨给一个补偿系数
    ball=(4*math.pi*(3/4*volume/math.pi)**(2/3))/area
    zjt,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    im1=img[:,:,int(centerz)]
    im2=img[:,int(centery),:]
    im3=img[int(centerx),:,:]
    m=cv2.moments(im1)
    h1=cv2.HuMoments(m)
    m=cv2.moments(im2)
    h2=cv2.HuMoments(m)
    m=cv2.moments(im3)
    h3=cv2.HuMoments(m)
    feature_shape=[volume,area,ball,h1[0][0],h1[1][0],h1[2][0],h1[3][0],h1[4][0],h1[5][0],h1[6][0],h2[0][0],h2[1][0],h2[2][0],h2[3][0],h2[4][0],h2[5][0],h2[6][0],h3[0][0],h3[1][0],h3[2][0],h3[3][0],h3[4][0],h3[5][0],h3[6][0]]
    return feature_shape

feature_train_list=[]
for i in range(len(train_voxel)):
    feature1=grayFeature(train_voxel[i])
    feature2=wlFeature(train_voxel[i])
    feature3=shapeFeature(train_seg[i],train_voxel[i])
    print(i,'in train_data')
    feature_train_list.append(feature1+feature2+feature3)
feature_test_list=[]
for i in range(len(test_voxel)):
    feature1=grayFeature(test_voxel[i])
    feature2=wlFeature(test_voxel[i])
    feature3=shapeFeature(test_seg[i],test_voxel[i])
    print(i,'in test_data')
    feature_test_list.append(feature1+feature2+feature3)

print('prepared for feature')
df_test = pd.DataFrame(feature_test_list, columns=['gray_mean','gray_var','gray_maxP','gray_minP','gray_energy','gray_skewness','gray_kurtosis','gray_entropy','gray_Ca','wl_energy_x','wl_energy_y','wl_energy_z','wl_entropy_x','wl_entropy_y','wl_entropy_z','wl_SP_x','wl_SP_y','wl_SP_z','wl_SI_x','wl_SI_y','wl_SI_z','shape_volume','shape_area','shape_ball','shape_h10','shape_h11','shape_h12','shape_h13','shape_h14','shape_h15','shape_h16','shape_h20','shape_h21','shape_h22','shape_h23','shape_h24','shape_h25','shape_h26','shape_h30','shape_h31','shape_h32','shape_h33','shape_h34','shape_h35','shape_h36'])
df_test.to_csv('feature_test.csv',index=0)

df_train = pd.DataFrame(feature_train_list, columns=['gray_mean','gray_var','gray_maxP','gray_minP','gray_energy','gray_skewness','gray_kurtosis','gray_entropy','gray_Ca','wl_energy_x','wl_energy_y','wl_energy_z','wl_entropy_x','wl_entropy_y','wl_entropy_z','wl_SP_x','wl_SP_y','wl_SP_z','wl_SI_x','wl_SI_y','wl_SI_z','shape_volume','shape_area','shape_ball','shape_h10','shape_h11','shape_h12','shape_h13','shape_h14','shape_h15','shape_h16','shape_h20','shape_h21','shape_h22','shape_h23','shape_h24','shape_h25','shape_h26','shape_h30','shape_h31','shape_h32','shape_h33','shape_h34','shape_h35','shape_h36'])
df_train.to_csv('feature_train.csv',index=0)
print('saved to feature.csv')




# plt.figure(1)
# plt.subplot(1,2,1)
# plt.imshow(train_seg[175][:,50,:])
# plt.subplot(1,2,2)
# plt.imshow(train_voxel[175][:,50,:])
# plt.show()
