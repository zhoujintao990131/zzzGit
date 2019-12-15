import argparse
import sys
import time
import cv2
import face
import numpy as np
import pandas as pd

text_n=4
base_path='gate/'
video_path=base_path+'door'+str(text_n)+'.mp4'#视频地址，如果不输的话也可以改成摄像头
video_capture = cv2.VideoCapture(video_path)#读取视频

TOTAL_FRAME = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) #获取视频总帧数
FPS = video_capture.get(cv2.CAP_PROP_FPS) #获取帧率单位帧每秒
size = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) #获取视频长宽尺寸

intervalT=0.5#取样采集脸间隔T
intervalF=round(intervalT*FPS)#隔多少帧取个脸
Ts=round(1000/FPS)#一帧对应多少毫秒
font = cv2.FONT_HERSHEY_COMPLEX#字体
attention=['Sampling....','Warning:ILLEGAL BREAK-IN','Warning:JUST ONE PLEASE','PLEASE IN']#sampling表示在采集人脸数据、Illegal表示非法闯入、Order

face_recognition = face.Recognition()

# debug='store_true'
debug=False
if debug:
    print("Debug enabled")
    face.debug = True

ROI=[int(size[0]/2-270),int(size[1]/2-350),int(size[0]/2+120),int(size[1]/2)-50]
def frame2time(x):#把帧改成时间的形式
    t0=0
    f0=0#可以手动定义t0和f0
    deltaF=x-f0
    deltaT=deltaF*Ts
    t=t0+deltaT
    ms=t%1000
    t=int((t-ms)/1000)
    s=t%60
    t=int((t-s)/60)
    m=t%60
    h=int((t-m)/60)
    y=str(h).zfill(2)+':'+str(m).zfill(2)+':'+str(s).zfill(2)+':'+str(ms).zfill(3)
    return y

def add_overlays(img,clock,FPS,attention,ROI,faces,roi_img,liblist):
    cv2.putText(img,clock, (15,30), font, 1, (120,250,10),2)
    cv2.putText(img,'FPS  '+str(round(FPS,3)),(15,60),font,1,(125,255,0),2)
    cv2.putText(img,attention,(15,int(size[1])-30),font,1,(100,50,200),3)
    cv2.rectangle(img, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 0),3)
    cnt=30
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(roi_img,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (20, 40, 165), 2)
            if face.name is not None:
                cv2.putText(roi_img, face.name, (face_bb[0]-15, face_bb[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (12, 20, 180),
                            thickness=2, lineType=2)
    for name in liblist:
        cnt=cnt+30
        cv2.putText(img,name, (int(size[0]-180),cnt), font, 1, (120,50,200),2)
start_time=time.time()
fps=FPS*2/3
fps_display_interval=2.5
frame0=0
state=0

current_num=0
gateT1=6*1000#上限阈值，8秒跑出去
gateT2=3*1000#下线阈值，给3秒走出框
time_list=[]#记录入馆时间
liblist=[]#记录实时在馆人数
state_list=[]
data=pd.read_csv('library.csv')
name=data['name'].values
name_list=name.tolist()#学生名字花名册
l=len(name_list)
for i in range(l):
    state_list.append(0)
gateo=40
cnt_out=gateo

for frame in range(TOTAL_FRAME):
    flag,img=video_capture.read()
    TT=frame/FPS*1000#现在的时间（按照帧算的）
    for i in range(len(time_list)):
        if time_list[i]+gateT1<=TT:
            x=time_list.pop(i)
            tmp=liblist.pop(i)#出馆的人
            cnt_out=0
    cnt_out=cnt_out+1
    if cnt_out<=gateo:
        cv2.putText(img,tmp+' is OUT!!',(15,int(size[1])-70),font,1,(100,50,200),3)
    if flag:
        # img=cv2.transpose(img)
        roi_img=img[ROI[1]:ROI[3]+1,ROI[0]:ROI[2]+1]
        clock=frame2time(frame)
        if frame % intervalF==0:
            faces=face_recognition.identify(roi_img)
            if len(faces)==0:
                state=0
            elif len(faces)>1:
                state=2
            else:#仅有一人的情况最重要
                people=faces[0].name
                if people in liblist:
                    j=liblist.index(people)
                    if time_list[j]+gateT2<TT:
                        time_list[j]=TT#对于想蒙混过关重复进入的更新最新闯关时间
                        state=1
                else:#唯一可以入内的
                    j=name_list.index(people)
                    state_list[j]=state_list[j]+1
                    if state_list[j]==2:#连续四次合法采集到
                        liblist.append(people)
                        time_list.append(TT)
                        state=3
                        state_list[j]=0
        end_time=time.time()
        delta_T=end_time-start_time
        if delta_T>fps_display_interval:
            fps=(frame-frame0)/delta_T
            frame0=frame
            start_time=time.time()
        add_overlays(img,clock,fps,attention[state],ROI,faces,roi_img,liblist)
        cv2.imshow('video',img)
        cv2.imshow('ROI',roi_img)
        C=cv2.waitKey(Ts)
    if C & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

