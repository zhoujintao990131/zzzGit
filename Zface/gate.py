from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import pandas as pd
import os
from scipy import misc
import sys
import tensorflow as tf
import facenet
import align.detect_face
import random
from time import sleep
from sklearn.svm import SVC
import math
import pickle
import shutil
from videoMaker import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def rgb2bgr(img):#将RGB转为BGR
    img=img[:,:,::-1]
    return img
def mtcnn(input_dir,output_dir,image_size,margin,random_order,gpu_memory_fraction,detect_multiple_faces):
    face_list=[]
    sleep(random.random())
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(input_dir)
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                # print(image_path,'hahah')
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                        origin_img=img
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces>0:
                            det = bounding_boxes[:,0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces>1:
                                if detect_multiple_faces:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                    det_arr.append(det[index,:])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0]-margin/2, 0)
                                bb[1] = np.maximum(det[1]-margin/2, 0)
                                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                                cv2.rectangle(origin_img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0),3)
                                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                                face_list.append(scaled)
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                misc.imsave(output_filename_n, scaled)
                                
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))                      
    return nrof_successfully_aligned,face_list,origin_img#返回三个值，一个是所含人脸的个数，第二个是人脸列表，第三个是带bbox的原图


output_dir='~/文档/zjt/facenet/Zface/gateout'
image_size=160
margin=32
random_order='store_true'
gpu_memory_fraction=0.25
detect_multiple_faces=True
# num,face_list,img=mtcnn(input_dir,output_dir,image_size,margin,random_order,gpu_memory_fraction,detect_multiple_faces)

text_n=1
base_path='gate/'
video_path=base_path+'door'+str(text_n)+'.mp4'
video_capture = cv2.VideoCapture(video_path)
TOTAL_FRAME = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) #获取视频总帧数
FPS = video_capture.get(cv2.CAP_PROP_FPS) #获取帧率单位帧每秒
size = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) #获取视频长宽
intervalT=2#取样采集脸间隔T
intervalF=round(intervalT*FPS)
Ts=round(1000/FPS)#每多少毫秒采一帧
font = cv2.FONT_HERSHEY_COMPLEX
windowname='zjt'

def frame2time(x):
    t0=0
    f0=0
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

for i in range(TOTAL_FRAME):
    flag,photo=video_capture.read()
    if flag:
        clock=frame2time(i)
        photo=cv2.transpose(photo)
        
        cv2.putText(photo, clock, (15,30), font, 1, (255,0,0),2)
        if i%intervalF==0:
            path0=base_path+clock
            path=path0+'/'+'photo'
            os.makedirs(path)
            cv2.imwrite(path+'/'+clock+'.jpg',photo)
            num,face_list,photo=mtcnn(path0,output_dir,image_size,margin,random_order,gpu_memory_fraction,detect_multiple_faces)
            photo=rgb2bgr(photo)
            shutil.rmtree(path0)  
        cv2.imshow(windowname,photo)
        cv2.imwrite('gateout/img/'+str(i)+'.jpg',photo)
        c=cv2.waitKey(Ts)
        if c & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()
# build_video(FPS,(1280,720),'gateout/img/','zjt.avi')
