from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
from six.moves import xrange
from time import sleep
import random
import cv2
import pandas as pd
input_dir='~/文档/zjt/facenet/Zface/hhh'
output_dir='~/文档/zjt/facenet/Zface/ooo'
image_size=160
margin=32
random_order='store_true'
gpu_memory_fraction=0.25
detect_multiple_faces=True


sleep(random.random())
output_dir = os.path.expanduser(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Store some git revision info in a text file in the log directory
src_path,_ = os.path.split(os.path.realpath(__file__))
facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
dataset = facenet.get_dataset(input_dir)

print('Creating networks and loading parameters')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.3 ]  # three steps's threshold
factor = 0.709 # scale factor

# Add a random key to the filename to allow alignment using multiple processes
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
Zlist = [] #记录提取的人脸图片的列表
Wlist=[]#记录boundingbox的列表
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
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                            nrof_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            misc.imsave(output_filename_n, scaled)
                            # 发现接口
                            Zlist.append(scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                            cv2.rectangle(origin_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0),1)
                            Wlist.append(bb)
                    else:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        
print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

# python3 contributed/predict.py zdata/newtext/test/zjt02.png src/models/20180402-114759 src/models/zclassify.pkl
# image_files=list(['new/liu_chang/liu_chang005.png','new/zhou_jintao/zhou_jintao124.png'])
model='~/文档/zjt/facenet/src/models/20180402-114759'
classifier_filename='~/文档/zjt/facenet/Zface/Zclassify.pkl'
image_size=160
seed=666
margin=44
gpu_memory_fraction=25
def load_and_align_data(image_list, image_size, margin, gpu_memory_fraction):
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_list)
    img_list=[]
    count_per_image = []
    for i in xrange(nrof_samples):
        img = image_list[i]
        # 在这里发现接口
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        count_per_image.append(len(bounding_boxes))
        for j in range(len(bounding_boxes)):	
                det = np.squeeze(bounding_boxes[j,0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                img_list.append(prewhitened)		
    images = np.stack(img_list)
    return images, count_per_image, nrof_samples

attend_list=[]
images, cout_per_image, nrof_samples = load_and_align_data(Zlist,image_size, margin, gpu_memory_fraction)
with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load the model
            facenet.load_model(model)
        # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images , phase_train_placeholder:False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
            predictions = model.predict_proba(emb)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            k=0     
    #print predictions       
            for i in range(nrof_samples):
                print('\n target'+str(i+1)+':')
                for j in range(cout_per_image[i]):
                    attend_list.append(class_names[best_class_indices[k]])
                    print('%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                    k+=1
origin_img=origin_img[:,:,::-1]
cv2.imwrite('tmp.png',origin_img)
origin_img=cv2.imread('tmp.png')
for i in range(k):
    j=Wlist[i]
    # cv2.rectangle(origin_img, (j[0], j[1]), (j[2], j[3]), (0, 255, 0),1)
    font = cv2.FONT_HERSHEY_COMPLEX
    text = str(class_names[best_class_indices[i]])+' '+str(round(best_class_probabilities[i],3))
    cv2.putText(origin_img, text, (j[0]-60, j[1]-10), font, 1, (255,0,255),1)
cv2.imwrite(filename_base+'out.png',origin_img)
cv2.imshow('output',origin_img)
cv2.waitKey()
data=pd.read_csv('class.csv')
name=data['name'].values
name_list=name.tolist()
for i in range(len(name_list)):
    if name_list[i] in attend_list:
        data.loc[i,'state']='attend'
    else :
        data.loc[i,'state']='cut class'
data.to_csv(filename_base+'class_record.csv',index=0)
print(data)
