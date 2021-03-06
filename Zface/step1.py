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
data=pd.read_csv('class.csv')
name=data['name'].values
name_list=name.tolist()
# path0='~/文档/zjt/zzzGit/Zface/'
path_list=list()
for i in range(len(name_list)):
    os.mkdir(str(name_list[i]))
    video_path=str(name_list[i])+'.mp4'
    video_capture = cv2.VideoCapture(video_path)
    TOTAL_FRAME = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) #获取视频总帧数
    print(TOTAL_FRAME)
    for j in range(TOTAL_FRAME):
        flag,img=video_capture.read()
        if flag:
            img=cv2.transpose(img)
            cv2.imwrite(str(name_list[i])+'/'+str(name_list[i])+str(j).zfill(3)+'.jpg',img)

os.system('python3 ~/文档/zjt/facenet/src/align/align_dataset_mtcnn.py ~/文档/zjt/facenet/Zface/old ~/文档/zjt/facenet/Zface/new --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25')


input_dir='~/文档/zjt/facenet/Zface/old'
output_dir='~/文档/zjt/facenet/Zface/new'
image_size=160
margin=32
random_order='store_true'
gpu_memory_fraction=0.25
detect_multiple_faces=False


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
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
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
                            text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                    else:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        
print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


#  python3 src/classifier.py TRAIN zdata/new src/models/20180402-114759 src/models/zclassify.pkl
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

mode='TRAIN'
data_dir='~/文档/zjt/facenet/Zface/new'
model='~/文档/zjt/facenet/src/models/20180402-114759'
classifier_filename='Zclassify.pkl'
use_split_dataset='store_true'
image_size=160
seed=666
min_nrof_images_per_class=20
nrof_train_images_per_class=120
batch_size=100


with tf.Graph().as_default():
    
    with tf.Session() as sess:
        
        np.random.seed(seed=seed)
        
        if use_split_dataset:
            dataset_tmp = facenet.get_dataset(data_dir)
            train_set, test_set = split_dataset(dataset_tmp, min_nrof_images_per_class, nrof_train_images_per_class)
            if (mode=='TRAIN'):
                dataset = train_set
            elif (mode=='CLASSIFY'):
                dataset = test_set
        else:
            dataset = facenet.get_dataset(data_dir)
         
                
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        
        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(model)
        
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
        
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        if (mode=='TRAIN'):
            # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)
        
            # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            

