import keras # 导入Keras
import numpy as np
import matplotlib.pyplot as plt # matplotlib的操作与matlab特别相似
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽AVX2FMA的warning
datapath='dataset/mnist.npz'
data=np.load(datapath)
x_train=data['x_train']#x_train表示图像的训练集
y_train=data['y_train']#y_train表示图像对应的标签训练集
x_test=data['x_test']#x_test表示图像的测试集
y_test=data['y_test']#y_test表示图像对应的标签测试集
# plt.imshow(x_train[12],cmap='gray')
# plt.title(y_train[12])
# plt.show()
##数据初始化
rows=28
cols=28
numcategory=10
input_shape=(rows,cols,1)
x_train=x_train.reshape(x_train.shape[0],rows,cols,1)
x_test=x_test.reshape(x_test.shape[0],rows,cols,1)
x_train=np.float32(x_train)
x_test=np.float32(x_test)
x_train/=255
x_test/=255
y_train=keras.utils.to_categorical(y_train,numcategory)
y_test=keras.utils.to_categorical(y_test,numcategory)
##构建model
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numcategory, activation='softmax'))
##编译model
model.compile(optimizer=keras.optimizers.Adadelta(),loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
batch_size = 128
num_epoch = 15
##model training
model_history =model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(x_test, y_test))
##model evalution
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])
##model saving
model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
model.save_weights("model_digit.h5")
print("Saved model to disk")
