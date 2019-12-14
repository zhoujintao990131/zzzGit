import cv2
for i in range(1,6):
    path='合照'+str(i)+'/'
    name='合照'+str(i)+'.jpg'
    img=cv2.imread(path+name)
    img=cv2.resize(img,(1200,900),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path+name,img)
