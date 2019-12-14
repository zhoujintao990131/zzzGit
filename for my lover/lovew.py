from wordcloud import WordCloud, STOPWORDS
from imageio import imread
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import csv
import numpy as np
from PIL import Image
import cv2
# 获取文章内容
with open("test.txt") as f:
    contents = f.read()
print("contents变量的类型：", type(contents))

# 使用jieba分词，获取词的列表
contents_cut = jieba.cut(contents)
print("contents_cut变量的类型：", type(contents_cut))
contents_list = " ".join(contents_cut)
print("contents_list变量的类型：", type(contents_list))
# alice_mask = np.array(Image.open("mask1.jpeg"))
mask=cv2.imread('heart1.jpg')
# mask=cv2.resize(mask,(1080,1080))
print(mask.shape)
# 制作词云图，collocations避免词云图中词的重复，mask定义词云图的形状，图片要有背景色
wc = WordCloud(stopwords=STOPWORDS.add("海涅"), collocations=False,
               background_color="white",
               font_path='/home/zjt990131/下载/font21257/有点甜.ttf',
               width=400, height=300, random_state=42,
               mask=mask,min_word_length=3)
            #    mask=imread('heart.jpg',pilmode="RGB"))
wc.generate(contents_list)
wc.to_file("out.png")
cv2.imwrite('1.jpg',mask)
# # 使用CountVectorizer统计词频
# cv = CountVectorizer()
# contents_count = cv.fit_transform([contents_list])
# # 词有哪些
# list1 = cv.get_feature_names()
# # 词的频率
# list2 = contents_count.toarray().tolist()[0] 
# # 将词与频率一一对应
# contents_dict = dict(zip(list1, list2))
# # 输出csv文件,newline=""，解决输出的csv隔行问题
# with open("caifu_output.csv", 'w', newline="") as f:
#     writer = csv.writer(f)
#     for key, value in contents_dict.items():
#         writer.writerow([key, value])
