import os
import shutil
from glob import glob
import random
import cv2
random.seed(3407)
def mymovefile(srcfile,dstpath):                       # 移动函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.move(srcfile, dstpath + fname)          # 移动文件
        print ("move %s -> %s"%(srcfile, dstpath + fname))
 
def split_dataset():
    src_dir = '/root/The_Project_of_Zhou/CV/Project/data/IRDehaze/train/ir/'
    dst_dir = '/root/The_Project_of_Zhou/CV/Project/data/IRDehaze/test/ir/'                                    # 目的路径记得加斜杠
    file_index = []
    for i in range(0,1080):
        file_index.append(i)
    random.shuffle(file_index)
    for i in range(0,108):
        fname = src_dir + str(file_index[i])+'.jpg'
        dst_name = dst_dir + str(file_index[i])+'.jpg'
        shutil.move(fname, dst_name)
    
    
def read_img(filename):
	img = cv2.imread(filename)
	return img[:, :, ::-1].astype('float32') / 255.0
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
img = read_img("./data/IRDehaze/train/hazy/229.jpg")
print(img.shape)


    

