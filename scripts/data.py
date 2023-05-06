import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs
def read_img(filename):
	img = cv2.imread(filename)
	return img[:, :, ::-1].astype('float32') / 255.0 ## 归一化 并且将cv2 bgr 转换为rgb
def read_img_grey(filename):
	img = cv2.imread(filename)
	gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	return gray_image.astype('float32') / 255.0 ## 归一化 并且将cv2 bgr 转换为rgb


def write_img(filename, img, to_uint=True):
	if to_uint: img = np.round(img * 255.0).astype('uint8')
	cv2.imwrite(filename, img[:, :, ::-1])
## 数据增强
def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs
def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()

def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()

class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		
		# ir_img = read_img(os.path.join(self.root_dir,'ir',img_name)) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
		# ir_img = ir_img.reshape(512,640,1)
		# source_img = np.concatenate((source_img,ir_img),axis=2)
		# print(source_img.shape)
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)


		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}
class UPairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		
		ir_img = (read_img_grey(os.path.join(self.root_dir,'ir',img_name)) * 2 - 1)*(-1)
		v_img = read_img_grey(os.path.join(self.root_dir,'hsv',img_name)) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
		ir_img = ir_img.reshape(512,640,1)
		v_img = v_img.reshape(512,640,1)
		source_img = np.concatenate((source_img,ir_img,v_img),axis=2)
		# print(source_img.shape)
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)


		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}

