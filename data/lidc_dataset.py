import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pdb

class LIDC(Dataset):
	def __init__(self, rater=4, split='Train', data_dir = './', transform=None,fold=0):
		super().__init__()

		self.data_dir = data_dir
		self.rater = rater
		self.transform = transform
		folds = [0,1,2,3,4]
		folds.remove(fold)
		if split == 'Valid':
			self.data, self.targets = torch.load(data_dir+'Fold'+repr(folds[0])+'.pt')
		elif split == 'Train':
			data0, targets0 = torch.load(data_dir+'Fold'+repr(folds[1])+'.pt')
			data1, targets1 = torch.load(data_dir+'Fold'+repr(folds[2])+'.pt')
			data2, targets2 = torch.load(data_dir+'Fold'+repr(folds[3])+'.pt')
			self.data = torch.cat((data0,data1,data2),dim=0)
			self.targets = torch.cat((targets0,targets1,targets2),dim=0)
		else:
			self.data, self.targets = torch.load(data_dir+'Fold'+repr(fold)+'.pt')
		self.targets = self.targets.type(torch.FloatTensor)		   
	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):
		image, label = self.data[index], self.targets[index]
		if self.rater == 4:
			label = (label.sum() > 2).type_as(self.targets)
		else:
			label = label[self.rater]
		image = image.type(torch.FloatTensor)/255.0
		if self.transform is not None:
			image = self.transform(image)
		return image, label

class LIDCSeg(Dataset):
	def __init__(self, rater=4, split='Train', data_dir = './', transform=None):
		super().__init__()

		self.data_dir = data_dir
		self.rater = rater
		self.transform = transform
		self.data, self.targets = torch.load(data_dir+split+'.pt')
		self.targets = self.targets.type(torch.FloatTensor)		   

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):
#		pdb.set_trace()
		image, label = self.data[index], self.targets[index]
		if self.rater == 4:
			label = (label.sum(0) > 2).type_as(self.targets)
		else:
			label = label[self.rater]
#		image = image.type(torch.FloatTensor)/255.0
		if self.transform is not None:
			image = self.transform(image)
		return image, label


