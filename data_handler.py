# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
import os
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools

from config import cfg
from utils import convert_from_color, get_random_pos, metrics

# Dataset class
class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, ids, data_files=None, label_files=None,
							cache=False, augmentation=True):
		"""
		Accepts a list of ids.
		Retrieves the data file names and label file names based on the ids.
		checks if the files exist or not.
		Finally, create cache if required (takes up significant memory)
		"""
		if data_files is None:
			data_files=cfg.DATA_FOLDER

		if label_files is None:
			label_files=cfg.LABEL_FOLDER

		super(CustomDataset, self).__init__()
		
		self.augmentation = augmentation
		self.cache = cache
		
		# List of files
		self.data_files = [cfg.DATA_FOLDER.format(id) for id in ids]
		self.label_files = [cfg.LABEL_FOLDER.format(id) for id in ids]

		# Sanity check : raise an error if some files do not exist
		for f in self.data_files + self.label_files:
			if not os.path.isfile(f):
				raise KeyError('{} is not a file !'.format(f))
		
		# Initialize cache dicts
		if self.cache:
			self.data_cache_ = {}
			self.label_cache_ = {}
			
	
	def __len__(self):
		# Default epoch size is 10 000 samples
		return 10000
	
	@classmethod
	def data_augmentation(cls, *arrays, **kwargs):
		flip = kwargs.get('flip', True)
		mirror = kwargs.get('mirror', True)

		will_flip, will_mirror = False, False
		
		if flip and random.random() < 0.5:
			will_flip = True
		if mirror and random.random() < 0.5:
			will_mirror = True
		
		results = []
		for array in arrays:
			if will_flip:
				if len(array.shape) == 2:
					array = array[::-1, :]
				else:
					array = array[:, ::-1, :]
			if will_mirror:
				if len(array.shape) == 2:
					array = array[:, ::-1]
				else:
					array = array[:, :, ::-1]
			results.append(np.copy(array))
			
		return tuple(results)
	
	def __getitem__(self, i):
		# Pick a random image
		random_idx = random.randint(0, len(self.data_files) - 1)
		
		# If the tile hasn't been loaded yet, put in cache
		if random_idx in self.data_cache_.keys():
			data = self.data_cache_[random_idx]
		else:
			# Data is normalized in [0, 1]
			data = 1/255.0 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
			if self.cache:
				self.data_cache_[random_idx] = data
			
		if random_idx in self.label_cache_.keys():
			label = self.label_cache_[random_idx]
		else: 
			# Labels are converted from RGB to their numeric values
			label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
			if self.cache:
				self.label_cache_[random_idx] = label

		# Get a random patch
		x1, x2, y1, y2 = get_random_pos(data, cfg.WINDOW_SIZE)
		data_p = data[:, x1:x2, y1:y2]
		label_p = label[x1:x2, y1:y2]
		
		# Data augmentation
		data_p, label_p = self.data_augmentation(data_p, label_p)

		# Return the torch.Tensor values
		return (torch.from_numpy(data_p),
				torch.from_numpy(label_p))