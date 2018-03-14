import torch
import numpy as np
import yaml
import os

class ProjectConfig:
	def __init__(self, cfg_path=None, exp_dir=None):
		self.WINDOW_WIDTH = None
		self.WINDOW_SIZE = None
		self.STRIDE = None
		self.IN_CHANNELS = None
		self.BATCH_SIZE = None
		self.FOLDER = None
		self.LABELS = None
		self.N_CLASSES = None
		self.WEIGHTS = None
		self.CACHE = None
		self.DATASET = None
		self.model_final = None
		self.MAIN_FOLDER = None
		self.DATA_FOLDER = None
		self.LABEL_FOLDER = None
		
		# Load a default config initially
		# TODO: Change this path
		self.init_paths(cfg_path='./experiment/cfg.yml', exp_dir='./experiment/')

	def init_paths(self, cfg_path=None, exp_dir=None):
		self.cfg_path = cfg_path
		self.exp_dir = exp_dir
		self.output_dir = self.exp_dir + 'output/'
		self.load_config()

	def load_config(self):
		# Load config from the config file path
		with open(self.cfg_path, 'r') as cfg_stream:
			data = yaml.load(cfg_stream)

		# parse this and save the items to their corresponding values
		self.WINDOW_WIDTH = data['model']['window_size']
		self.WINDOW_SIZE = (self.WINDOW_WIDTH, self.WINDOW_WIDTH)

		self.STRIDE = data['model']['stride']
		self.IN_CHANNELS = data['model']['in_channels']
		self.BATCH_SIZE = data['model']['batch_size']

		self.FOLDER = data['data_folder']

		self.LABELS = data['labels']
		self.N_CLASSES = len(self.LABELS)
		self.WEIGHTS = torch.from_numpy(np.asarray(data['class_weights'], dtype=np.float32))

		self.CACHE = data['cache']
		self.DATASET = data['data']['dataset']

		self.model_final = data['model_final_path']

		self.MODEL_PATH = data['model_checkpoint']

		self.MAIN_FOLDER = data['data']['dataset_path']
		self.DATA_FOLDER = self.MAIN_FOLDER + data['data']['input_folder'] + '/{}.' + data['data']['data_format']
		self.LABEL_FOLDER = self.MAIN_FOLDER + data['data']['label_folder'] + '/{}.tif'

# create an object
cfg = ProjectConfig()