# imports and stuff
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

import os
import sys
import urllib
import gflags

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable

from config import cfg
from model import MSDNet
from data_handler import CustomDataset
from utils import convert_to_color, convert_from_color, CrossEntropy2d, accuracy, sliding_window, count_sliding_window
from utils import grouper, metrics


def train(net, train_loader, optimizer, epochs, scheduler=None, weights=None, save_epoch=5, continue_epoch=1):
	# Default arguments
	if weights is None:
		weights=cfg.WEIGHTS
	
	losses = np.zeros(1000000)
	mean_losses = np.zeros(100000000)
	weights = weights.cuda()

	all_losses = []
	all_accs = []

	criterion = nn.NLLLoss2d(weight=weights)
	iter_ = 0
	
	for e in range(continue_epoch, epochs + 1):
		if scheduler is not None:
			scheduler.step()
		net.train()

		epoch_losses = []
		epoch_accs = []
		
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = Variable(data.cuda()), Variable(target.cuda())
			optimizer.zero_grad()
			output = net(data)
			loss = CrossEntropy2d(output, target, weight=weights)
			# print loss
			# print '-'*80

			loss.backward()
			optimizer.step()
			
			# Get current images
			rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
			pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
			gt = target.data.cpu().numpy()[0]

			# get current loss
			epoch_losses.append(loss.data[0])

			# get current accuracy
			epoch_accs.append(accuracy(pred, gt))

			losses[iter_] = loss.data[0]
			mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
			if iter_ % 100 == 0:
				print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
					e, epochs, batch_idx, len(train_loader),
					100. * batch_idx / len(train_loader), np.mean(epoch_losses), np.mean(epoch_accs)))

				if iter_ % 500 == 0:
					# plt.plot(mean_losses[:iter_]) and plt.show()
					# fig = plt.figure()
					# fig.add_subplot(131)
					# plt.imshow(rgb)
					# plt.title('RGB')
					# fig.add_subplot(132)
					# plt.imshow(convert_to_color(gt))
					# plt.title('Ground truth')
					# fig.add_subplot(133)
					# plt.title('Prediction')
					# plt.imshow(convert_to_color(pred))
					# plt.show()

					# Save RGB image, ground truth and prediction
					im_name = 'epoch_{}_iter_{}_{}.png'
					np.save(cfg.output_dir + 'training/' + im_name.format(e, iter_, 'SEG'), pred)
					io.imsave(cfg.output_dir + 'training/' + im_name.format(e, iter_, 'RGB'), rgb)
					io.imsave(cfg.output_dir + 'training/' + im_name.format(e, iter_, 'PRED'), convert_to_color(pred))
					io.imsave(cfg.output_dir + 'training/' + im_name.format(e, iter_, 'GT'), convert_to_color(gt))

			iter_ += 1
			
			del(data, target, loss)

		# Save current epoch loss and accuracy
		current_epoch_loss = np.mean(epoch_losses)
		current_epoch_acc = np.mean(epoch_accs)

		all_losses.append(current_epoch_loss)
		all_accs.append(current_epoch_acc)

		print '-'*80
		print "Epoch {}/{} |\t Loss: {} |\t Accuracy: {}".format(e, epochs, current_epoch_loss, current_epoch_acc)
		print '-'*80
		
		# Check if current loss is better than last 3 losses
		if e % save_epoch == 0 or (np.asarray(all_losses[-4:-1]) > current_epoch_loss).sum() == 3:
			# We validate with the largest possible stride for faster computing
			# acc = test(net, test_ids, all=False, stride=min(cfg.WINDOW_SIZE))
			torch.save(net.state_dict(), cfg.exp_dir + cfg.MODEL_PATH.format(e))
	torch.save(net.state_dict(), cfg.exp_dir + 'final_model_50_epochs')

	# Save all losses and Accuracies in a numpy array
	np.save(cfg.exp_dir + 'loss_50_epoch', np.asarray(all_losses))
	np.save(cfg.exp_dir + 'acc_50_epoch', np.asarray(all_accs))


def pred_and_display(net, test_ids, stride=None, batch_size=None, window_size=None):
	# Default params
	if stride is None:
		stride=cfg.WINDOW_WIDTH
	
	if batch_size is None:
		batch_size=cfg.BATCH_SIZE

	if window_size is None:
		window_size=cfg.WINDOW_SIZE

	test_images = (1 / 255.0 * np.asarray(io.imread(cfg.DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)

	all_preds = []
	net.eval()

	for img in tqdm(test_images, total=len(test_ids), leave=False):
		pred = np.zeros(img.shape[:2] + (cfg.N_CLASSES,))

		total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
		# print total
		# exit(0)
		for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
			image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
			image_patches = np.asarray(image_patches)
			image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
			
			# Do the inference
			outs = net(image_patches)
			outs = outs.data.cpu().numpy()

			for out, (x, y, w, h) in zip(outs, coords):
				out = out.transpose((1,2,0))
				pred[x:x+w, y:y+h] += out
			del(outs)

		pred = np.argmax(pred, axis=-1)

		# Display the result
		# clear_output()
		fig = plt.figure()
		fig.add_subplot(1,2,1)
		plt.imshow(np.asarray(255 * img, dtype='uint8'))
		fig.add_subplot(1,2,2)
		plt.imshow(convert_to_color(pred))
		plt.show()

		all_preds.append(pred)
	return all_preds



def test(net, test_ids, all=False, stride=None, batch_size=None, window_size=None):
	# Default params
	if stride is None:
		stride=cfg.WINDOW_WIDTH
	
	if batch_size is None:
		batch_size=cfg.BATCH_SIZE
	
	if window_size is None:
		window_size=cfg.WINDOW_SIZE
	
	# Use the network on the test set
	test_images = (1 / 255.0 * np.asarray(io.imread(cfg.DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
	test_labels = (np.asarray(io.imread(cfg.LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
	eroded_labels = (convert_from_color(io.imread(cfg.ERODED_FOLDER.format(id))) for id in test_ids)
	
	all_preds = []
	all_gts = []
	
	# Switch the network to inference mode
	net.eval()

	# Start a loop to get image, ground truth and eroded ground truth
	for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
		# container for predection
		pred = np.zeros(img.shape[:2] + (cfg.N_CLASSES,))

		total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
		print "Total windows in image: {}".format(total)
		for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
			# Display in progress results
			print "{} of {} done....".format(i, total)
			"""if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
					_pred = np.argmax(pred, axis=-1)
					fig = plt.figure()
					fig.add_subplot(1,3,1)
					plt.imshow(np.asarray(255 * img, dtype='uint8'))
					fig.add_subplot(1,3,2)
					plt.imshow(convert_to_color(_pred))
					fig.add_subplot(1,3,3)
					plt.imshow(gt)
					# clear_output()
					plt.show()"""
					
			# Build the tensor
			image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
			image_patches = np.asarray(image_patches)
			image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
			
			# Do the inference
			outs = net(image_patches)
			outs = outs.data.cpu().numpy()
			
			# Fill in the results array
			for out, (x, y, w, h) in zip(outs, coords):
				out = out.transpose((1,2,0))
				pred[x:x+w, y:y+h] += out
			del(outs)

		pred = np.argmax(pred, axis=-1)

		# Display the result
		# clear_output()
		# fig = plt.figure()
		# fig.add_subplot(1,3,1)
		# plt.imshow(np.asarray(255 * img, dtype='uint8'))
		# fig.add_subplot(1,3,2)
		# plt.imshow(convert_to_color(pred))
		# fig.add_subplot(1,3,3)
		# plt.imshow(gt)
		# plt.show()

		all_preds.append(pred)
		all_gts.append(gt_e)

		# clear_output()
		# Compute some metrics
		metrics(pred.ravel(), gt_e.ravel())
		accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
	if all:
		return accuracy, all_preds, all_gts
	else:
		return accuracy

def main(save=False, pretrained=True, task='viz'):
	# instantiate the network
	net = MSDNet()

	# Load the model on GPU
	net.cuda()

	# Load the datasets
	if cfg.DATASET == 'Potsdam':
		all_files = sorted(glob(cfg.LABEL_FOLDER.replace('{}', '*')))
		all_ids = ["_".join(f.split('_')[3:5]) for f in all_files]
	elif cfg.DATASET == 'Vaihingen':
		#all_ids = 
		all_files = sorted(glob(cfg.LABEL_FOLDER.replace('{}', '*')))
		all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
	else:
		# For all other datasets
		all_files = sorted(glob(cfg.LABEL_FOLDER.replace('{}', '*')))
		all_ids = [f.split('/')[-1].split('.')[0] for f in all_files]
	
	# Random tile numbers for train/test split
	# train_ids = random.sample(all_ids, 2 * len(all_ids) // 3 + 1)
	# test_ids = list(set(all_ids) - set(train_ids))

	# Get all in training for now
	train_ids = all_ids
	test_ids = []

	# Exemple of a train/test split on Vaihingen :
	# train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
	# test_ids = ['39']#, '40'] 

	print("Tiles for training : ", train_ids)
	print("Tiles for testing : ", test_ids)

	train_set = CustomDataset(train_ids, cache=cfg.CACHE)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.BATCH_SIZE)

	# Design the optimizer
	# base_lr = 0.01
	# Change LR for Adam optimizer
	base_lr = 0.0005
	params_dict = dict(net.named_parameters())
	params = []
	for key, value in params_dict.items():
		if '_D' in key:
			# Decoder weights are trained at the nominal learning rate
			params += [{'params':[value],'lr': base_lr}]
		else:
			# Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
			params += [{'params':[value],'lr': base_lr / 2}]

	# optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
	# Change to Adam optimizer
	optimizer = optim.SGD(net.parameters(), lr=base_lr)
	# We define the scheduler
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

	continue_epoch = 1
	if pretrained:
		# load the pretrained model
		net.load_state_dict(torch.load(cfg.model_final))

		if 'epoch' in cfg.model_final:
			continue_epoch = int(cfg.model_final.split('_')[-1]) + 1

	if task == 'train':
		# Train model
		train(net, train_loader, optimizer, 50, scheduler, continue_epoch=continue_epoch)
	elif task == 'test':
		# Run tests
		_, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
	elif task == 'viz':
		all_preds = pred_and_display(net, test_ids, stride=32)

	if save:
		for p, id_ in zip(all_preds, test_ids):
			img = convert_to_color(p)
			io.imsave(cfg.output_dir + 'inference_tile_{}.png'.format(id_), img)

def get_config():
	return cfg

if __name__ == '__main__':
	gflags.DEFINE_boolean('write_images', False, 'Save the outputs produced by the model?')
	gflags.DEFINE_boolean('pretrained', True, 'Use pretrained model')

	gflags.DEFINE_string('exp_dir', None, 'Path to experiment dump directory')
	gflags.DEFINE_string('cfg', None, 'Path to experiment configuration file')

	gflags.DEFINE_boolean('train', False, 'Train the network')
	gflags.DEFINE_boolean('viz', True, 'Run tests with vizualizations only')

	gflags.FLAGS(sys.argv)
	cfg.init_paths(gflags.FLAGS.cfg, gflags.FLAGS.exp_dir)

	task = 'viz'
	if gflags.FLAGS.train is True:
		task = 'train'
	elif gflags.FLAGS.viz is False:
		task = 'test'

	main(save=gflags.FLAGS.write_images, pretrained=gflags.FLAGS.pretrained, task=task)
