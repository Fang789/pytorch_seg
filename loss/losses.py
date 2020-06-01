import torch.nn as nn
import torch
from . import functional as Fu
from torch.nn import functional as F

class CrossEntropy(nn.Module):
	def __init__(self, ignore_label=-1, weight=None):
		super(CrossEntropy, self).__init__()
		self.ignore_label = ignore_label
		self.criterion = nn.CrossEntropyLoss(weight=weight, 
											 ignore_index=ignore_label)

	def forward(self, score, target):
		ph, pw = score.size(2), score.size(3)
		h, w = target.size(1), target.size(2)
		if ph != h or pw != w:
			score = F.upsample(
					input=score, size=(h, w), mode='bilinear')

		loss = self.criterion(score, target)

		return loss

class OhemCrossEntropy(nn.Module): 
	def __init__(self, ignore_label=-1, thres=0.7, 
		min_kept=100000, weight=None): 
		super(OhemCrossEntropy, self).__init__() 
		self.thresh = thres
		self.min_kept = max(1, min_kept)
		self.ignore_label = ignore_label 
		self.criterion = nn.CrossEntropyLoss(weight=weight, 
											 ignore_index=ignore_label, 
											 reduction='none') 

	def forward(self, score, target, **kwargs):
		ph, pw = score.size(2), score.size(3)
		h, w = target.size(1), target.size(2)
		if ph != h or pw != w:
			score = F.upsample(input=score, size=(h, w), mode='bilinear')
		pred = F.softmax(score, dim=1)
		pixel_losses = self.criterion(score, target).contiguous().view(-1)
		mask = target.contiguous().view(-1) != self.ignore_label         
		
		tmp_target = target.clone() 
		tmp_target[tmp_target == self.ignore_label] = 0 
		pred = pred.gather(1, tmp_target.unsqueeze(1)) 
		pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
		min_value = pred[min(self.min_kept, pred.numel() - 1)] 
		threshold = max(min_value, self.thresh) 
		
		pixel_losses = pixel_losses[mask][ind]
		pixel_losses = pixel_losses[pred < threshold] 
		return pixel_losses.mean()

class JaccardLoss(nn.Module):

	def __init__(self, eps=1., ignore_channels=None, **kwargs):
		super().__init__(**kwargs)
		self.eps = eps
		self.ignore_channels = ignore_channels

	def forward(self, y_pr, y_gt):
		y_pr = F.softmax(y_pr, dim=1)
		return 1 - Fu.jaccard(
			y_pr, y_gt,
			eps=self.eps,
			threshold=None,
			ignore_channels=self.ignore_channels,
		)

class DiceLoss(nn.Module):

	def __init__(self, eps=1., beta=1., ignore_channels=None, **kwargs):
		super().__init__(**kwargs)
		self.eps = eps
		self.beta = beta
		self.ignore_channels = ignore_channels

	def forward(self, y_pr, y_gt):
		y_pr = F.softmax(y_pr, dim=1)
		return 1 - Fu.f_score(
			y_pr, y_gt,
			beta=self.beta,
			eps=self.eps,
			threshold=None,
			ignore_channels=self.ignore_channels,
		)

class CategoricalCELoss(nn.Module):

	def __init__(self, class_weights=None,ignore_channels=None,**kwargs):
		super().__init__(**kwargs)
		self.class_weights = class_weights if class_weights is not None else 1
		self.ignore_channels = ignore_channels

	def forward(self, y_pr, y_gt):
		y_pr = F.softmax(y_pr, dim=1)
		return Fu.categorical_crossentropy(
			y_pr, y_gt,
			class_weights=self.class_weights,
			ignore_channels=self.ignore_channels,
		)

class CategoricalFocalLoss(nn.Module):

	def __init__(self, alpha=0.25, gamma=2.,activation="softmax",ignore_channels=None,**kwargs):
		super().__init__(**kwargs)
		self.alpha = alpha
		self.gamma = gamma
		self.ignore_channels=ignore_channels
		if activation=="sigmoid":
			self.activation = nn.Sigmoid()
		elif activation =="softmax":
			self.activation = nn.Softmax(dim=1)

	def forward(self,y_pr, y_gt):
		y_pr = self.activation(y_pr)
		return Fu.categorical_focal_loss(
			y_pr, y_gt,
			alpha=self.alpha,
			gamma=self.gamma,
			ignore_channels=self.ignore_channels,
		)

