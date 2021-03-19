import os
import time
import nltk
import torch
import numpy as np
import constants as C
from PIL import Image
from torch import nn
from torch import optim
from dataset import VQAv2Dataset
from model import VqaSimpleBaseline
from torch.utils.data import DataLoader


class Experiment(object):
	def __init__(self, model_name="model.pt", learning_rate=0.001, batch_size=64,
	             embed_size=1024, word_embed_size=300, num_layers=2, hidden_size=512):
		super(Experiment, self).__init__()
		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.model_path = os.path.join(C.DIR_MODELS, model_name)
		# load datasets and vocabulary
		self.trainDataset = VQAv2Dataset(mode="train")
		self.validDataset = VQAv2Dataset(mode="validation")
		qstVocab_size = len(self.trainDataset.qstVocab)
		ansVocab_size = len(self.trainDataset.ansVocab)
		self.trainLoader = DataLoader(self.trainDataset, batch_size=batch_size, num_workers=48)
		self.validLoader = DataLoader(self.validDataset, batch_size=batch_size, num_workers=48)
		# define model
		self.model = VqaSimpleBaseline(embed_size=embed_size,
		                               word_embed_size=word_embed_size,
		                               num_layers=num_layers,
		                               hidden_size=hidden_size,
		                               qstVocab_size=qstVocab_size,
		                               ansVocab_size=ansVocab_size)
		self.criterion = nn.CrossEntropyLoss()
		params = list(self.model.img_encoder.fc.parameters()) + list(self.model.qst_encoder.parameters()) \
		         + list(self.model.fc1.parameters()) + list(self.model.fc2.parameters())
		self.optimizer = optim.Adam(params, lr=learning_rate)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)
		# use GPU
		self.criterion = self.criterion.to(self.device)
		self.model = self.model.to(self.device)

	def run(self, num_epoch):
		print("running train and validation")
		trainLosses, validLosses = [], []
		# load model if model_path exists
		self.load_model()
		self.model.to(self.device)
		start_time = time.time()
		for epoch in range(num_epoch):
			# train
			trainLoss = self.train()
			trainLosses.append(trainLoss)
			print("%d epoch, training loss = %.2f" % (epoch, trainLoss))
			print("%d epoch, taken %d seconds" % (epoch, time.time() - start_time))
			# # validation
			# validLoss = self.valid()
			# validLosses.append(validLoss)
			# print("%d epoch, validation loss = %d" % (epoch, validLoss))
			self.save_model(epoch, self.model.state_dict(), self.optimizer.state_dict())
		return trainLosses, validLosses

	def train(self):
		self.model.train()
		training_loss = 0.0
		with torch.enable_grad():
			for i, (img, ans, qst) in enumerate(self.trainLoader):
				img, ans, qst = img.to(self.device), ans.to(self.device), qst.to(self.device)
				self.optimizer.zero_grad()
				# compute predictions and losses
				outputs = self.model(img, qst)
				loss = self.criterion(outputs, ans)
				# backpropagation
				loss.backward()
				self.optimizer.step()
				training_loss += loss.item() / len(self.trainLoader)
		self.scheduler.step()
		return training_loss

	def valid(self):
		self.model.eval()
		validation_loss = 0.0
		# with torch.no_grad():
		# 	for i, (img, ans, qst) in enumerate(self.validLoader):
		# 		img, ans, qst = img.to(self.device), ans.to(self.device), qst.to(self.device)
		# validation_loss = validation_loss / len(self.validLoader)
		return validation_loss

	def load_model(self):
		if os.path.exists(self.model_path):
			state = torch.load(self.model_path)
			self.model.load_state_dict(state["modelState"])
			self.optimizer.load_state_dict(state["optState"])

	def save_model(self, epoch, modelState, optState):
		torch.save({"epoch": epoch, "modelState": modelState, "optState": optState},
		           os.path.join(self.model_path))

	def test(self, img_path, qst):
		self.load_model()
		tokenizer = nltk.tokenize.TweetTokenizer()
		self.model.to("cpu")
		img_path = os.path.join(C.DIR_EXP, img_path)
		img = Image.open(img_path).convert("RGB")
		img = self.trainDataset.transform(img)
		img = img.unsqueeze(0)
		qst = tokenizer.tokenize(qst)
		qst2idx = np.array([self.trainDataset.qstVocab("<pad>")] * self.trainDataset.max_len_qst)
		for i in range(len(qst)):
			qst2idx[i] = self.trainDataset.qstVocab(qst[i])
		qst2idx = torch.tensor(qst2idx).unsqueeze(0)
		pred = self.model(img, qst2idx)
		pred = torch.topk(pred, 5).indices.squeeze().tolist()
		pred = [self.trainDataset.ansVocab.idx2word[i] for i in pred]
		print("model answer: " + str(pred))
