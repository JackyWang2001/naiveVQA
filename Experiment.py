import os
import torch
import torch.nn as nn
import constants as C
from model import VqaModel
from dataset import VQAv2Dataset
from torch.utils.data import DataLoader


class Experiment(object):
	def __init__(self, model_name="model.pt", learning_rate=0.01, embed_size=1024, word_embed_size=300,
	             num_layers=2, hidden_size=512):
		super(Experiment, self).__init__()
		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.model_path = os.path.join(C.DIR_MODELS, model_name)
		# load datasets and vocabulary
		self.trainDataset = VQAv2Dataset(mode="train")
		self.validDataset = VQAv2Dataset(mode="validation")
		self.trainLoader = DataLoader(self.trainDataset)
		self.validLoader = DataLoader(self.validDataset)
		qstVocab_size = len(self.trainDataset.qstVocab)
		ansVocab_size = len(self.validDataset.ansVocab)
		# define model
		self.model = VqaModel(embed_size=embed_size,
		                      qstVocab_size=qstVocab_size,
		                      ansVocab_size=ansVocab_size,
		                      word_embed_size=word_embed_size,
		                      num_layers=num_layers,
		                      hidden_size=hidden_size)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
		# use GPU
		self.model.to(device=self.device)
		self.criterion = nn.CrossEntropyLoss().to(device=self.device)

	def run(self, num_epoch):
		return

	def train(self, batch_size):
		return

	def valid(self, batch_size):
		return

	def save_model(self):
		return
		












