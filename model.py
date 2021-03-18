import torch
import torch.nn as nn
from torchvision import models


class VqaSimpleBaseline(nn.Module):
	""" Simple VQA Model """
	def __init__(self, embed_size, qstVocab_size, ansVocab_size, word_embed_size, num_layers, hidden_size):
		super(VqaSimpleBaseline, self).__init__()
		self.img_encoder = ImgEncoder(embed_size)
		self.qst_encoder = QstEncoder(qstVocab_size, word_embed_size, embed_size, num_layers, hidden_size)
		self.tanh = nn.Tanh()
		self.fc = nn.Linear(embed_size, ansVocab_size)

	def forward(self, img, qst):
		img_feature = self.img_encoder(img)  # [batch_size, embed_size]
		qst_feature = self.qst_encoder(qst)  # [batch_size, embed_size]
		y = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
		y = self.tanh(y)
		y = self.fc(y)  # [batch_size, ansVocab_size]
		return y


class ImgEncoder(nn.Module):
	def __init__(self, embed_size):
		super(ImgEncoder, self).__init__()
		self.model = models.mobilenet_v2(pretrained=True)
		# remove the fully connected layer
		in_features = self.model.classifier[1].in_features
		self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
		self.fc = nn.Linear(in_features, embed_size)

	def forward(self, image):
		with torch.no_grad():
			img_feature = self.model(image)  # [batch_size, 1280]
		img_feature = self.fc(img_feature)  # [batch_size, embed_size]
		return img_feature


class QstEncoder(nn.Module):
	def __init__(self, qstVocab_size, word_embed_size, embed_size, num_layers, hidden_size):
		super(QstEncoder, self).__init__()
		self.word2vec = nn.Embedding(qstVocab_size, word_embed_size)
		self.tanh = nn.Tanh()
		self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
		self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)  # 2 for hidden and cell states

	def forward(self, question):
		qst_vec = self.word2vec(question)  # [batch_size, max_len_qst=30, word_embed_size]
		qst_vec = self.tanh(qst_vec)
		qst_vec = qst_vec.transpose(0, 1)  # [max_len_qst, batch_size, word_embed_size]
		_, (hidden, cell) = self.lstm(qst_vec)  # [num_layers, batch_size, hidden_size]
		qst_feature = torch.cat((hidden, cell), 2)  # [num_layers, batch_size, 2*hidden_size]
		qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers, 2*hidden_size]
		qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size]
		qst_feature = self.tanh(qst_feature)
		qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]
		return qst_feature




