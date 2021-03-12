import os
import vocabulary
import preparation
import numpy as np
import constants as C
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class VQAv2Dataset(Dataset):
	"""
	VQA v2 dataset
	"""
	def __init__(self, mode, transform=None, max_len_qst=30, max_num_ans=10,
	             ansFilename="vocab_answers.txt", qstFilename="vocab_questions.txt"):
		"""
		create dataset based on mode
		:param mode: 'train' for training dataset and 'validation' for validation dataset
		:param transform: image transformation or augmentation
		:param max_len_qst: max length of question, default 30
		:param max_num_ans: max number of answer, default 10
		:param ansFilename: answer filename
		:param qstFilename: question filename
		"""
		super(VQAv2Dataset, self).__init__()
		self.mode = mode
		self.root = C.ROOT
		self.vqa = preparation.load_dataset(self.mode)
		self.ansVocab, self.qstVocab = vocabulary.load_vocab(ansFilename, qstFilename)
		self.max_len_qst = max_len_qst
		self.max_num_ans = max_num_ans
		if transform is None:
			self.transform = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				# compute normalizing terms in prepare_dataset.py
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])

	def __len__(self):
		return len(self.vqa)

	def __getitem__(self, item):
		imgPath = self.vqa[item]["image_path"]
		img = Image.open(imgPath).convert("RGB")
		qst2idx = np.array([self.qstVocab("<pad>")] * self.max_len_qst)
		for i, word in enumerate(self.vqa[item]["question_tokens"]):
			if i < self.max_len_qst:
				qst2idx[i] = self.qstVocab(word)
			else:
				break
		ans2idx = [self.ansVocab(word) for word in self.vqa[item]["valid_answers"]]
		ans2idx = np.random.choice(ans2idx)  # choose one answer
		img = self.transform(img)
		return img, ans2idx, qst2idx


