from model import VqaModel
from dataset import VQAv2Dataset
from torch.utils.data import DataLoader


class Experiment(object):
	def __init__(self):
		super(Experiment, self).__init__()
