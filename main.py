import torch
import dataset
import vocabulary
import preparation
import constants as C


from torch.utils.data import DataLoader

dataType = "mscoco"
dataSubtype = "train2014"
# preparation.prepare(C.ROOT, dataType, dataSubtype, set(), taskType="OpenEnded")

dataset = dataset.VQAv2Dataset("train")
loader = DataLoader(dataset)

print()