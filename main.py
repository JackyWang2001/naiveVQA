import torch
import dataset
import vocabulary
import preparation
import constants as C
from experiment import Experiment

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dataType = "mscoco"
dataSubtype = "train2014"
# preparation.prepare(C.ROOT, dataType, dataSubtype, set(), taskType="OpenEnded")

exp = Experiment()
exp.run(5)


print()