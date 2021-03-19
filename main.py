import torch
import dataset
import vocabulary
import preparation
import constants as C
from experiment import Experiment

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# preparation.prepare(C.ROOT, dataType, dataSubtype, set(), taskType="OpenEnded")

exp = Experiment()
exp.load_model()
# exp.test("abstract_v002_test2015_000000030000.png", "who is under the tree?")
# exp.test("abstract_v002_test2015_000000030004.png", "is there any pictures on the wall?")
# exp.test("COCO_test2015_000000000016.jpg", "what is the man doing?")
# exp.test("COCO_test2015_000000000027.jpg", "what is this object?")

exp.test("index.jpg", "what is in this picture?")
exp.test("index.jpg", "what color is this barn?")
exp.test("index.jpg", "are there trees in this picture")

print()