# This file contains a list of constants
from os.path import abspath

ROOT = abspath("/media/jpl/T7/VQAv2")  # location of VQAv2 dataset
VERSION = "v2_"  # we are using VQAv2 dataset
TASK_TYPE = "OpenEnded"

TRAIN = {"mscoco": ["train2014"], "abstract_v002": ["train2017"]}
VALIDATION = {"mscoco": ["val2014"], "abstract_v002": ["val2017"]}
TEST = {"mscoco": ["test2015"], "abstract_v002": ["test2015"]}

DIR_DATA = "datasets"  # directory name of transferred datasets in `preparation`
DIR_MODELS = "saved_models"  # directory name of saved models `Experiment`
