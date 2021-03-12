import os
import constants as C


def formName(root, dataType, dataSubtype, taskType="OpenEnded"):
	"""
	helper function to form image name template, annotation file path, question file path with params.
	:param root: location of VQAv2 dataset
	:param dataType: 'mscoco' for real and 'abstract_v002' for abstract
	:param dataSubtype: 'train / valid / test' + 'year'
	:param taskType: 'MultipleChoice' or 'OpenEnded'. We only consider 'OpenEnded' here.
	:return: imageNameFormat, annPath, qstPath
	"""
	if dataType == "mscoco":
		imgNameFormat = os.path.join(dataType, dataSubtype, "COCO_" + dataSubtype + "_%012d" + ".jpg")
		annFilename = C.VERSION + "%s_%s_annotations.json" % (dataType, dataSubtype)
		qstFilename = C.VERSION + "%s_%s_%s_questions.json" % (taskType, dataType, dataSubtype)
	else:  # dataType == "abstract_v002"
		imgFolder = "scene_img_" + dataType + "_" + dataSubtype
		imgNameFormat = os.path.join(imgFolder, dataType + "_" + dataSubtype + "_%012d" + ".png")
		annFilename = "%s_%s_annotations.json" % (dataType, dataSubtype)
		qstFilename = "%s_%s_%s_questions.json" % (taskType, dataType, dataSubtype)
	annPath = os.path.join(root, "Annotations", annFilename)
	qstPath = os.path.join(root, "Questions", qstFilename)
	return imgNameFormat, annPath, qstPath