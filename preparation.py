import os
import json
import nltk
import utils
import vocabulary
import numpy as np
import constants as C


def prepare(root, dataType, dataSubtype, validAnsSet, taskType="OpenEnded"):
	"""
	helper function to prepare vqa dataset given the params
	:param root: location of VQAv2 dataset
	:param dataType: 'mscoco' for real and 'abstract_v002' for abstract
	:param dataSubtype: 'train / valid' + 'year'. Cannot used for test.
	:param validAnsSet: answer vocabulary (i.e. answers containing unknown vocab is invalid)
	:param taskType: 'MultipleChoice' or 'OpenEnded'. We only consider 'OpenEnded' here.
	:return: list of length=num_questions, each entry is a dict with following keys: {image_name, image_path,
		     question_id, question_tokens, question_string, all_answers, valid_answers, question_len}
	"""
	imgDir = os.path.join(root, "Images")
	tokenizer = nltk.tokenize.TweetTokenizer()  # deal with ' better
	# initialize the locations of files
	imgNameFormat, annPath, qstPath = utils.formName(root, dataType, dataSubtype, taskType)
	# load annotations, save as dict {key=qstID, value=ann}
	assert os.path.exists(annPath)
	with open(annPath) as file:
		annotations = json.load(file)["annotations"]
		qst2ann_dict = {ann["question_id"]: ann for ann in annotations}
	# load questions
	assert os.path.exists(qstPath)
	with open(qstPath) as file:
		questions = json.load(file)["questions"]
	# combine data we need
	dataset = []
	num_unk_ans = 0
	for i, qst in enumerate(questions):
		# print process
		if (i+1) % 10000 == 0:
			print("Processed %d / %d questions" % (i+1, len(questions)))
		imgID = qst["image_id"]
		qstID = qst["question_id"]
		qstStr = qst["question"]
		imgName = imgNameFormat % imgID
		imgPath = os.path.join(imgDir, imgName)
		qstToken = tokenizer.tokenize(qstStr)  # a list of tokens
		qstLen = len(qstToken)  # for attention
		imgInfo = dict(image_name=imgName, image_path=imgPath,
		               question_id=qstID, question_tokens=qstToken,
		               question_string=qstStr, question_len=qstLen)
		# load answers for train or validation
		ann = qst2ann_dict[qstID]
		allAns = [a["answer"] for a in ann["answers"]]
		validAns = [a for a in allAns if a in validAnsSet]
		# answers may not appear in our vocabulary
		if len(validAns) == 0:
			validAns = ["<unk>"]
			num_unk_ans += 1
		imgInfo["all_answers"] = allAns
		imgInfo["valid_answers"] = validAns
		dataset.append(imgInfo)
	return dataset


def load_dataset(mode, trainFile="train.npy", validFile="valid.npy",
                 ansFilename="vocab_answers.txt", qstFilename="vocab_questions.txt"):
	"""
	load prepared train and validation dataset
	:param mode: 'train' or 'valid'
	:param trainFile: filename of prepare train dataset
	:param validFile: filename of prepare validation dataset
	:param ansFilename: filename of answer vocab
	:param qstFilename: filename of question vocab
	:return: prepare train data if mode == 'train' else validation data
	"""
	# load savedVocab
	ansVocab, _ = vocabulary.load_vocab(ansFilename, qstFilename)
	savedAnsSet = set(ansVocab.wordList)
	trainPath = os.path.join(C.DIR_DATA, trainFile)
	validPath = os.path.join(C.DIR_DATA, validFile)
	# if no saved files, call `prepare()` and save it
	if not (os.path.exists(trainPath) and os.path.exists(validPath)):
		train, validation = [], []
		for dataType in C.TRAIN:
			subtypes = C.TRAIN[dataType]
			for subtype in subtypes:
				print("prepare %s %s" % (dataType, subtype))
				train += prepare(C.ROOT, dataType, subtype, savedAnsSet)
		for dataType in C.VALIDATION:
			subtypes = C.VALIDATION[dataType]
			for subtype in subtypes:
				print("prepare %s %s" % (dataType, subtype))
				validation += prepare(C.ROOT, dataType, subtype, savedAnsSet)
		np.save(trainPath, np.array(train))
		np.save(validPath, np.array(validation))
	# load .npy files
	train, validation = np.load(trainPath, allow_pickle=True), np.load(validPath, allow_pickle=True)
	return train if mode == "train" else validation
