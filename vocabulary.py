import os
import json
import nltk
import utils
import constants as C
from collections import defaultdict


class Vocabulary(object):
    """
    Simple vocabulary wrapper.
    """
    def __init__(self, filename):
        self.filename = filename
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        with open(filename) as file:
            lines = file.readlines()
        self.wordList = [line.strip() for line in lines]
        for word in self.wordList:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word.lower() not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word.lower()]

    def __len__(self):
        return len(self.wordList)


def make_vocab(num_top=2000):
    """
    create vocabulary for data
    :param num_top: number of top answers saved for vocabulary
    :return: vocab for answers, vocab for questions
    """
    tokenizer = nltk.tokenize.TweetTokenizer()  # deal with ' better
    ansDict = defaultdict(lambda: 0)
    qstSet = set()
    data_train_val = {k: C.VALIDATION[k] + C.TRAIN[k] for k in C.TRAIN}
    for dataType in data_train_val:
        subtypes = data_train_val[dataType]
        for subtype in subtypes:
            _, annPath, qstPath = utils.formName(C.ROOT, dataType, subtype, C.TASK_TYPE)
            print("make vocabulary for %s and %s" % (annPath, qstPath))
            with open(annPath) as file:
                annotations = json.load(file)["annotations"]
            with open(qstPath) as file:
                questions = json.load(file)["questions"]
            # add ann, qst into collections
            for annotation in annotations:
                for answer in annotation["answers"]:
                    word = tokenizer.tokenize(answer["answer"].lower())
                    word = [w.replace("'", "") for w in word if w.isalpha() or w.isalnum()]
                    for w in word:
                        ansDict[w.strip()] += 1
            for question in questions:
                words = tokenizer.tokenize(question["question"].lower())
                words = [w.replace("'", "").strip() for w in words if len(w.replace("'", "").strip()) > 0]
                qstSet.update(words)  # don't --> dont
    # choose high-freq vocab for answers
    ansDict = sorted(ansDict, key=ansDict.get, reverse=True)
    assert "<unk>" not in ansDict
    vocabAns = ["<unk>"] + ansDict[:num_top-1]
    print("save %d vocab in total num of %d vocab for answers" % (num_top, len(ansDict)))
    # choose all vocab for questions
    vocabQst = list(qstSet)
    vocabQst.sort()
    vocabQst.insert(0, "<pad>")
    vocabQst.insert(1, "<unk>")
    return vocabAns, vocabQst


def load_vocab(ansFilename="vocab_answers.txt", qstFilename="vocab_questions.txt", num_top=2000):
    """
    load vocabulary file; if vocab file not created, call `make_vocab` and save results
    :param ansFilename: filename of answer vocab
    :param qstFilename: filename of question vocab
    :param num_top: number of top answers saved for vocabulary
    :return: ansVocab, qstVocab
    """
    ansFile = os.path.join("datasets", ansFilename)
    qstFile = os.path.join("datasets", qstFilename)
    # if files not created, create such files
    if not (os.path.isfile(ansFile) and os.path.isfile(qstFile)):
        vocabAns, vocabQst = make_vocab(num_top)
        with open(ansFile, "w") as file:
            file.writelines([w + "\n" for w in vocabAns])
        with open(qstFile, "w") as file:
            file.writelines([w + "\n" for w in vocabQst])
    # read files and save in Vocabulary instance
    ansVocab = Vocabulary(ansFile)
    qstVocab = Vocabulary(qstFile)
    return ansVocab, qstVocab

