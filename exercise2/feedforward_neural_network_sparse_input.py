__author__ = "bplank"

"""
A first version of a simple feedforward NN for animacy classification
"""
import numpy as np
np.random.seed(113) #set seed before any keras import
import argparse
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.metrics import accuracy_score, classification_report, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('train', help="animacy data training file")
parser.add_argument('dev', help="animacy data dev file")
parser.add_argument('test', help="animacy data test file")
parser.add_argument('--iters', help="epochs (iterations)", type=int, default=10)
args = parser.parse_args()


def get_index(word, word2idx, freeze=False):
    """
    map words to indices
    keep special OOV at position 0
    """
    if word in word2idx:
        return word2idx[word]
    else:
        if not freeze:
            word2idx[word]=len(word2idx) #new index
            return word2idx[word]
        else:
            return word2idx["_UNK"]


def convert_to_n_hot(X, vocab_size):
    out = []
    for instance in X:
        n_hot = np.zeros(vocab_size)
        for w_idx in instance:
            n_hot[w_idx] = 1   # Q1: What happens here?
        out.append(n_hot)
    return np.array(out)

def load_data(trainfile, devfile, testfile):
    ### load data
    train_sents, train_y = load_animacy_sentences_and_labels(trainfile)
    dev_sents, dev_y = load_animacy_sentences_and_labels(devfile)
    test_sents, test_y = load_animacy_sentences_and_labels(testfile)

    ### create mapping word to indices
    word2idx = {"_UNK": 0}  # reserve 0 for OOV

    ### convert training etc data to indices
    X_train = [[get_index(w,word2idx) for w in x] for x in train_sents]
    freeze=True
    X_dev = [[get_index(w,word2idx,freeze) for w in x] for x in dev_sents]
    X_test = [[get_index(w,word2idx,freeze) for w in x] for x in test_sents]

    vocab_size = len(word2idx)
    X_train = convert_to_n_hot(X_train, vocab_size)
    X_dev = convert_to_n_hot(X_dev, vocab_size)
    X_test = convert_to_n_hot(X_test, vocab_size)

    ### convert labels to one-hot
    label2idx = {label: i for i, label in enumerate(set(train_y+dev_y))}
    num_labels = len(label2idx.keys())
    train_y = np_utils.to_categorical([label2idx[label] for label in train_y], num_classes=num_labels)
    dev_y = np_utils.to_categorical([label2idx[label] for label in dev_y], num_classes=num_labels)
    test_y = np_utils.to_categorical([label2idx[label] for label in test_y], num_classes=num_labels)

    return X_train, train_y, X_dev, dev_y, X_test, test_y, word2idx, label2idx

def load_animacy_sentences_and_labels(datafile):
    """
    loads the data set
    """
    input = [line.strip().split("\t") for line in open(datafile)]
    sentences = [sentence.split() for sentence, label in input]
    labels = [label for sentence, label in input]
    return sentences, labels

## read input data
print("load data..")
X_train, y_train, X_dev, y_dev, X_test, y_test, word2idx, tag2idx = load_data(args.train, args.dev, args.test)

print("#train instances: {}\n#test instances: {}\n#dev instances: {}".format(len(X_train),len(X_test), len(X_dev)))
assert(len(X_train)==len(y_train))
assert(len(X_test)==len(y_test))
assert(len(X_dev)==len(y_dev))

vocabulary_size=len(word2idx.keys())
num_labels = len(tag2idx)
input_size = len(X_train[0])
print(input_size)
print("build model")
model = Sequential()
### simplest neural network with sparse input
### Q2: your code here, create a simple feedforward neural network



## end your code
print("train model..")
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=args.iters,
          batch_size=100, validation_data=(X_dev, y_dev))


score = model.evaluate(X_test, y_test)
print("evaluate model..")
score, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

## lets get predictions (evaluate just gets us the scores)

## Q3: get the actual predictions (argmax per class) and use classification report from sklearn
