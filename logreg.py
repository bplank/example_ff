__author__ = "bplank"

"""
A simple sklearn classifier for animacy classification
"""
import numpy as np
np.random.seed(113) #set seed before any keras import
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report


parser = argparse.ArgumentParser()
parser.add_argument('train', help="animacy data training file")
parser.add_argument('dev', help="animacy data dev file")
parser.add_argument('test', help="animacy data test file")
parser.add_argument('--iters', help="epochs (iterations)", type=int, default=10)
args = parser.parse_args()


def load_data(trainfile, devfile, testfile):
    ### load data
    train_sents, train_y = load_animacy_sentences_and_labels(trainfile)
    dev_sents, dev_y = load_animacy_sentences_and_labels(devfile)
    test_sents, test_y = load_animacy_sentences_and_labels(testfile)

    vectorizer = CountVectorizer(binary=True)
    ### convert training etc data to indices
    X_train = vectorizer.fit_transform(train_sents)
    X_dev = vectorizer.transform(dev_sents)
    X_test = vectorizer.transform(test_sents)


    ### convert labels to one-hot
    label2idx = {label: i for i, label in enumerate(set(train_y+dev_y+test_y))}
    train_y = [label2idx[label] for label in train_y]
    dev_y = [label2idx[label] for label in dev_y]
    test_y = [label2idx[label] for label in test_y]

    return X_train, train_y, X_dev, dev_y, X_test, test_y, label2idx

def load_animacy_sentences_and_labels(datafile):
    """
    loads the data set
    """
    input = [line.strip().split("\t") for line in open(datafile)]
    sentences = [sentence for sentence, label in input]
    labels = [label for sentence, label in input]
    return sentences, labels

## read input data
print("load data..")
X_train, y_train, X_dev, y_dev, X_test, y_test, tag2idx = load_data(args.train, args.dev, args.test)

print("#train instances: {}\n#test instances: {}\n#dev instances: {}".format(X_train.shape[0],X_test.shape[0], X_dev.shape[0]))

model = LogisticRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
print("evaluate model..")
acc = accuracy_score(y_test, y_predicted)
print('Test accuracy:', acc)
target_names = [key for val, key in sorted([(value,key) for key, value in tag2idx.items()])]
print(classification_report(y_test, y_predicted, target_names=target_names))
