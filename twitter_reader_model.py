from matplotlib.pyplot import cla
from statistics import mode
import nltk
import random
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        self.votes = []

    def classify(self, features):
        self.votes = []
        for c in self._classifiers:
            v = c.classify(features)
            self.votes.append(v)
        return mode(self.votes)

    def confidence(self, features):
        choice_votes = self.votes.count(mode(self.votes))
        conf = choice_votes / len(self.votes)
        return conf

documents_load = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/documents.pickle", "rb")
documents = pickle.load(documents_load)
documents_load.close()  
        
word_feats_load =open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/word_features.pickle", "rb")
word_features = pickle.load(word_feats_load)
word_feats_load.close()

def find_features(document):
    words = set(word_tokenize(document))
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print("pickling shuffled features")
featuresets_load = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_load)
featuresets_load.close()
  
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

print("loading saved classifier")
classifier_load = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/classifier.pickle", "rb")
classifier = pickle.load(classifier_load)
classifier_load.close()

NMB_classifier_load = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/MNB_classifier.pickle", "rb")
NMB_classifier = pickle.load(NMB_classifier_load)
NMB_classifier_load.close()

BernoulliNB_classifier_load = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_load)
BernoulliNB_classifier_load.close()

LogisticRegression_classifier_load = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_load)
LogisticRegression_classifier_load.close()

LinearSVC_classifier_load = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_load)
LinearSVC_classifier_load.close()

voted_classifier = VoteClassifier(classifier, LinearSVC_classifier, NMB_classifier, BernoulliNB_classifier, LogisticRegression_classifier)

def sentiment(text):
    features = find_features(text)
    return voted_classifier.classify(features), voted_classifier.confidence(features)