import nltk
import random

from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


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
        # self.votes = []
        # for c in self._classifiers:
        #     v = c.classify(features)
        #     votes.append(v)
        choice_votes = self.votes.count(mode(self.votes))
        conf = choice_votes / len(self.votes)
        return conf
        
        
documents = []
all_words = []

# Word types: j adjectives, v verbs, r adverbs, n nouns.
allowed_words = ["J"]

print("creating documents")
short_pos = open("C:/Users/parte/pythonapps/twitter_sentiment/dataset/positive.txt", "r").read()
short_neg = open("C:/Users/parte/pythonapps/twitter_sentiment/dataset/negative.txt", "r").read()    

for r in short_pos.split('\n'):
    documents.append((r, "pos"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_words:
            all_words.append(w[0].lower())            

for r in short_neg.split('\n'):
    documents.append((r, "neg"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_words:
            all_words.append(w[0].lower())

print("pickling documents step")
save_documents = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

print("creating freq distribution")
all_words = nltk.FreqDist(all_words)

print("creating word features")
word_features = list(all_words.keys())[:5000]

print("pickling word features")
save_word_feats = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/word_features.pickle", "wb")
pickle.dump(word_features, save_word_feats)
save_word_feats.close()

def find_features(document):
    words = set(word_tokenize(document))
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print("creating features")
featuresets = [(find_features(rev), category) for (rev, category) in documents]

print("shuffle features")
random.shuffle(featuresets)

print("pickle shuffled features")
save_feature_sets = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/featuresets.pickle", "wb")
pickle.dump(featuresets, save_feature_sets)
save_feature_sets.close()

# positive data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

print("training the base NaiveBayes Algorithm")
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("pickle classifier")
save_classifier = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

print("Original Naive Bayes Algorithm accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

print("training Multinomial NB")
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

print("pickle MNB_classifier")
save_MNB_classifier = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, save_MNB_classifier)
save_classifier.close()

print("training Bernoulli")
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

print("pickle BernoulliNB_classifier")
save_BernoulliNB_classifier = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/BernoulliNB_classifier.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_BernoulliNB_classifier)
save_classifier.close()

print("training LogisticRegression")
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

print("pickle LogisticRegression_classifier")
save_LogisticRegression_classifier = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/LogisticRegression_classifier.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_LogisticRegression_classifier)
save_classifier.close()

print("training linearSVC")
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

print("pickle LinearSVC_classifier")
save_LinearSVC_classifier = open("C:/Users/parte/pythonapps/twitter_sentiment/save_data/LinearSVC_classifier.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_LinearSVC_classifier)
save_classifier.close()