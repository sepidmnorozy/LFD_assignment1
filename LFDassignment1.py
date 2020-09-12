from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score as f_score
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report


# COMMENT THIS
# for reading the corpus and getting documents by using the strip and split methods
# isolating the labels from the text
# use_sentiment = 1 -> 2-class classification
# use_sentiment = 0 -> 6-class classification
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

# COMMENT THIS
# X = documents, Y = labels
# 2-class observation
X_2, Y_2 = read_corpus('trainset.txt', use_sentiment=True)
# separating the corpus into 75% for train and 25% for test
split_point = int(0.75*len(X_2))
Xtrain_2 = X_2[:split_point]
Ytrain_2 = Y_2[:split_point]
Xtest_2 = X_2[split_point:]
Ytest_2 = Y_2[split_point:]

# 6-class observation
X_6, Y_6 = read_corpus('trainset.txt', use_sentiment=False)
# separating the corpus into 75% for train and 25% for test
Xtrain_6 = X_6[:split_point]
Ytrain_6 = Y_6[:split_point]
Xtest_6 = X_6[split_point:]
Ytest_6 = Y_6[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)

# combine the vectorizer with a Naive Bayes classifier
classifier_2 = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )
classifier_6 = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )


# COMMENT THIS
# training the model
classifier_2.fit(Xtrain_2, Ytrain_2)
classifier_6.fit(Xtrain_6, Ytrain_6)


# COMMENT THIS
# getting the predicted labels for test data from the model  
Yguess_2 = classifier_2.predict(Xtest_2)
Yguess_6 = classifier_6.predict(Xtest_6)

# COMMENT THIS
# printing the accuracy_score result for measuring the model performance
print("accuracy_score for 2-class")
print(accuracy_score(Ytest_2, Yguess_2))
print("accuracy_score for 6-class")
print(accuracy_score(Ytest_6, Yguess_6))

#1.3
print("fscore, recall, precision for 2-class")
print(classification_report(Ytest_2, Yguess_2))
print("fscore, recall, precision for 6-class")
print(classification_report(Ytest_6, Yguess_6))

print("confusion_matrix for 2-class")
print(confusion_matrix(Ytest_2, Yguess_2))
print("confusion_matrix for 6-class")
print(confusion_matrix(Ytest_6, Yguess_6))

import matplotlib.pyplot as plt
plot_confusion_matrix(classifier_2, Xtest_2, Ytest_2)
plot_confusion_matrix(classifier_6, Xtest_6, Ytest_6)
plt.show()
#1.4
print("probabilities for 2_class")
print(classifier_2.predict_proba(Xtest_2))
print("probabilities for 6_class")
print(classifier_6.predict_proba(Xtest_6))

