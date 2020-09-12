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
# 2-class or 6-class
X, Y = read_corpus('trainset.txt', use_sentiment=False)
# seprating the corpus into 75% for train and 25% for test
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

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
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )


# COMMENT THIS
# training the model
classifier.fit(Xtrain, Ytrain)

# COMMENT THIS
# getting the predicted labels for test data from the model  
# Yguess = classifier.predict(Xtest)

# COMMENT THIS
# printing the accuracy_score result for measuring the model performance
# print(accuracy_score(Ytest, Yguess))


# print(classification_report(Ytest, Yguess))
# print(confusion_matrix(Ytest, Yguess))
import matplotlib.pyplot as plt
plot_confusion_matrix(classifier, Xtest, Ytest)
plt.show()

report_2_class = '''precision    recall  f1-score   support

         neg       0.71      0.93      0.81       731
         pos       0.91      0.64      0.75       769

    accuracy                           0.78      1500
   macro avg       0.81      0.79      0.78      1500
weighted avg       0.81      0.78      0.78      1500'''

report_6_class = '''precision    recall  f1-score   support

       books       0.94      0.91      0.93       233
      camera       0.83      0.94      0.88       258
         dvd       0.88      0.91      0.89       242
      health       0.97      0.79      0.87       243
       music       0.96      0.95      0.95       260
    software       0.89      0.93      0.91       264

    accuracy                           0.91      1500
   macro avg       0.91      0.91      0.91      1500
weighted avg       0.91      0.91      0.91      1500'''
