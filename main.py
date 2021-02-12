# Natural Language Processing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# cleaning section
import re # Regular expression operations
import nltk # to make machines understand human language and reply to it with an appropriate response

nltk.download('stopwords') # Removing stop words with NLTK like {‘ourselves’, ‘hers’, ‘between’, ‘yourself’, ‘but’, ‘again’,..}
from nltk.corpus import stopwords # for stopwords
from nltk.stem.porter import PorterStemmer
# stemming the reviews: 'I loved it' -> love == 'I love it' (all conjugations become a present tense)
# without stemming loved and love are different words and it becomes higher dimensions = more complex

# for bag of words
from sklearn.feature_extraction.text import CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# for splitting dataset
from sklearn.model_selection import train_test_split

# for naive bays model
from sklearn.naive_bayes import GaussianNB

# for kernel svm
from sklearn.svm import SVC

# confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix


def nlp():
    # import dataset
    dataset = pd.read_csv('Restaurant_Reviews.tsv', sep='\t', quoting=3)
    # reading the files tab separate value (tsv) -> sep='\t' or delimiter='\t'
    # always rid get of quote from the text because it might be errors
    '''
    quoting = 0, 1, 2, and 3 for QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONE, and QUOTE_NONNUMERIC, respectively.
    '''
    # print(dataset)

    # cleaning the texts
    # cleaning all the rows = removing unnecessary words line by line like 'ourselves’, ‘hers’, ‘between’, ‘yourself’, ‘but’, ‘again’,..
    corpus = [] # this will being received only cleaned words
    for i in range (len(dataset)):

        # cleaning '' , : in each sentence replaces to space ' '
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # re.sub('[not replace these ones]', others' replaced by user said, where repalce the value)
        review = review.lower() # store non comma sentences as lower cases
        review = review.split() # for stemming

        # cleaning with stemming
        ps = PorterStemmer()

        # stopwords include 'not' -> 'not good' becomes 'good'
        all_words = stopwords.words('english')
        all_words.remove('not') # removed 'not' from the stopwords list

        review = [ps.stem(word) for word in review if not word in set(all_words)]
        # when the sentence has the stopword where I set in the loop, the loop stop and goes before the words into ps.stem()
        # ex) 'I was loved but...' -> 'but' is stopwords -> ps.steam('[I], [am], [love ]')

        review = ' '.join(review) # join the each words with 'space'

        # adding the selected words into corpus
        corpus.append(review) # end of the loop

    # print(corpus)

    # creating bag of the words
    vectorizer = CountVectorizer() # we should decide max words as some words like 'steve' are not useful for decision making, but we don't know the specific value yet
    x = vectorizer.fit_transform(corpus).toarray() # x must be toArray() for training
    max_words = int(len(x[0])*0.9) # assume 10 % of the words in the reviews are useless

    # re-train the dataset with max_words: if the correctness is less than 80%, change the int(len(x[0])*0.9)  or change the models
    vectorizer = CountVectorizer(max_features=max_words)  # now, we know the value
    x = vectorizer.fit_transform(corpus).toarray()

    # splitting dataset into 4 parts
    x_train, x_test, y_train, y_test = train_test_split(x, dataset['Liked'], train_size=0.8, random_state=1)

    # naive bays
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    # svm
    sv = SVC(kernel='rbf', random_state=0)
    sv.fit(x_train, y_train)

    # print(y_pred)

    # confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=classifier.predict(x_test))
    print(cm)

    print('\nThe correctness with Bays is %.2f percent\n' % accuracy_score(y_true=y_test, y_pred=classifier.predict(x_test)))

    cm = confusion_matrix(y_true=y_test, y_pred=sv.predict(x_test))
    print(cm)

    print('\nThe correctness with SVM is %.2f percent\n' % accuracy_score(y_true=y_test, y_pred=sv.predict(x_test)))

    bays = accuracy_score(y_true=y_test, y_pred=classifier.predict(x_test))
    svm = accuracy_score(y_true=y_test, y_pred=sv.predict(x_test))

    if (bays > svm):
        print('Bays is better than SVM\n')
    else:
        print('SVM is better than Bays\n')

    # predict single sentence
    print('Expect one sentence is positive or negative review\n')
    sentence = input('please enter one sentence, here:')

    new_review = re.sub('[^a-zA-Z]', ' ', sentence)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = vectorizer.transform(new_corpus).toarray()

    if (bays > svm):
        new_y_pred = classifier.predict(new_X_test)
    else:
        new_y_pred = sv.predict(new_X_test)

    if(int(new_y_pred) == 1):
        print('You gave a positive review, Thank you.')
    else:
        print('You gave a negative review, how can I improve?')

if __name__ == '__main__':
    nlp()