from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

from naive_bayes import NaiveBayes
from load_data import load_mails_dataset


def main():
    corpus, y = load_mails_dataset('./mails')
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.33)

    clf = MultinomialNB()
    # clf = NaiveBayes()
    clf.fit(X_train, y_train)
    print('==> train acc: ', clf.score(X_train, y_train))
    print('==> test acc: ', clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    target_names = ['ham', 'spam']
    print('\n', classification_report(y_test, y_pred, target_names=target_names))

    # print topk frequent tokens in spam letters
    tokens = np.asanyarray(vectorizer.get_feature_names())
    token_vector = np.eye(tokens.shape[0])
    topk_index = np.argsort(clf.predict_proba(token_vector)[:, 1])[:10]
    print(tokens[topk_index])


if __name__ == '__main__':
    main()
