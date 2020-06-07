import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from logistic import LogisticRegression


def load_income(file):
    df = pd.read_csv(file, sep=',', header=None)
    df = (df - df.min()) / (df.max() - df.min())        # normalization
    return (np.asanyarray(df.iloc[:3000, 1:-1]), np.asanyarray(df.iloc[:3000, -1]),
            np.asanyarray(df.iloc[-1000:, 1:-1]), np.asanyarray(df.iloc[-1000:, -1]))


def main():
    X_train, y_train, X_test, y_test = load_income('./income.csv')

    lr = LogisticRegression(C=1000, lr_decay='step').fit(X_train, y_train)

    # lr.score(X_train, y_train)
    # lr.score(X_test, y_test)
    y_pred = lr.predict(X_train)
    print('\n==> train:\n', classification_report(y_train, y_pred))
    y_pred = lr.predict(X_test)
    print('\n==> test:\n', classification_report(y_test, y_pred))

    # uncomment to plot the result of different learning rate decay strategies

    # fig, ax = plt.subplots(1, 2)
    # ax[0].set_xlabel('epoch')
    # ax[0].set_ylabel('acc')
    # ax[0].set_title('acc of epoch')
    # ax[1].set_xlabel('epoch')
    # ax[1].set_ylabel('lr')
    # ax[1].set_title('decayed lr of epoch')

    # lr = LogisticRegression(eta=0.001).fit(X_train, y_train)
    # ax[0].plot(lr.epoch_, lr.acc_, label='no-decay')
    # ax[1].plot(lr.epoch_, lr.lr_, label='no-decay')

    # lr = LogisticRegression(eta=0.01, lr_decay='step').fit(X_train, y_train)
    # ax[0].plot(lr.epoch_, lr.acc_, label='step')
    # ax[1].plot(lr.epoch_, lr.lr_, label='step')

    # lr = LogisticRegression(eta=0.01, lr_decay='exp').fit(X_train, y_train)
    # ax[0].plot(lr.epoch_, lr.acc_, label='exp')
    # ax[1].plot(lr.epoch_, lr.lr_, label='exp')

    # ax[0].legend()
    # ax[1].legend()
    # plt.show()


if __name__ == '__main__':
    main()
