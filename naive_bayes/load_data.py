import pandas as pd
import os


def get_file_content(file):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content.lower()


def load_mails_dataset(dir):
    """Loads ham/spam emails dataset and returns X and y"""
    X = []
    y = []
    hams = os.listdir(os.path.join(dir, 'ham'))
    spams = os.listdir(os.path.join(dir, 'spam'))

    for f in hams:
        X.append(get_file_content(os.path.join(dir, 'ham', f)))
        y.append(0)
    for f in spams:
        X.append(get_file_content(os.path.join(dir, 'spam', f)))
        y.append(1)
    return X, y


def load_csv(file):
    ...


def load_weibo_dataset(dir):
    ...
