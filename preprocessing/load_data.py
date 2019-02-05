import os

import pandas as pd
import requests


def load_sts_dataset(path):
    """
     Loads a subset of the STS dataset into a DataFrame.
     In particular both sentences and their human rated similarity score.
    :param filename:
    :return:
    """

    return pd.read_csv(path, sep='\t', header=0)


def download_and_load_sts_data():
    sts_dev = load_sts_dataset(os.path.join("data", "sts", "english", "sts_dev.csv"))
    sts_test = load_sts_dataset(os.path.join("data", "sts", "english", "sts_dev.csv"))
    sts_train = load_sts_dataset(os.path.join("data", "sts", "english", "sts_dev.csv"))

    return sts_dev, sts_test, sts_train


def download_sick_dataset(url):
    response = requests.get(url).text

    lines = response.split("\n")[1:]
    lines = [l.split("\t") for l in lines if len(l) > 0]
    lines = [l for l in lines if len(l) == 5]

    df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
    df['sim'] = pd.to_numeric(df['sim'])
    return df


def download_and_load_sick_dataset():
    sick_train = download_sick_dataset(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt")
    sick_dev = download_sick_dataset(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt")
    sick_test = download_sick_dataset(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt")
    sick_all = sick_train.append(sick_test).append(sick_dev)

    return sick_all, sick_train, sick_test, sick_dev
