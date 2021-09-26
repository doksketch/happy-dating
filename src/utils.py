import re
import typing
import pandas as pd


# очистка текста
def preprocess_text(text):
    text = text.lower()

    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = re.sub(r"[\"#/@;:<>{}`+=~|?,]", "", text)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    return text


# приведение таргета к необходимому виду
def prepare_labels(data):
    datasets_titles = [x.lower() for x in
                       set(data['dataset_title'].unique()).union(set(data['dataset_label'].unique()))]

    labels = []
    for index in data['Id']:
        publication_text = data[data['Id'] == index].text.str.cat(sep='\n').lower()
        label = []
        for dataset_title in datasets_titles:
            if dataset_title in publication_text:
                label.append(preprocess_text(dataset_title))
        labels.append(' | '.join(label))

    return labels


# преобразование таргета к целому числу и обратно
def target_integers(data, target: str):
    targets = list(data[target].unique())
    integers = [i for i in range(len(target))]

    return targets, integers


def encode_decode_target(data, target: str,
                         targets: typing.List[int], integers: typing.List[int],
                         encode: bool):
    if encode:
        encoding = dict(zip(targets, integers))
        data[target] = data[target].map(encoding)
    else:
        decoding = dict(zip(integers, target))
        data[target] = data[target].map(decoding)

    return data
