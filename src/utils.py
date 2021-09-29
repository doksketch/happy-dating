import re
import typing
from torch.utils.data import DataLoader
from src.settings import rnn_params
import torch


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
def target_integers(data):
    targets = list(data.unique())
    integers = [i for i in range(len(targets))]

    return targets, integers


def encode_decode_target(data,
                         targets: typing.List[int], integers: typing.List[int],
                         encode: bool):
    if encode:
        encoding = dict(zip(targets, integers))
        data = data.map(encoding)
    else:
        decoding = dict(zip(integers, targets))
        data = data.map(decoding)

    return data


# сбор векторизированных данных в батч
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device='cpu'):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            drop_last=drop_last, shuffle=shuffle)

    for data_dict in dataloader:
        out_data_dict = dict()

        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)

        yield out_data_dict


# для тренировки нейросети
def get_train_state():
    return dict(early_stop=True,
                learning_rate=rnn_params['learning_rate'],
                epoch_index=0,
                train_loss=list(),
                train_jac=list(),
                valid_loss=list(),
                valid_jacc=list(),
                test_loss=-1,
                test_jac=-1,
                model_filname=rnn_params['model_state_file'])


def update_train_state(model, train_state):
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])

    # сохраняем одну наилучшую модель
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # если лосс ухудшился, обновляем шаг ранней остановки
        if loss_t >= loss_tm1:
            train_state['early_stopping_step'] += 1
        # если лосс улучшился, то сохраняем модель
        else:
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t

            # сбрасываем счётчик шагов ранней остановки
            train_state['early_stopping_step'] = 0

        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= rnn_params['early_stopping_criteria']

        return train_state