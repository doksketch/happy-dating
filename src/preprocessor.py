import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import tensor, float32
import json
from collections import defaultdict


# представление очищенного датасета в pytorch
class DatasetModel(Dataset):

    def __init__(self, df, vectorizer):
        self.df = df
        self._vectorizer = vectorizer

        self._max_seq_length = max(map(len, self.df.predictor)) + 2

        self.train_df = self.df[self.df.split == 'train']
        self.train_size = len(self.train_df)

        self.valid_df = self.df[self.df.split == 'valid']
        self.valid_size = len(self.valid_df)

        self.test_df = self.df[self.df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'valid': (self.valid_df, self.valid_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # веса для классов
        class_counts = self.train_df.target.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.target_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequences = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / tensor(frequences, dtype=float32)

    # загружает данные и создаёт векторизатор
    @classmethod
    def make_vectorizer(cls, path: str):
        df = pd.read_csv(path)
        train_df = df[df.split == 'train']
        return cls(df, PredictorVectorizer.from_dataframe(train_df))

    def get_vectorizer(self):
        return self._vectorizer()

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def set_split(self, split='train'):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    # точка входа для данных в pytorch
    def __getitem__(self, index):
        "index - индекс точки данных"
        row = self._target_df.iloc[index]
        predictor_vector, vec_length = self._vectorizer.vectorize(row.predictor, self._max_seq_length)
        target_index = self._vectorizer.target_vocab.lookup_token(row.target)

        return {'x_data': predictor_vector,
                'y_target': target_index,
                'x_length': vec_length}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


# векторизатор, приводящий словари в соотвествие друг другу и использующий их
class PredictorVectorizer:

    def __init__(self, char_vocab, target_vocab):
        """
        Аргументы:
            char_vocab(Vocabulary) - последовательности в словари
            target_vocab - таргет(категория) в словари
        """
        self.char_vocab = char_vocab
        self.target_vocab = target_vocab

    def vectorize(self, predictor, vector_length=-1):
        """
        Аргументы:
            predictor - размер вложений символов
            vector_length - длина вектора индексов
        """

        indices = [self.char_vocab.begin_seq_index]
        indices.extend(self.char_vocab.lookup_token(token)
                       for token in predictor)
        indices.append(self.char_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.char_vocab.mask_index

        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        char_vocab = SequenceVocabulary()
        target_vocab = Vocabulary()

        for index, row in df.iterrows():
            tokens = row.predictor.split(' ')
            for token in tokens:
                char_vocab.add_token(token)
            target_vocab.add_token(row.target)

        return cls(char_vocab, target_vocab)

    @classmethod
    def from_serializable(cls, contents):
        char_vocab = SequenceVocabulary.from_serializable(contents['char_vocab'])
        target_vocab = Vocabulary.from_serializable(contents['target_vocab'])

        return cls(char_vocab=char_vocab, target_vocab=target_vocab)

    def to_serializable(self):
        return {'char_vocab': self.char_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}


# отображение токенов в числовую форму - технические словари
class Vocabulary:
    """
    Аргументы:
        token_to_idx: dict - соотвествие токенов индексам
        add_unk: bool - нужно ли добавлять токен UNK
        unk_token - добавляемый в словарь токен UNK
    """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token='<UNK>'):
        if token_to_idx is None:
            token_to_idx = dict()
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    # сериализуемый словарь
    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    # экземпляр класса на основе сериализованного словаря
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    # обновляет словари отображения - если токен не найден, то добавляет в словарь
    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index

    # извлекает соответствующий токену индекс или индекс UNK, если токен не найден
    def lookup_token(self, token):
        if self._add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    # возвращает соотвествующий индексу токен
    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError('Индекс (%d) не в словаре' % index)
        return self._idx_to_token[index]

    def __str__(self):
        return '<Словарь (size=%d)>' % len(self)

    def __len__(self):
        return len(self._token_to_idx)


# токенизация последовательностей
class SequenceVocabulary(Vocabulary):

    def __init__(self, token_to_idx=None, unk_token='<UNK>',
                 mask_token="<MASK>", begin_seq_token='<BEGIN>',
                 end_seq_token='<END>'):

        super(SequenceVocabulary, self).__init__(token_to_idx)
        self._mask_token = mask_token  # для работы с последовательностями переменной длины
        self._unk_token = unk_token  # для обозначения отсуствующих токенов в словаре
        self._begin_seq_token = begin_seq_token  # начало предложения
        self._end_seq_token = end_seq_token  # конец предложения

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class TrainValidSplit:

    def __init__(self,
                 train_proportion: float,
                 valid_proportion: float,
                 test_proportion: float,
                 raw_df_path: str,
                 seed: int):
        self.by_target = self.get_target_dict(raw_df_path)
        self.final_list = self.make_split(self.by_target, train_proportion, valid_proportion,
                                          test_proportion, seed)

    @staticmethod
    def get_target_dict(raw_df_path):
        df = pd.read_csv(raw_df_path)

        by_target = defaultdict(list)
        for _, row in df.iterrows():
            by_target[row.target].append(row.to_dict())

        return by_target

    @staticmethod
    def make_split(by_target, train_proportion, valid_proportion, test_proportion, seed):
        final_list = []
        np.random.seed(seed)

        for _, item_list in sorted(by_target.items()):
            np.random.shuffle(item_list)
            n = len(item_list)
            n_train = int(train_proportion * n)
            n_valid = int(valid_proportion * n)
            n_test = int(test_proportion * n)

            for item in item_list[:n_train]:
                item['split'] = 'train'
            for item in item_list[n_train:n_train + n_valid]:
                item['split'] = 'valid'
            for item in item_list[n_train + n_valid:]:
                item['split'] = 'test'

            final_list.extend(item_list)

        return final_list

    def save_prepared_data(self, prepared_df_path):
        prepared_data = pd.DataFrame(self.final_list)
        return prepared_data.to_csv(prepared_df_path, index=False, encoding='utf-8')