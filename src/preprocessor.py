import pandas as pd
from torch.utils.data import Dataset, DataLoader


# представление очищенного датасета в pytorch
class DatasetModel(Dataset):

    def __init__(self, data, vectorizer):
        self.data = data
        self._vectorizer = vectorizer

        self.train_data = self.data[self.data.split == 'train']
        self.train_size = len(self.train_data)

        self.valid_data = self.data[self.data.split == 'valid']
        self.valid_size = len(self.valid_data)

        self.test_data = self.data[self.data.split == 'test']
        self.test_size = len(self.test_data)

        self._lookup_dict = {'train': (self.train_data, self.train_size),
                             'valid': (self.valid_data, self.valid_size),
                             'test': (self.test_data, self.test_size)}

        self.set_split('train')

    # загрузка и векторизация данных на основе класса DataVectorizer
    @classmethod
    def load_and_vectorize(cls, path):
        data = pd.read_csv(path)
        return cls(data, DataVector.from_dataframe(data))

    def get_vectorizer(self):
        return self._vectorizer

    # выбор фрагмента данных по колонке
    def set_split(self, split='train'):
        self._target_split = split
        self._target_data, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    # точка входа для данных в pytorch
    def __getitem__(self, index):
        "index - индекс точки данных"
        row = self._target_data.iloc[index]
        data_vector = self._vectorizer.vectorize(row.data)
        target_index = self._vectorizer.target_vocab.lookup_token(row.target)

        return {'x_data': data_vector,
                'y_target': target_index}


# отображение токенов в числовую форму - технические словари
class VectorDictionaries:

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

    # экземпляр класса на основе словаря
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

    # извлекает соответствующий токену индекс
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


# преобразование текста в векторы на основе словарей класса VectorDictionaries
class DataVector:

    def __init__(self, data_vocab, target_vocab):
        self.data_vocab = data_vocab
        self.target_vocab = target_vocab

    # задаётся форма вектора для обзора
    def vectorize(self, data):
        pass

    @classmethod
    def from_dataframe(cls, data):
        data_vocab = VectorDictionaries(add_unk=True)
        target_vocab = VectorDictionaries(add_unk=False)

        return cls(data_vocab, target_vocab)

    @classmethod
    def from_serializable(cls, contents):
        data_vocab = VectorDictionaries.from_serializable(contents['data_vocab'])
        target_vocab = VectorDictionaries.from_serializable(contents['target_vocab'])
        return cls(data_vocab, target_vocab)

    def to_serializable(self):
        return {'data_vocab': self.data_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}


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