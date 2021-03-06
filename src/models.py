import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import torch
import torch.nn.functional


class GramsClassifier:

    def __init__(self, model):
        self.model = model

        self.pipeline = Pipeline(steps=[
            ('vectorizer', CountVectorizer(ngram_range=(2, 2), max_features=50)),  # строим биграммы
            ('tf-idf', TfidfTransformer()),
            ('model', self.model)
        ])

    def fit(self, X: list, y: list):
        logging.info('Fit model')
        self.pipeline.fit(X, y)

    def predict(self, X: list):
        predictions = self.pipeline.predict_proba(X)

        return predictions


# RNN (в этом случае GRU, но можно и другую)
class ContentClassifier(torch.nn.Module):

    def __init__(self, embedding_size: int, num_embeddings: int, num_classes: int,
                 rnn_hidden_size: int, batch_first=True, padding_idx=0):
        """
        Аргументы:
            embedding_size - размер вложений символов
            num_embeddings - количество символов для создания вложений
            num_classes - размер целевого вектора (общее количество классов)
            rnn_hidden_size - размер скрытого слоя состояния
            batch_first - будут ли в нулевом измерении входных тензоров находиться данные пакета или последовательности
            padding_idx - индекс для дополнения нулями тензора
        """

        super(ContentClassifier, self).__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size,
                                             padding_idx=padding_idx)  # слой вложений
        self.rnn = torch.nn.GRU(input_size=embedding_size, hidden_size=rnn_hidden_size,
                                batch_first=batch_first)  # реккурентная нейросеть - получаем скрытые состояния
        self.fc1 = torch.nn.Linear(in_features=rnn_hidden_size,
                                   out_features=rnn_hidden_size)  # сводный выходной вектор
        self.fc2 = torch.nn.Linear(in_features=rnn_hidden_size,
                                   out_features=num_classes)  # вектор предсказаний

    # прямой проход классификатора
    def forward(self, x_in, x_lenghts=None, apply_softmax=True):
        """
        Аргументы:
            x_in (torch.Tensor) - тензор входных данных размерностью (batch, input_dim)
            x_lenghts (torch.Tensor) - длины всех последовательностей батча, используемые для
                поиска завершающих векторов последовательностей
            apply_softmax: bool - применяем или нет многомерную логистическую функцию активации
        """

        x_embedded = self.embeddings(x_in)
        y_out = self.rnn(x_embedded)

        if x_lenghts is not None:
            y_out = self._extract_final_vector(y_out, x_lenghts)
        else:
            y_out = y_out[:, -1, :]

        y_out = torch.nn.functional.dropout(y_out, p=0.5)
        y_out = torch.nn.functional.relu(self.fc1(y_out))
        y_out = torch.nn.functional.dropout(y_out, p=0.5)
        y_out = self.fc2(y_out)

        if apply_softmax:
            y_out = torch.nn.functional.softmax(y_out, dim=1)

        return y_out

    @classmethod
    def _extract_final_vector(cls, y_out, x_lenghts):
        """
        Извлекает завершающие вектора в каждой последовательности. Вектор находится на позиции, соотвествующей длине
        последовательности
        Аргументы:
            x_in (torch.Tensor) - тензор входных данных размерностью (batch, input_dim)
            x_lenghts (torch.Tensor) - длины всех последовательностей батча, используемые для
                        поиска завершающих векторов последовательностей
            apply_softmax: bool - применяем или нет многомерную логистическую функцию активации
        """

        x_lenghts = x_lenghts.long().detach().cpu().numpy() - 1

        out = list()
        for batch_index, column_index in enumerate(x_lenghts):
            out.append(y_out[batch_index, column_index])

        return torch.stack(out)
