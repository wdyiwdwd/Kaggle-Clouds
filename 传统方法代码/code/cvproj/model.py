from abc import abstractmethod
import numpy as np

# import joblib
# from sklearn.naive_bayes import GaussianNB
import torch
# from sklearn.svm import SVC

from .slicer import sizes
from .utils import path_join, mkdir_if_not_exist
from .features import feature_vector_size
from .config import label_cnt

model_index = {size: i for i, size in sizes}


class Model(object):
    def __init__(self, classes):
        self.classes = classes

    @abstractmethod
    def fit(self, X, y) -> bool:
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @classmethod
    def load(cls, filename, classes):
        return cls.load(filename, classes)

    @classmethod
    def need_loop(cls):
        return cls.need_loop()

    def get_last_loss(self):
        return None

    @abstractmethod
    def predict(self, X):
        pass

    def predict_one(self, x):
        return self.predict(x[np.newaxis, :])[0]

    @abstractmethod
    def predict_score(self, X):
        pass

    def predict_one_score(self, x):
        return self.predict_score(x[np.newaxis, :])[0, :]

    def set_train(self):
        pass

    def set_eval(self):
        pass


# class NaiveBayesian(Model):
#     def __init__(self, classes):
#         super(NaiveBayesian, self).__init__(classes)
#         self.__model = GaussianNB()
#
#     def fit(self, X, y):
#         self.__model.partial_fit(X, y, self.classes)
#         return False
#
#     def save(self, filename):
#         joblib.dump(self.__model, filename)
#
#     @classmethod
#     def load(cls, filename, classes):
#         instance = NaiveBayesian(classes)
#         instance.__model = joblib.load(filename)
#         return instance
#
#     @classmethod
#     def need_loop(cls):
#         return False
#
#     def predict(self, X):
#         return self.__model.predict(X)
#
#     def predict_score(self, X):
#         return self.__model.predict_log_proba(X)


hidden_layer_size = feature_vector_size  # int((feature_vector_size * 5) ** 0.5)


class SingleHiddenLayerNN(Model):
    def __init__(self, classes, weight):
        super(SingleHiddenLayerNN, self).__init__(classes)
        self.__model = torch.nn.Sequential(
            torch.nn.Linear(feature_vector_size, hidden_layer_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_layer_size, label_cnt),
            # torch.nn.Sigmoid(),
            torch.nn.LogSoftmax(dim=1)
        )
        for module in self.__model.children():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight.data, 1)
        self.__optimizer = torch.optim.SGD(self.__model.parameters(), lr=0.1)
        self.__optimizer.zero_grad()
        if isinstance(weight, np.ndarray):
            self.__weight = torch.from_numpy(weight.astype(np.float32))
        else:
            self.__weight = weight
        self.__loss = torch.nn.NLLLoss(reduction='mean', weight=self.__weight)
        self.__last_loss = np.inf
        # cnt = len(self.classes)
        # self.__one_hot = np.eye(cnt, dtype=np.int_)  # [:, 1:]

    def fit(self, X, y):
        y_hat = self.__model(torch.from_numpy(X.astype(np.float32)))  # [:,1:]
        # y_real = torch.from_numpy(self.__one_hot[y, :])
        y_real = torch.from_numpy(y.astype(np.int64))
        loss = self.__loss(y_hat, y_real)
        loss_ = loss.detach().numpy()
        loss.backward()
        self.__optimizer.step()
        self.__optimizer.zero_grad()
        if abs(loss_ - self.__last_loss) < 1e-8:  # converge
            return True
        else:
            self.__last_loss = loss_
            return False

    def save(self, filename):
        torch.save(self.__model.state_dict(), filename)
        torch.save(self.__weight, filename + '.sample_weights')

    @classmethod
    def load(cls, filename, classes):
        weights = torch.load(filename + '.sample_weights')
        instance = SingleHiddenLayerNN(classes, weights)
        instance.__model.load_state_dict(torch.load(filename))
        return instance

    @classmethod
    def need_loop(cls):
        return True

    def get_last_loss(self):
        return self.__last_loss

    def predict(self, X):
        return self.predict_score(X).argmax(axis=1)

    def predict_score(self, X):
        return torch.exp(self.__model(torch.from_numpy(X.astype(np.float32)))).detach().numpy()

    def set_train(self):
        self.__model.train()

    def set_eval(self):
        self.__model.eval()


# class RbfSVM(Model): # 数据量过大无法使用
#     def __init__(self, classes):
#         super(RbfSVM, self).__init__(classes)
#         self.__model = SVC(max_iter=1000)
#         self.__one_hot = np.eye(5)
#
#     @classmethod
#     def need_loop(cls):
#         return False
#
#     def fit(self, X, y) -> bool:
#         self.__model.fit(X, y)
#         return False
#
#     @classmethod
#     def load(cls, filename, classes):
#         instance = RbfSVM(classes)
#         instance.__model = joblib.load(filename)
#         return instance
#
#     def save(self, filename):
#         joblib.dump(self.__model, filename)
#
#     def predict(self, X):
#         return self.__model.predict(X)
#
#     def predict_score(self, X):
#         return self.__one_hot[self.predict(X), :]


def save(models, dirname):
    mkdir_if_not_exist(dirname)
    for model, (step, size) in zip(models, sizes):
        model.save(path_join(dirname, str(size) + '.model'))


def load(type_, dirname):
    set_type(type_)
    classes = [i for i in range(label_cnt)]
    models = [model_type.load(path_join(dirname, str(size) + '.model'), classes) for _, size in sizes]
    return models


model_type = SingleHiddenLayerNN
model_types = {
    # 'naive_bayesian': NaiveBayesian,
    'single_hidden_layer': SingleHiddenLayerNN,
    'shl': SingleHiddenLayerNN,
    # 'svm': RbfSVM,
}


def set_type(type_):
    global model_type
    if type_ is None or type_ not in model_types:
        model_type = SingleHiddenLayerNN
    else:
        model_type = model_types[type_]


def make_models(weights_for_size):
    classes = [i for i in range(label_cnt)]
    return [model_type(classes, weight) for weight in weights_for_size]


def set_train(models):
    for model in models:
        model.set_train()


def set_eval(models):
    for model in models:
        model.set_eval()


def need_loop():
    return model_type.need_loop()
