from sklearn.model_selection import KFold
from sklearn import svm

from . import utils
from .constants import MIDIMIDI_FEATURE_KEYS

class Runner():
    def __init__(self, total_data, train_data, test_data):
        self.total_data = total_data
        self.train_data = train_data
        self.test_data = test_data

        self.total_X, self.total_Y, _ = utils.make_X_Y(self.total_data)
        self.train_X, self.train_Y, _ = utils.make_X_Y(self.train_data)
        self.test_X, self.test_Y, _ = utils.make_X_Y(self.test_data)

    def run_svm(self):
        


    
    