from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

class Analyser():
    def __init__(self, x, y, feature_list):
        self.x = x
        self.y = y
        self.feature_list = feature_list
    
    def run_feature_selection(self):
        self._select_k_best()
        self._extra_trees()
        self._get_corr()

    
    def _select_k_best(self):
        best_features = SelectKBest(score_func=f_classif, k=20)
        fit = best_features.fit(self.x, self.y)

        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(self.feature_list)

        feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
        feature_scores.columns = ['Feature Name', 'Score']
        
        feature_scores.nlargest(20, 'Score').plot.barh(x='Feature Name', y='Score', title='Select K Best', figsize=(20,10))
        plt.savefig('./kbest.png')

    def _extra_trees(self):
        model = ExtraTreesClassifier()
        model.fit(self.x, self.y)

        feat_importances = pd.Series(model.feature_importances_, index=self.feature_list)
        
        feat_importances.nlargest(20).plot(kind='barh', title='Extra Tree Classifier', figsize=(20,10))
        plt.savefig('./extra_trees.png')


    def _get_corr(self):
        x_df = pd.DataFrame(self.x, columns=self.feature_list)
        y_df = pd.DataFrame(self.y, columns=['label'])

        df = pd.merge(x_df, y_df, how="outer", left_index=True, right_index=True)
        corrmat = df.corr()

        corr = corrmat['label'].sort_values(ascending=False)
        plt.figure(figsize=(15,15))
        corr.plot(kind='barh', title='Correlation', figsize=(20, 20))
        plt.savefig('./corr.png')

