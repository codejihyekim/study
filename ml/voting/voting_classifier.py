import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Solution:
    def __init__(self):
        self.cancer = load_breast_cancer()

    def test(self):
        cancer = self.cancer
        data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        print(data_df.head(3))

if __name__ == '__main__':
    s = Solution()
    s.test()
