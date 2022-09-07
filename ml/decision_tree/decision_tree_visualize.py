import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import export_graphviz
import graphviz

class Solution:
    def __init__(self):
        pass

    def test(self):
        # DecisionTreeClassifier 생성
        dt_clf = DecisionTreeClassifier(random_state=156)

        # 붗꽃 데이터를 로딩하고, 학습과 테스트 데이터 세트로 분리
        iris_data = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11)

        # DecisionTreeClassifier 학습
        dt_clf.fit(x_train, y_train)

        #export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함
        #export_graphviz(dt_clf, out_file='./save/tree.dot', class_names=iris_data.target_names, feature_names=iris_data.feature_names, impurity=True, filled=True)

        # 위에서 생성된 tree.dot 파일을 Graphviz가 읽어서 시각화
        with open("./save/Source.gv") as f:
            dot_graph = f.read()
        #graph = graphviz.Source(dot_graph)
        #graph.render(view=True)

        # feature별 importance 추출
        print("Feature importances: \n{0}".format(np.round(dt_clf.feature_importances_, 3)))

        #feature별 inportance 매핑
        for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
            print('{0} : {1: .3f}'.format(name, value))

        #feature importance를 columns 별로 시각화하기
        sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)
        plt.show()
        #plt.savefig('./save/feature_importance.png')

    def overfiting(self):
        plt.title('3 Class values with 2 Feature Sample data creation')

        # 2차원 시각화를 위해서 피처는 2개, 클래스는 3가지 유형의 분류 샘픔 데이터 생성
        x_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1, random_state=0)
        #print(x_features[:, 0])

        # 그래프 형태로 2개의 피처로 2차원 좌표 시각화, 각 클래스 값은 다른 색깔로 표시
        plt.scatter(x_features[:, 0], x_features[:, 1], marker='o', c=y_labels, s=25, edgecolors='k')
        #plt.show()
        #plt.savefig('./save/test_data.png')

        #dt_clf = DecisionTreeClassifier(random_state=156).fit(x_features, y_labels)
        dt_clf = DecisionTreeClassifier(min_samples_leaf=6, random_state=156).fit(x_features, y_labels)
        self.visualize_boundary(dt_clf, x_features, y_labels)

    def visualize_boundary(self, model, X, y):
        fig, ax = plt.subplots()

        # 학습 데이타 scatter plot으로 나타내기
        ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
                   clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')
        ax.axis('off')
        xlim_start, xlim_end = ax.get_xlim()
        ylim_start, ylim_end = ax.get_ylim()

        # 호출 파라미터로 들어온 training 데이타로 model 학습 .
        model.fit(X, y)
        # meshgrid 형태인 모든 좌표값으로 예측 수행.
        xx, yy = np.meshgrid(np.linspace(xlim_start, xlim_end, num=200), np.linspace(ylim_start, ylim_end, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # contourf() 를 이용하여 class boundary 를 visualization 수행.
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3,
                               levels=np.arange(n_classes + 1) - 0.5,
                               cmap='rainbow', clim=(y.min(), y.max()),
                               zorder=1)
        #plt.show()
        #plt.savefig('./save/visualize_boundary_leaf.png')

if __name__ == '__main__':
    s = Solution()
    s.overfiting()
