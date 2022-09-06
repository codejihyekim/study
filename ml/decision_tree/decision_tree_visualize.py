from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
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
        x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=66)

        # DecisionTreeClassifier 학습
        dt_clf.fit(x_train, y_train)

        #export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함
        #export_graphviz(dt_clf, out_file='./save/tree.dot', class_names=iris_data.target_names, feature_names=iris_data.feature_names, impurity=True, filled=True)

        # 위에서 생성된 tree.dot 파일을 Graphviz가 읽어서 시각화
        with open("./save/Source.gv") as f:
            dot_graph = f.read()
        #graph = graphviz.Source(dot_graph)
        #graph.render(view=True)

if __name__ == '__main__':
    s = Solution()
    s.test()
