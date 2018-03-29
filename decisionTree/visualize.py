# -*- encoding: utf-8 -*-

from decisionTree import *
from sklearn import tree
from IPython.display import Image
import pydotplus
import sys
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dtc = DecisionTreeClassifier(random_state=14)
X_prewins = df[['Hlastwin', 'Vlastwin', 'h_last5', 'v_last5','hLastDays', 'vLastDays', 'HrankHigher', 'h_wonlast']].values
y_true = df['Homewin'].values
clf = dtc.fit(X_prewins, y_true)

##  第一种方式,利用graphviz的dot命令生成决策树的可视化文件
with open("nba.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
#注意，这个命令在命令行执行
# dot -Tpdf nba.dot -o nba.pdf    ##  生成pdf的图片文件

## 第二种方式，用pydotplus生成pdf。这样就不用再命令行去专门生成pdf文件了
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("nba.pdf")

## 第三种方式，直接在jupyter notebook中生成
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=['Hlastwin', 'Vlastwin', 'h_last5', 'v_last5','hLastDays', 'vLastDays', 'HrankHigher', 'h_wonlast'],
                         class_names=['Homewin'],
                         filled=True, rounded=True,
                         special_characters=False)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
