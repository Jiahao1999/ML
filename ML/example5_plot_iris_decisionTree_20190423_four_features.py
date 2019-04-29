"""
基于决策树的鸢(yuan)尾花分类
考察基于四个特征联合描述样本，构造的二叉分类决策树模型
决策树的可视化

(1)此程序运行之前需要在Anaconda终端运行命令：
   conda install graphviz
(2)之后，添加环境变量:
   C:\ProgramData\Anaconda3\Library\bin\graphviz
(3)之后在终端运行命令：
   pip install pydotplus-2.0.2-py2.py3-none-any.whl
"""
## 0. 显示上述说明
print(__doc__)


## 1. (1)导入scikit-learn内置的datasets 数据集模块
#     (2)导入scikit-learn内置的tree包的 DecisionTreeClassifier API接口模块
from sklearn.datasets import load_iris
from sklearn import tree

## 2-1. PyDotPlus is an improved version of the old pydot project 
#     that provides a Python Interface to Graphviz's Dot language.
import pydotplus 

## 2-2. 决策树的可视化
from PIL import Image


## 3. 有关参数设置： Parameters
# 类别数=3； 绘制颜色表； 步长
n_classes = 3 
plot_colors = "ryb"
plot_step = 0.02

## 4.  Load data
# 加载iris数据集 ,得到数据集对象iris：3类别(0,1,2)，150个样本，4个特征
# 含5个成员：data,target,feature_names等
# 获取该数据集对象的data部分，以及类别标号
iris = load_iris()


## 5. 样本原始特征数目=4

X = iris.data
y = iris.target

# (2) 初始化决策树分类模型实例；并基于X，y 训练集，学习一棵分类树
clf = tree.DecisionTreeClassifier()
clf=clf.fit(X, y)
    
# 采用默认参数创建并初始化分类树实例DecisionTreeClassifier
#  criterion --切分特征选择准则，默认值为 ”gini” 
#  splitter -- default=”best” 选择切分点位置采用最小gini值，而不是随机法
#  max_depth --设定树的最大深度
#              If None, then nodes are expanded until all leaves are pure or 
#              until all leaves contain less than min_samples_split samples.
#  min_samples_split --default=2
#              The minimum number of samples required to split an internal node:
#              If int, then consider min_samples_split as the minimum number.
#              If float, then min_samples_split is a percentage and ceil(min_samples_split *
#              n_samples) are the minimum number of samples for each split.
#  min_samples_leaf : int, float, optional (default=1)
#              The minimum number of samples required to be at a leaf node:
#              If int, then consider min_samples_leaf as the minimum number.
#              If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf *
#                 n_samples) are the minimum number of samples for each node.
#  min_weight_fraction_leaf : float, optional (default=0.)
#              The minimum weighted fraction of the sum total of weights (of all the input samples)
#              required to be at a leaf node. Samples have equal weight when sample_weight is not
#              provided.
#  max_features : int, float, string or None, optional (default=None)
#              The number of features to consider when looking for the best split:
#              If int, then consider max_features features at each split.
#              If float, then max_features is a percentage and int(max_features * n_features) features
#              are considered at each split.
#              If “auto”, then max_features=sqrt(n_features).
#              If “sqrt”, then max_features=sqrt(n_features).
#              If “log2”, then max_features=log2(n_features).
#              If None, then max_features=n_features.

# (3)clf表示已经训练好的模型，即已经调用过DecisionTreeClassifier实例的fit(X_train, y_train)方法
# A.导出已经生成的决策树文件
# tree.export_graphviz(clf, out_file='tree.dot', 
#   feature_names=[iris.feature_names[pair[0]], iris.feature_names[pair[1]]])

# B.导出决策树模型，以pdf格式或gif格式文件将其保存在当前python文件所在目录
#dot_data=tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names)
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf('iris_FourFeature_tree.pdf')
graph.write_gif('iris_FourFeature_tree.gif')

im = Image.open('iris_FourFeature_tree.gif')  
im.show()   



