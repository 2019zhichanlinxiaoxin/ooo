#coding:gbk
"""
利用决策树算法进行分类
植产一班林孝欣
日期：2020/5/11
"""
import pandas as pds        #调入需要用的第三方库
import numpy as npy
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import seaborn as sbn
#%matplotlib inline
# 调入数据
df=pds.read_csv("frenchwine.csv")
df.columns=["alcohol", "malic_acid", "ash", "alcalinity ash", "magnesium","species"]
# 查看前5条数据
df.head()
print(df.head()) 
# 查看数据描述性统计信息
df.describe()
print(df.describe())

def scatter_plot_by_category(feat, x, y): #数据的可视化 
    alpha = 0.5
    a = df.groupby(feat)
    b = cm.rainbow(npy.linspace(0, 1, len(a)))
    for g, c in zip(a, b):
        plot.scatter(g[1][x], g[1][y], color=c, alpha=alpha)

plot.figure(figsize=(20,5))
plot.subplot(131)
scatter_plot_by_category("species", "alcohol", "ash")
plot.xlabel("alcohol")
plot.ylabel("ash")
plot.title("species")
plot.show()

plot.figure(figsize=(20, 10)) #利用seaborn库绘制三种Iris花不同参数图
for column_index, column in enumerate(df.columns):
    if column == "species":
        continue
    plot.subplot(3, 2, column_index + 1)
    sbn.violinplot(x="species", y=column, data=df)
plot.show()

# 首先对数据进行切分，即划分出训练集和测试集
from sklearn.model_selection import train_test_split#调入sklearn库中交叉检验，划分训练集和测试集
all_inputs = df[["alcohol", "malic_acid",
                             "ash", "alcalinity ash","magnesium"]].values
all_species = df["species"].values

(X_train,
 X_test,
 Y_train,
 Y_test) = train_test_split(all_inputs, all_species, train_size=0.8, random_state=1)#80%的数据选为训练集

# 使用决策树算法进行训练
from sklearn.tree import DecisionTreeClassifier #调入sklearn库中的DecisionTreeClassifier来构建决策树
# 定义一个决策树对象
decision_tree_classifier = DecisionTreeClassifier()
# 训练模型
model = decision_tree_classifier.fit(X_train, Y_train)
# 输出模型的准确度
print(decision_tree_classifier.score(X_test, Y_test)) 


# 使用训练的模型进行预测，为了方便，
# 案例直接把测试集里面的数据拿出来三条
print(X_test[0:3])#利用3个数据进行测试，即取3个数据作为模型的输入端
model.predict(X_test[0:3])
predict_result = []
for i in model.predict(X_test[0:3]):
	if i == 'Zinfandel':
		i = '仙粉黛'
		predict_result.append(i)
	elif i == 'Syrah':
		i = '西拉'
		predict_result.append(i)
	elif i == 'Sauvignon':
         i = '赤霞珠'
         predict_result.append(i)
print(predict_result) #输出测试的结果，即输出模型预测的结果
 
