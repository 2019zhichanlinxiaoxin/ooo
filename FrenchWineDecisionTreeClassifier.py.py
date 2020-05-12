#coding:gbk
"""
���þ������㷨���з���
ֲ��һ����Т��
���ڣ�2020/5/11
"""
import pandas as pds        #������Ҫ�õĵ�������
import numpy as npy
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import seaborn as sbn
#%matplotlib inline
# ��������
df=pds.read_csv("frenchwine.csv")
df.columns=["alcohol", "malic_acid", "ash", "alcalinity ash", "magnesium","species"]
# �鿴ǰ5������
df.head()
print(df.head()) 
# �鿴����������ͳ����Ϣ
df.describe()
print(df.describe())

def scatter_plot_by_category(feat, x, y): #���ݵĿ��ӻ� 
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

plot.figure(figsize=(20, 10)) #����seaborn���������Iris����ͬ����ͼ
for column_index, column in enumerate(df.columns):
    if column == "species":
        continue
    plot.subplot(3, 2, column_index + 1)
    sbn.violinplot(x="species", y=column, data=df)
plot.show()

# ���ȶ����ݽ����з֣������ֳ�ѵ�����Ͳ��Լ�
from sklearn.model_selection import train_test_split#����sklearn���н�����飬����ѵ�����Ͳ��Լ�
all_inputs = df[["alcohol", "malic_acid",
                             "ash", "alcalinity ash","magnesium"]].values
all_species = df["species"].values

(X_train,
 X_test,
 Y_train,
 Y_test) = train_test_split(all_inputs, all_species, train_size=0.8, random_state=1)#80%������ѡΪѵ����

# ʹ�þ������㷨����ѵ��
from sklearn.tree import DecisionTreeClassifier #����sklearn���е�DecisionTreeClassifier������������
# ����һ������������
decision_tree_classifier = DecisionTreeClassifier()
# ѵ��ģ��
model = decision_tree_classifier.fit(X_train, Y_train)
# ���ģ�͵�׼ȷ��
print(decision_tree_classifier.score(X_test, Y_test)) 


# ʹ��ѵ����ģ�ͽ���Ԥ�⣬Ϊ�˷��㣬
# ����ֱ�ӰѲ��Լ�����������ó�������
print(X_test[0:3])#����3�����ݽ��в��ԣ���ȡ3��������Ϊģ�͵������
model.predict(X_test[0:3])
predict_result = []
for i in model.predict(X_test[0:3]):
	if i == 'Zinfandel':
		i = '�ɷ���'
		predict_result.append(i)
	elif i == 'Syrah':
		i = '����'
		predict_result.append(i)
	elif i == 'Sauvignon':
         i = '��ϼ��'
         predict_result.append(i)
print(predict_result) #������ԵĽ���������ģ��Ԥ��Ľ��
 
