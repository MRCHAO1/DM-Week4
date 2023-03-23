import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from math import isnan

#显示所有列
pd.set_option('display.max_columns',None)
#显示所有行
pd.set_option('display.max_rows',None)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_ori(data,cl,xlabel,title, rank):
    num = rank
    data = data.values
    counter = Counter(data[:, cl])
    frequency = counter.most_common()  # 取前n项
    num_list = []
    name_list = []
    if  num > len(frequency):
        num = len(frequency)
    for i in range(num):
        num_list.append(int(frequency[i][1]))
        name_list.append(str(frequency[i][0]))
    fig, ax = plt.subplots()
    b = ax.bar(name_list, num_list)
    plt.bar(range(len(num_list)), num_list, color='blue', tick_label=name_list)
    for a, b in zip(name_list, num_list):
        ax.text(a, b + 1, b, ha='center', va='bottom')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.show()

# 3.1 数据摘要和可视化
# 3.1.1 标称属性
path = 'C:/Users/admin/Desktop/数据挖掘作业cy/w4/movies_dataset.csv'
data_movie = pd.read_csv(path, header=0, index_col=0, engine='python', encoding='utf-8')
data_movie_value = data_movie.values
print('************************************************属性的值和频数如下所示↓************************************************')
for i in range(data_movie_value.shape[1]):  # 对所有列进行频数的统计
    counter = Counter(data_movie_value[:, i])
    print(counter.most_common(5))  # 取前5项

# 3.1.2 数值属性
print("************************************************本数据集的五数概括、有效个数、平均值等如下所示↓************************************************")
print(data_movie.describe())  # 五数概括、有效个数、平均值等

# 3.1.3 数据可视化：使用直方图、盒图检查数据分布及离群点
# 直方图检查数据分布
draw_ori(data_movie, 0, 'IMDb-rating','Original Data', 20)
# 盒图检查数据分布
data_movie['IMDb-rating'].plot.box(title="Box Table")
plt.grid(linestyle="--")
plt.show()

# 3.2 缺失数据处理
# 3.2.1 构建查找缺失项函数，输出原始数据
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
print('***********************************原始数据缺失情况↓***********************************')
print(missing_data(data_movie))

# 3.2.1 将缺失部分剔除
data_del = data_movie.dropna()
draw_ori(data_movie, 1, 'appropriate_for','Original Data', 20)
draw_ori(data_del, 1, 'appropriate_for', 'After Delete', 20)
print('***********************************将缺失部分剔除后的数据缺失情况↓***********************************')
print(missing_data(data_del))

# 3.2.2 用最高频率值来填补缺失值
miss_features = ['IMDb-rating','appropriate_for','director']
data_fill = data_movie
for col in miss_features:
     word_counts = Counter(data_fill[col])
     top = word_counts.most_common(1)[0][0]
     if type(top) != str:
         if isnan(top):
            top = word_counts.most_common(2)[1][0]
     temp = data_fill[col].fillna(top)
     data_fill[col] = temp
draw_ori(data_fill, 1, 'appropriate_for', 'After Fill with Max Frequency', 20)
print('***********************************用最高频率值来填补缺失值数据缺失情况↓***********************************')
print(missing_data(data_fill))

# 3.2.3 通过属性的相关关系来填补缺失值
data_fill_name = data_movie
temp = data_fill_name['storyline'].fillna(data_fill_name['language'])
data_fill_name['storyline'] = temp
draw_ori(data_movie, 10, 'storyline', 'Original Data', 30)
draw_ori(data_fill_name, 10, 'storyline', 'Use Correlance for storyline', 30)
print('***********************************通过属性的相关关系来填补缺失值(用language来填补storyline)数据缺失情况↓***********************************')
print(missing_data(data_fill_name))

# 3.2.4 通过数据对象之间的相似性来填补缺失值
# 根据用众数填充好的director来填补storyline
data_fill_storyline = data_fill_name
miss_features = ['director']
data_fill = data_movie
for col in miss_features:
     word_counts = Counter(data_fill[col])
     top = word_counts.most_common(1)[0][0]
     if type(top) != str:
         if isnan(top):
            top = word_counts.most_common(2)[1][0]
     temp = data_fill[col].fillna(top)
     data_fill[col] = temp
temp = data_fill_storyline['storyline'].fillna(data_fill['director'])
data_fill_storyline['storyline'] = temp
draw_ori(data_movie, 10, 'storyline', 'Original Data', 30)
draw_ori(data_fill_storyline, 10, 'storyline', 'Use Similarity for storyline', 30)
print('***********************************通过数据对象之间的相似性来填补缺失值(用众数来填补director，再用填充好的director来填补storyline)数据缺失情况↓***********************************')
print(missing_data(data_fill_storyline))