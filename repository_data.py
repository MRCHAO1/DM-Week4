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
path = 'C:/Users/admin/Desktop/数据挖掘作业cy/w4/repository_data.csv'
data_git = pd.read_csv(path, header=0, engine='python', encoding='utf-8')
data_git_value = data_git.values
print('属性的值和频数如下所示：')
for i in range(data_git_value.shape[1]):  # 对所有列进行频数的统计
    counter = Counter(data_git_value[:, i])
    print(counter.most_common(5))  # 取前5项

# 3.1.2 数值属性
print("本数据集的五数概括、有效个数、平均值等如下所示：")
print(data_git.describe())  # 五数概括、有效个数、平均值等

# 3.1.3 数据可视化：使用直方图、盒图检查数据分布及离群点
# 直方图检查数据分布
draw_ori(data_git, 7, 'commit_count','Original Data', 30)
# 盒图检查数据分布
data_git['commit_count'].plot.box(title="Box Table")
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
print(missing_data(data_git))

# 3.2.1 将缺失部分剔除
data_del = data_git.dropna()
draw_ori(data_del, 7, 'commit_count', 'After Delete', 30)
print('***********************************将缺失部分剔除后的数据缺失情况↓***********************************')
print(missing_data(data_del))

# 3.2.2 用最高频率值来填补缺失值
miss_features = ['name','commit_count','licence']
data_fill = data_git
for col in miss_features:
     word_counts = Counter(data_fill[col])
     top = word_counts.most_common(1)[0][0]
     if type(top) != str:
         if isnan(top):
            top = word_counts.most_common(2)[1][0]
     temp = data_fill[col].fillna(top)
     data_fill[col] = temp
draw_ori(data_fill, 7, 'commit_count', 'After Fill with Max Frequency', 30)
print('***********************************用最高频率值来填补缺失值数据缺失情况↓***********************************')
print(missing_data(data_fill))

# 3.2.3 通过属性的相关关系来填补缺失值
data_fill_name = data_git
temp = data_fill_name['primary_language'].fillna(data_git['name'])
data_fill_name['primary_language'] = temp
draw_ori(data_git, 5, 'primary_language', 'Original Data', 30)
draw_ori(data_fill_name, 5, 'primary_language', 'Use Correlance for primary_language', 30)
print('***********************************通过属性的相关关系来填补缺失值(用name来填补primary_language)数据缺失情况↓***********************************')
print(missing_data(data_fill_name))

# 3.2.4 通过数据对象之间的相似性来填补缺失值
# 先用众数填充primary_language
miss_features = ['primary_language']
data_fill_language = data_git
for col in miss_features:
     word_counts = Counter(data_fill_language[col])
     top = word_counts.most_common(1)[0][0]
     if type(top) != str:
         if isnan(top):
            top = word_counts.most_common(2)[1][0]
     # print(top, type(top))
     temp = data_fill_language[col].fillna(top)
     data_fill_language[col] = temp
# 根据填充好的primary_language来填补languages_used
data_fill_languages_used = data_fill_language
temp = data_fill_languages_used['languages_used'].fillna(data_fill_language['primary_language'])
data_fill_languages_used['languages_used'] = temp
draw_ori(data_git, 6, 'languages_used', 'Original Data', 30)
draw_ori(data_fill_languages_used, 6, 'languages_used', 'Use Similarity for languages_used', 30)
print('***********************************通过数据对象之间的相似性来填补缺失值(用众数来填补primary_language，再用填充好的primary_language来填补languages_used)数据缺失情况↓***********************************')
print(missing_data(data_fill_languages_used))