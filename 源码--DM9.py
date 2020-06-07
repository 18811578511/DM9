import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('vgsales.csv')
data.dropna(inplace=True)
print(data)

'''
最受欢迎的游戏、类型、发行平台以及发行商
'''

''' 受欢迎的游戏，前十名 '''
popular_game = data[:10][['Name', 'Global_Sales']]
popular_game.set_index('Name', inplace=True)
print(popular_game.index)
popular_game.plot.bar()


''' 收欢迎的类型、平台以及发行商，前十名'''
k = 10
idx = ['Genre', 'Platform', 'Publisher']
for attr in idx:
    attr_data = data[[attr, 'Global_Sales']]
    attr_data.set_index(attr, inplace=True)
    attr_data_counts = attr_data.groupby(attr).sum()
    attr_data_counts.sort_values(by='Global_Sales', ascending=False, inplace=True)
    attr_data_counts = attr_data_counts[:k]
    print(attr_data_counts.index)
    plt.figure()
    attr_data_counts.plot.bar()
    plt.show()

'''
最受欢迎的10游戏分别是:
'Wii Sports', 'Super Mario Bros.', 'Mario Kart Wii','Wii Sports Resort', 'Pokemon Red/Pokemon Blue', 'Tetris',
'New Super Mario Bros.', 'Wii Play', 'New Super Mario Bros. Wii','Duck Hunt'

最受欢迎的10个游戏类型、平台、发行商分别是:

类型:
'Action', 'Sports', 'Shooter', 'Role-Playing', 'Platform', 'Misc','Racing', 'Fighting', 'Simulation', 'Puzzle'

平台:
'PS2', 'X360', 'PS3', 'Wii', 'DS', 'PS', 'GBA', 'PSP', 'PS4', 'PC'

发行商:
'Nintendo', 'Electronic Arts', 'Activision','Sony Computer Entertainment', 'Ubisoft', 'Take-Two Interactive', 'THQ',
'Konami Digital Entertainment', 'Sega', 'Namco Bandai Games'
'''


'''预测每年的游戏销售额
我们推测，每年的游戏销售额应与以下几个因素相关:
(1). 不同发行商这一年发行的游戏数量
(2). 这一年内发行的游戏种类
(3). 这一年游戏平台的数量
这里我们以年为单位，自变量为上述因素，因变量为全球销售额，做一个回归预测任务
销量前20的游戏平台、发行商占总销量的98.35% 与 85.72%
因此在考虑平台、发行商影响时，只考虑销量前20的的平台与发行商
游戏类型有12个，因此每一年的输入向量是一个(12 + 20 + 20) = 52维的向量
也就是说,输入向量的1~12表示12个发行商在这一年发行的游戏数量
13~32, 33~52分别表示20种类型、20个平台在这一年的游戏数量
'''


total_years = list(set(data['Year']))
total_years = total_years[:-2]
print(total_years)
len_years = len(total_years)
# x = np.zeros((len_years, 52))
# print(x.shape)

y = []
for year in total_years:
    year_idx = data['Year'] == year
    year_sales = data['Global_Sales'][year_idx].sum()
    y.append(year_sales)
print(y)
print(len(y))

'''
确定输入向量位置含义
'''
top_k = 20
idx = ['Genre', 'Platform', 'Publisher']
step = 0
vec_dict = {}
for attr in idx:
    attr_data = data[[attr, 'Global_Sales']]
    attr_data.set_index(attr, inplace=True)
    attr_data_counts = attr_data.groupby(attr).sum()
    attr_data_counts.sort_values(by='Global_Sales', ascending=False, inplace=True)
    attr_data_counts = attr_data_counts[:top_k]

    for item in attr_data_counts.index:
        vec_dict[item] = step
        step += 1
print(vec_dict)


idx = ['Genre', 'Platform', 'Publisher']

attr_feat_dim = {
    'Genre': 12,
    'Platform': 20,
    'Publisher': 20
}
attr_feats = np.zeros((len_years, 52))

for attr in idx:
    attr_data = data[[attr, 'Year']]
    for k, year in enumerate(total_years):
        year_idx = attr_data['Year'] == year
        year_sales = attr_data[year_idx][[attr]]
        counts = pd.Series(np.squeeze(year_sales.values)).value_counts().to_frame()
        for item, count in counts.iterrows():
            if item in vec_dict.keys():
                pos = vec_dict[item]
                attr_feats[k][pos] = count.values[0]

print(attr_feats)
print(attr_feats.shape)


'''
用线性回归来做回归预测
pearson系数与rmse作为评价指标
pearson系数越接近1， rmse越小则预测效果越好
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
x = attr_feats
y = np.array(y)


from scipy.stats import pearsonr
# 做一个列平均
x = x  / np.sum(x, 0)
print(x)
print(x.shape)
print(y.shape)
kf = KFold(n_splits=5, shuffle=True, random_state=123)
model = LinearRegression()

y_pred = np.zeros(y.shape)
for train, test in kf.split(x, y):
    model.fit(x[train], y[train])
    y_pred[test] = model.predict(x[test])

rmse = np.sqrt(np.sum((y_pred - y) ** 2) / y.shape[0])
print(rmse)
r = pearsonr(y, y_pred)[0]
print(r)
plt.scatter(y, y_pred)
plt.xlabel('real sales')
plt.ylabel('predicted sales')
plt.show()

'''
结果显示模型的预测效果不错，表明一开始定义的三个因素与年销售额非常相关
'''



