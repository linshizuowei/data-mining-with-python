# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

date ='Date'
start = 'Start'
visitor = 'Visitor'
vpts = 'Vpts'
home = 'Home'
hpts = 'Hpts'
stype = 'Score Type'
ot = 'OT'
notes = 'Notes'
homewin = 'Homewin'

def min_max_normalize(df, name):
    # 归一化
    max_number = df[name].max()
    min_number = df[name].min()
    # assert max_number != min_number, 'max == min in COLUMN {0}'.format(name)
    df[name] = df[name].map(lambda x: float(x - min_number + 1) / float(max_number - min_number + 1))
    # 做简单的平滑,试试效果如何
    return df

def cal_mean(res):
    # 计算均值
    win = 0
    for i in res:
        win += i
    return win * 1.0 / len(res) if len(res) > 0 else 0

def get_key(s1, s2):
    if s1 < s2:
        return s1 + s2
    else:
        return s2 + s1

def dataset_cleaning(df):
    # 数据集的列名有重复，而且不够简洁， 重新命名
    df.columns = ['Date', 'Start',	'Visitor',	'Vpts',	 'Home',	 'Hpts',	 'Score Type', 'OT', 'Notes']
    # 处理数据集中的缺失值
    df['OT'] = df['OT'].map(lambda x: 1 if x == 'OT' else 0)
    df['Notes'].fillna(-1, inplace=True)
    return df

def homeWin(df):
    # 提取标签  ----  主队是否获胜
    series = pd.Series(list(map(lambda x, y: 1 if x > y else 0, df['Hpts'], df['Vpts'])))
    series.name = 'Homewin'
    df = df.join(series)
    return df

def lastWin(df):
    # 提取特征  ---- 主队上一场比赛是否获胜；客队上一场比赛是否获胜；
    lastwin = {}
    hlast = []
    vlast = []
    for index, row in df.iterrows():
        visitor = row['Visitor']
        home = row['Home']
        if visitor not in lastwin:
            lastwin[visitor] = 0
        if home not in lastwin:
            lastwin[home] = 0
        hlast.append(lastwin[home])
        vlast.append(lastwin[visitor])
        lastwin[home] = 1 if row['Vpts'] < row['Hpts'] else 0
        lastwin[visitor] = 0 if row['Vpts'] < row['Hpts'] else 1
    series_h = pd.Series(hlast)
    series_h.name = 'Hlastwin'
    series_v = pd.Series(vlast)
    series_v.name = 'Vlastwin'
    df = df.join(series_h)
    df = df.join(series_v)
    return df



def last_5_games(df):
    passgames = {}
    h_last5 = []
    v_last5 = []
    for index, row in df.iterrows():
        if row[home] not in passgames:
            passgames[row[home]] = []
        if row[visitor] not in passgames:
            passgames[row[visitor]] = []
        h_last5.append(cal_mean(passgames[row[home]]))
        v_last5.append(cal_mean(passgames[row[visitor]]))
        passgames[row[home]].append(row[homewin])
        passgames[row[visitor]].append(1-row[homewin])
        if len(passgames[row[home]]) > 5:
            passgames[row[home]].pop(0)
        if len(passgames[row[visitor]]) > 5:
            passgames[row[visitor]].pop(0)

    hseries = pd.Series(h_last5, name='h_last5')
    vseries = pd.Series(v_last5, name='v_last5')
    df = df.join(hseries)
    return df.join(vseries)

def lastDays(df):
    # 球队距离上次比赛的间隔天数
    # 加入该特征后准确性下降，原因待查~
    lastdate = {}
    hlastdays = []
    vlastdays = []
    for index, row in df.iterrows():
        if row[visitor] not in lastdate:
            lastdate[row[visitor]] = row[date]
        if row[home] not in lastdate:
            lastdate[row[home]] = row[date]
        hdays = (row[date] - lastdate[row[home]]).days
        vdays = (row[date] - lastdate[row[visitor]]).days
        hlastdays.append(hdays)
        vlastdays.append(vdays)
        lastdate[row[home]] = row[date]
        lastdate[row[visitor]] = row[date]
    series_h = pd.Series(hlastdays, name='hLastDays')
    series_v = pd.Series(vlastdays, name='vLastDays')
    df = df.join(series_h)
    df = df.join(series_v)
    df = min_max_normalize(df, series_h.name)
    df = min_max_normalize(df, series_v.name)
    return df

def h_team_rankhiger(df):
    # 主队是否上赛季排名更高
    standing = pd.read_table('new.txt', sep=',', skiprows=[0])
    teamrank = {}
    for i, row in standing.iterrows():
        if row['Team'] == 'New Orleans Hornets':
            row['Team'] = 'New Orleans Pelicans'
        teamrank[row['Team']] = row['Rk']
    series = pd.Series(list(map(lambda x, y : 1 if teamrank[x] < teamrank[y] else 0, df['Home'], df['Visitor'])))
    series.name = 'HrankHigher'
    return df.join(series)

def h_team_wonlast(df):
    # 主队是否在上次双方对阵时获胜
    teamres = {}
    lastwin = []
    for index, row in df.iterrows():
        key = get_key(row[home], row[visitor])
        if key not in teamres:
            teamres[key] = ''
            lastwin.append(-1)
        else:
            lastwin.append(1 if row[home] == teamres[key] else 0)
        teamres[key] = row[home] if row[hpts] > row[vpts] else row[visitor]
    series = pd.Series(lastwin, name='h_wonlast')
    return df.join(series)

def feature_extract(df, meths):
    for func in features_meth:
        df = func(df)
    df.to_csv('features.csv')

def decisionTree(df):
    # 模型训练及预测
    dtc = DecisionTreeClassifier(random_state=14)
    X_prewins = df[['Hlastwin', 'Vlastwin']].values
    y_true = df['Homewin'].values
    cross_val_score(dtc, X_prewins, y_true, scoring='accuracy')

if __name__ == '__main__':
    features_meth = [lastWin, last_5_games, lastDays, h_team_rankhiger, h_team_wonlast]
    df = pd.read_csv('data.csv', parse_dates=['Date'])
    df = dataset_cleaning(df)
    df = homeWin(df)
    feature_extract(df, features_meth)