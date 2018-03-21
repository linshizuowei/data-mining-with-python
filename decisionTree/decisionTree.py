# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

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

def lastDays(df):
    # 球队距离上次比赛的间隔天数
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

def last_5_games(df, col):
    # 球队最近5场比赛的胜率
    passgames = {}
    last5 = []
    for index, row in df.iterrows():
        if row[home] not in passgames:
            passgames[row[home]] = []
        if row[visitor] not in passgames:
            passgames[row[visitor]] = []
        last5.append(cal_mean(passgames[row[col]]))
        passgames[row[home]].append(row[homewin])
        passgames[row[visitor]].append(1-row[homewin])
        if len(passgames[row[home]]) > 5:
            passgames[row[home]].pop(0)
        if len(passgames[row[visitor]]) > 5:
            passgames[row[visitor]].pop(0)

    series = pd.Series(last5, name=col+'last5')
    return df.join(series)


def decisionTree(df):
    # 模型训练及预测
    dtc = DecisionTreeClassifier(random_state=14)
    X_prewins = df[['Hlastwin', 'Vlastwin']].values
    y_true = df['Homewin'].values
    cross_val_score(dtc, X_prewins, y_true, scoring='accuracy')