# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

def dataset_cleaning(df):
    # 数据集的列名有重复，而且不够简洁， 重新命名
    df.columns = ['Date', 'Start',	'Visitor',	'Vpts',	 'Home',	 'Hpts',	 'Score Type', 'OT', 'Notes']
    # 处理数据集中的缺失值
    df['OT'] = df['OT'].map(lambda x: 1 if x == 'OT' else 0)
    df['Notes'].fillna(-1, inplace=True)
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

def homeWin(df):
    # 提取标签  ----  主队是否获胜
    series = pd.Series(list(map(lambda x, y: 1 if x > y else 0, df['Hpts'], df['Vpts'])))
    series.name = 'Homewin'
    df = df.join(series)
    return df

def decisionTree(df):
    # 模型训练及预测
    dtc = DecisionTreeClassifier(random_state=14)
    X_prewins = df[['Hlastwin', 'Vlastwin']].values
    y_true = df['Homewin'].values
    cross_val_score(dtc, X_prewins, y_true, scoring='accuracy')