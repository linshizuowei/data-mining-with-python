# -*- encoding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter

userid = 'userId'
movieid = 'movieId'
rating = 'rating'
date = 'timestamp'
favorable = 'Favorable'

def data_clean_and_static(datafile):
    df = pd.read_csv(datafile, delimiter='\t', header=None, names=[userid, movieid, rating, date])
    df[date] = pd.to_datetime(df[date], unit='s')

    ## 统计数据基本信息
    print 'there are total %d logs' % (df.shape[0])
    print 'there are %d users' % (df[userid].nunique())
    print 'there are %d movies' % (df[movieid].nunique())
    print 'is there null:'
    print df.isnull().sum()
    return df

def find_frequent_set(favorable_reviews_by_users, k_1_itemsets, min_support):
    cnt = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other in reviews-itemset:
                    current_superset = itemset | frozenset((other,))
                    cnt[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in cnt.items() if frequency >= min_support])

def favorable_by_users(df):
    """
    提取用户喜欢的电影
    :param df:
    :return:
    """
    favorable_rating = df[df[favorable]]
    favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_rating.groupby(userid)[movieid])
    return favorable_reviews_by_users

def extract_frequent_sets(df):
    """
    从数据集中提取频繁项集
    :param df:
    :return:
    """
    favorable_reviews_by_users = favorable_by_users(df)
    num_favorable_by_movie = df[[movieid, favorable]].groupby(movieid).sum()  # 挑出每部电影有多少人喜欢（打分3分以上）， 而不是有多少人打分
    frequent_itemsets = {}
    min_support = 50
    frequent_itemsets[1] = dict((frozenset((movie_id,)), row[favorable])
                                for movie_id, row in num_favorable_by_movie.iterrows() if row[favorable] > min_support)

    for k in range(2, 20):
        cur_frequent_itemsets = find_frequent_set(favorable_reviews_by_users, frequent_itemsets[k-1], min_support)
        frequent_itemsets[k] = cur_frequent_itemsets
        if len(cur_frequent_itemsets) == 0:
            break
    del frequent_itemsets[1]
    return frequent_itemsets

def extract_rules(frequent_itemsets):
    """
    抽取关联规则
    :param frequent_itemsets:
    :return:
    """
    candidate_rules = []
    for itemset_length, itemset_counts in frequent_itemsets.items():
        for itemset in itemset_counts.keys():
            for conclusion in itemset:
                premise = itemset - set((conclusion,))
                candidate_rules.append((premise, conclusion))
    return candidate_rules

def cal_confidence(candidate_rules, favorable_reviews_by_users):
    """
    计算每条规则的置信度
    :param candidate_rules:
    :param favorable_reviews_by_users:
    :return:
    """
    correct_cnts = defaultdict(int)
    incorrect_cnts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for rule in candidate_rules:
            premise, conclusion = rule
            if premise.issubset(reviews):
                if conclusion in reviews:
                    correct_cnts[rule] += 1
                else:
                    incorrect_cnts[rule] += 1
    rule_confidence = {rule: correct_cnts[rule] / float(correct_cnts[rule] + incorrect_cnts[rule]) for rule in
                       candidate_rules}
    sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)
    return sorted_confidence

def get_movie_name(movie_id):
    """
    获取电影名称
    :param movie_id:
    :return:
    """
    return movie_name[movie_name[0] == movie_id][1].values[0]

def test_dataset(df):
    return favorable_by_users(df)

def test_cal_confidence(candidate_rules, test_favorable_reviews_by_users):
    return cal_confidence(candidate_rules, test_favorable_reviews_by_users)

if __name__ == '__main__':
    data_folder = os.path.join(os.path.expanduser('~'), 'Data', 'ml-100k')
    filename = os.path.join((data_folder, 'u.data'))
    movie_name = pd.read_csv('u.item', delimiter='|', header=None, encoding='mac-roman')

    for index in range(10):
        print 'Rule # %d' % (index)
        (premise, conclusion) = sorted_confidence[index][0]
        premise_name = ', '.join(get_movie_name(idx) for idx in premise)
        conclusion_name = get_movie_name(conclusion)
        print '     if a person recommends %s, they will also recommend %s' % (premise_name, conclusion_name)
        print ' - Train Confidence: %.3f' % (rule_confidence.get((premise, conclusion), -1))
        print ' - test Confidence: %.3f' % (test_confidence.get((premise, conclusion), -1))
        print
