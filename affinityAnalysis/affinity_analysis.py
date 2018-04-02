import os
import pandas as pd
import numpy as np
from collections import defaultdict
userid = 'userId'
movieid = 'movieId'
rating = 'rating'
date = 'timestamp'

def find_frequent_set(user_bymovie, itemset, min_support):
    cnt = defaultdict(int)
    for user, reviews in user_bymovie.items():
        for item in itemset:
            if item.issubset(reviews):
                for other in reviews-item:
                    current_superset = item | frozenset((other,))
                    cnt[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in cnt.items() if frequency >= min_support])

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

if __name__ == '__main__':
    data_folder = os.path.join(os.path.expanduser('~'), 'Data', 'ml-100k')
    filename = os.path.join((data_folder, 'u.data'))