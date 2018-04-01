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
