from datetime import datetime, timedelta
import pandas as pd


def num_to_word(num):
    dict = { 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five', 6 : 'six',
            7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten', 11 : 'eleven'}
    return dict[num]

def datetime_range(start, end, diff):
    while start < end:
        yield start
        start += diff

def extract_time(start, end, diff):
    t = [dt.strftime('%H:%M') for dt in datetime_range(start, end, diff)]
    return t

def user_input_features(input, df_col_name):
    dict = {}
    f = [[float(i)] for i in input]
    for j in range(len(df_col_name)):
        dict[df_col_name[j]] = f[j]
    new_data = pd.DataFrame(dict)
    return new_data

def cluster_mapper(n_clus):
    cl_dict = {}
    for i in range(0, n_clus+1):
        cl_dict[i] = f"Cluster {i+1}"
    return cl_dict

def cluster_predict(n_clus, pred):
    dict_ = cluster_mapper(n_clus)
    return dict_[pred]
