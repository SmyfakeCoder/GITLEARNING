from global_setting import *

def download_model(model):
    client_model = [model]*client_N
    return client_model
def fedavg(list):
    sum_dict = dict()
    for key in list[0].keys():
        sum_dict[key] = list[0][key] * 0
    for key in list[0].keys():
        for i in range(client_N):
            sum_dict[key] += list[i][key]
    for key in sum_dict.keys():
        sum_dict[key] /= client_N
    return sum_dict