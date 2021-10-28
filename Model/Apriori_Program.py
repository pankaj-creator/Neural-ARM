# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:01:26 2019

@author: H K PATEL
"""
import time
start = time.time()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from sklearn.decomposition import NMF
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import seaborn as sns

sns.set_style("whitegrid")

cols = ['1','2','3','4','5','6','7','8','9','10','11','12',
        '13','14','15','16','17','18','19','20','21','22',
        '23','24','25','26','27','28','29','30','31','32']

df = pd.read_csv("groceries.csv", sep = ",", 
                 names = cols, engine = "python")
data = np.array(df)
t=df.head(5)
print(t)

def get_unique_items(data):
    ncol = data.shape[1]
    items = set()
    for c in range(ncol):
        items = items.union(data[:,c])
    items = np.array(list(items))
    items = items[items != np.array(None)]

    return np.unique(items)

def get_onehot_items(data,unique_items):
    onehot_items = np.zeros((len(data),len(unique_items)),dtype = np.int)
    for i, r in enumerate(data):
        for j, c in enumerate(unique_items):
            onehot_items[i,j] = int(c in r)
            
    return onehot_items

def get_items_from_ohe(ohe,unique_items):
    return unique_items[np.flatnonzero(ohe)]
unique_items = get_unique_items(data)
onehot_items = np.array(get_onehot_items(data, unique_items))


n = 15
item_counts = (onehot_items != 0).sum(0)
items_max_args = item_counts.argsort()[-n:][::-1]
ic = pd.DataFrame({"Items":unique_items[items_max_args], "Frequency":item_counts[items_max_args]})


fig = plt.figure(figsize = (16,8))
sns.barplot(x="Items", y="Frequency", data=ic, palette=sns.color_palette("Set2", 10))
plt.xlabel("Items")
plt.ylabel("Frequency")
plt.title(str(n) + " Most frequent items in the dataset")
plt.show()


records = []
df = pd.read_csv("groceries.csv", sep = ",", names = cols, engine = "python")
for i in range(0, len(df)):
    records.append([str(df.values[i,j]) for j in range(0, 32)])


association_rules = apriori(records,min_support=0.004, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)
result_apriory = pd.DataFrame()
for item in association_results:
    if len((list(item.ordered_statistics[0].items_base)))< 2:
       continue
    
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    item1=list(item.ordered_statistics[0].items_base)
    if 'nan' in item1: item1.remove('nan')
    if 'None' in item1: item1.remove('None')
    item2=list(item.ordered_statistics[0].items_add)
    if 'nan' in item2: item2.remove('nan')
    if 'None' in item2: item2.remove('None')
    
    print("Rule: " + str(item1) + " -> " + str(item2))

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th 
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
        
    temp = pd.DataFrame({'item1':[ item1], 'item2': [item2], 'Support': (item[1]), 'Confidence': (item[2][0][2]), 'Lift': (item[2][0][3])})
    result_apriory = pd.concat([result_apriory, temp])

result_apriory.to_csv("result_apriory.csv", sep=',')    
end = time.time() 
print("Time consumed in working: ",end - start)