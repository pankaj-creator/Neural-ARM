# -*- coding: utf-8 -*-
"""
@author: Pankaj, Aditya
"""
import time
start = time.time()
import warnings
warnings.simplefilter("ignore", UserWarning)
%pip install pandas
%pip install numpy
%pip install plotly
%pip install mlxtend --upgrade
!pip install pyfpgrowth
import pyfpgrowth
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# dataset
cols = ['1','2','3','4','5','6','7','8','9','10','11','12',
        '13','14','15','16','17','18','19','20','21','22',
        '23','24','25','26','27','28','29','30','31','32']

df = pd.read_csv("groceries.csv", sep = ",", 
                 names = cols, engine = "python")
data = np.array(df)

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


dataset = pd.read_csv("groceries.csv", sep = ",", 
                 names = cols, engine = "python")
#printing the shape of the dataset
dataset.shape
# printing the columns and few rows using head
dataset.head()


#Generating numpy transactions array
# importing module
import numpy as np
# Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, dataset.shape[0]):
    for j in range(0, dataset.shape[1]):
        transaction.append(dataset.values[i,j])
# converting to numpy array
transaction = np.array(transaction)
print(transaction)

#Top 5 items
#  Transform Them a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"]) 

# Put 1 to Each Item For Making Countable Table, to be able to perform Group By
df["incident_count"] = 1 

#  Delete NaN Items from Dataset
indexNames = df[df['items'] == "nan" ].index
df.drop(indexNames , inplace=True)

# Making a New Appropriate Pandas DataFrame for Visualizations  
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

#  Initial Visualizations
df_table.head(5).style.background_gradient(cmap='Blues')


#Tree map
# importing required module
import plotly.express as px

# to have a same origin
df_table["all"] = "Top 50 items" 

# creating tree map using plotly
fig = px.treemap(df_table.head(50), path=['all', "items"], values='incident_count',
                  color=df_table["incident_count"].head(50), hover_data=['items'],
                  color_continuous_scale='Blues',
                )
# ploting the treemap
fig.show()

#Encoding
# Transform Every Transaction to Seperate List & Gather Them into Numpy Array
transaction = []
for i in range(dataset.shape[0]):
    transaction.append([str(dataset.values[i,j]) for j in range(dataset.shape[1])])

# creating the numpy array of the transactions
transaction = np.array(transaction)

# importing the required module
from mlxtend.preprocessing import TransactionEncoder

# initializing the transactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)

# dataset after encoded
dataset.head()
#Top 40 items
# select top 40 items
first40 = df_table["items"].head(40).values 
print(first40)
# Extract Top 40
dataset = dataset.loc[:,first40] 
# shape of the dataset
dataset.shape

#FP growth algorithm
#Importing Libraries


from mlxtend.frequent_patterns import fpgrowth
#from mlxtend.frequent_patterns import fpgrowth
#running the fpgrowth algorithm
res=fpgrowth(dataset, min_support=0.002, use_colnames=True)

# printing top 10
res.head(10)

#Association rules
# importing required module
from mlxtend.frequent_patterns import association_rules

# creating asssociation rules
fpgrowth_rules=association_rules(res, metric="lift", min_threshold=0.5,)

# printing association rules
fpgrowth_rules
#Sorting
# Sort values based on confidence
fpgrowth_rerult=fpgrowth_rules.sort_values("confidence",ascending=False)

fpgrowth_rerult.to_csv("fpgrowth_rerult.csv", sep=',') 


end = time.time()
print("Time consumed in working of FP-Growth algorithm: ",end - start)