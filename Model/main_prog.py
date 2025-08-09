# -*- coding: utf-8 -*-
"""
@author: Pankaj, Aditya
"""
import time
start = time.time()
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter("ignore", UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

cols = ['1','2','3','4','5','6','7','8','9','10','11','12',
        '13','14','15','16','17','18','19','20','21','22',
        '23','24','25','26','27','28','29','30','31','32']

df = pd.read_csv("dataset/groceries-1.csv", sep = ",", 
                 names = cols, engine = "python")
data = np.array(df)
df.head(5)

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

train_test_split = np.random.rand(len(onehot_items)) < 0.80
train_x = onehot_items[train_test_split]
test_x = onehot_items[~train_test_split]
print()

train_validation_split = np.random.rand(len(train_x)) < 0.80
validation_x = train_x[~train_validation_split]
train_x = train_x[train_validation_split]

def weight_variable(weight_name, weight_shape):
    return tf.get_variable(name = "weight_" + weight_name,shape = weight_shape, 
            initializer = tf.contrib.layers.xavier_initializer()) 

def bias_variable(bias_shape):
    initial = tf.constant(0.1, shape = bias_shape)
    return tf.Variable(initial)

def encoder(x):
    l1 = tf.nn.softsign(tf.add(tf.matmul(x,e_weights_h1),e_biases_h1))
    l2 = tf.nn.softsign(tf.add(tf.matmul(l1,e_weights_h2),e_biases_h2))
    l3 = tf.nn.softsign(tf.add(tf.matmul(l2,e_weights_h3),e_biases_h3))
    l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3,e_weights_h4),e_biases_h4))
    return l4
    
def decoder(x):
    l1 = tf.nn.softsign(tf.add(tf.matmul(x,d_weights_h1),d_biases_h1))
    l2 = tf.nn.softsign(tf.add(tf.matmul(l1,d_weights_h2),d_biases_h2))
    l3 = tf.nn.softsign(tf.add(tf.matmul(l2,d_weights_h3),d_biases_h3))
    l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3,d_weights_h4),d_biases_h4))
    return l4

input_dim = 169

n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 32
n_hidden_4 = 16

training_epochs = 30
batch_size = 50
total_batches = (train_x.shape[0] // batch_size)

learning_rate = 0.0010 #data obtain in range 0.0002 to 0.0010 
keep_prob = 0.8
l2_reg_rate = 0.000001

tf.reset_default_graph()

is_training = tf.placeholder_with_default(False, shape = ())
X = tf.placeholder(tf.float32, shape=[None,input_dim])
X_drop = tf.contrib.layers.dropout(X, keep_prob, is_training = is_training)
#X_drop = tf.contrib.layers.dropout(X, keep_prob, noise_shape=None, is_training=True, outputs_collections=None, scope=None, seed=None )
print("X_drop", X_drop)
# --------------------- Encoder Variables --------------- #

e_weights_h1 = weight_variable("el1",[input_dim, n_hidden_1])
e_biases_h1 = bias_variable([n_hidden_1])

e_weights_h2 = weight_variable("el2",[n_hidden_1, n_hidden_2])
e_biases_h2 = bias_variable([n_hidden_2])

e_weights_h3 = weight_variable("el3",[n_hidden_2, n_hidden_3])
e_biases_h3 = bias_variable([n_hidden_3])

e_weights_h4 = weight_variable("el4",[n_hidden_3, n_hidden_4])
e_biases_h4 = bias_variable([n_hidden_4])
print(e_biases_h4)
# --------------------------------------------------------- #


# --------------------- Decoder Variables --------------- #

d_weights_h1 = weight_variable("dl1",[n_hidden_4, n_hidden_3])
d_biases_h1 = bias_variable([n_hidden_3])

d_weights_h2 = weight_variable("dl2",[n_hidden_3, n_hidden_2])
d_biases_h2 = bias_variable([n_hidden_2])

d_weights_h3 = weight_variable("dl3",[n_hidden_2, n_hidden_1])
d_biases_h3 = bias_variable([n_hidden_1])

d_weights_h4 = weight_variable("dl4",[n_hidden_1, input_dim])
d_biases_h4 = bias_variable([input_dim])

# --------------------------------------------------------- #

encoded = encoder(X_drop)
print("HHHH==",encoded)
decoded = decoder(encoded)
print("MMMM==", decoded) 

regularizer = tf.contrib.layers.l2_regularizer(l2_reg_rate)
reg_loss = regularizer(e_weights_h1) + regularizer(e_weights_h2) + regularizer(e_weights_h3) + regularizer(e_weights_h4) 
cost_function = -tf.reduce_mean(((X * tf.log(decoded)) + ((1 - X) * tf.log(1 - decoded)))) + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)


with tf.Session() as session:
    tf.global_variables_initializer().run()
    print("Epoch","  ","Tr. Loss"," ","Val. Loss")
    for epoch in range(training_epochs):
        for b in range(total_batches):
            offset = (b * batch_size) % (train_x.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :] 
            _, c = session.run([optimizer, cost_function],feed_dict={X: batch_x, is_training: True})
    
        tr_c = session.run(cost_function,feed_dict={X: train_x, is_training: False})
        val_c = session.run(cost_function,feed_dict={X: validation_x, is_training: False})
        print(epoch+1,"\t",tr_c," ",val_c)
    
    tr_p = session.run(decoded,feed_dict={X: train_x, is_training: False})
    roc_auc = roc_auc_score(train_x,tr_p,average = "samples")
    print("Training ROC AUC: ", round(roc_auc,4))

    val_p = session.run(decoded,feed_dict={X: validation_x, is_training: False})
    roc_auc = roc_auc_score(validation_x,val_p,average = "samples")
    print("Validation ROC AUC: ", round(roc_auc,4))
    
    ts_p = session.run(decoded,feed_dict={X: test_x, is_training: False})
    roc_auc = roc_auc_score(test_x,ts_p,average = "samples")
    print("Test ROC AUC: ", round(roc_auc,4),"\n")
    
    
    # -------------------------------------------------------------------------------- #
    item_preds = session.run(decoded,feed_dict={X: test_x.reshape(-1,169), is_training: False})
    print("ttt==",len(item_preds))
    item_preds[item_preds >= 0.05] = 1
    item_preds[item_preds < 0.05] = 0
    
"""   
i = 100
#print("Items in basket: ",get_items_from_ohe(test_x[i],unique_items))
print("Recommended item(s): ",get_items_from_ohe(item_preds[i],unique_items))
"""
model = NMF(n_components=100, init='random', random_state=0)
transation_w = model.fit_transform(item_preds)
items_h = model.components_

#get_items_from_ohe(items_h[1],unique_items)

print("Frequent item(s): ",get_items_from_ohe(items_h[0],unique_items))


total_transection=9835
for i in range(100):
    xx=np.asarray(np.nonzero(items_h[i]))
    [m,n]=(np.asarray(np.nonzero(items_h[i]))).shape
    support_A_B=len(np.intersect1d(np.nonzero(onehot_items[:,xx[0,n-1]]),np.nonzero(onehot_items[:,xx[0,n-2]])))/total_transection
    #print("{" + str(unique_items[xx[0,n-1]]) + "}" +"=>" "{" + str(unique_items[xx[0,n-2]]) + "}")
    [xxxx,sup_A]=(np.asarray(np.nonzero(onehot_items[:,xx[0,n-1]]))).shape
    [xxxx,sup_B]=(np.asarray(np.nonzero(onehot_items[:,xx[0,n-2]]))).shape
    support_A=sup_A/total_transection
    support_B=sup_B/total_transection
    confidence_A_B=support_A_B / (support_A)
    confidence_B_A=support_A_B / (support_B)
    #lift and conviction
    lift_A_B=confidence_A_B / support_B
    conviction_A_B=(1-support_B)/(1-confidence_A_B)
    print(i+1, "{" + str(unique_items[xx[0,n-1]]) + "}" +"=>" "{" + str(unique_items[xx[0,n-2]]) + "}"+" "+ "support="+ str(round(support_A_B,2)) +" "+ "confidence=" + str(round(confidence_A_B,2))  +" "+ "lift=" + str(round(lift_A_B,2))  +" "+ "conviction=" + str(round(conviction_A_B,2)) )

'''    temp = pd.DataFrame({'item1':[ item1], 'item2': [item2], 'Support': (item[1]), 'Confidence': (item[2][0][2]), 'Lift': (item[2][0][3]), 'Conviction':(Convicn)})
    result_apriory = pd.concat([result_mainProg, temp])
result_mainProg.to_csv("result_mainProg.csv", sep=',') '''
end = time.time()
print("Time consumed in working optimized algorithm: ",end - start)