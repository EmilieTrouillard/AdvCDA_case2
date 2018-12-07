# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:34:34 2018

@author: caspe
"""
import numpy as np
import time
import matplotlib.pyplot as plt

def confget(ante,conseq):
    #Antecedent +consequent, antecedent, frequent itemset for superset, freq items for ant. ,
    #occurence for ant + cons, 
    rulelen=len(conseq)
    antsup=F[len(ante)][ante]
    conseqsup=F[rulelen][conseq]
    conf=conseqsup/float(antsup)
    return conf

#%%
import pandas as pd
from itertools import combinations, groupby
from collections import Counter

orders_file = 'orders.csv'
products_file = 'products.csv'
prior_file = 'order_products__prior.csv'
train_file = 'order_products__train.csv'
aisles_file = 'aisles.csv'
departments_file = 'departments.csv'
#orders_file = 'instacart_2017_05_01/orders.csv'
#products_file = 'instacart_2017_05_01/products.csv'
#prior_file = 'instacart_2017_05_01/order_products__prior.csv'
#train_file = 'instacart_2017_05_01/order_products__train.csv'
#aisles_file = 'instacart_2017_05_01/aisles.csv'
#departments_file = 'instacart_2017_05_01/departments.csv'


#%%
df_orders = pd.read_csv(orders_file, engine='python')
NUMBER_ORDERS_TOTAL = len(df_orders)

#%% read the csv files
df_orderProductsTrain = pd.read_csv(train_file)
df_orderProductsPrior = pd.read_csv(prior_file)
df_allOrdersProducts = pd.concat([df_orderProductsPrior, df_orderProductsTrain], axis = 0)
df_allOrdersProducts = df_allOrdersProducts.reset_index()
df_products= pd.read_csv(products_file)
#df_departments= pd.read_csv(departments_file)
df_aisles= pd.read_csv(aisles_file)

#%% Set the index of the dataframe aisles to be the aisle.id
df_aisles = df_aisles.set_index(df_aisles['aisle_id'].values)
df_aisles = df_aisles.drop(['aisle_id'], axis=1)

#%% Display the number of products per aisle
aisles = df_products.groupby(['aisle_id']).count()
aisles = aisles.reset_index()
aisles = aisles.sort_values(by=['product_id'], ascending = False)
def get_aisle_name(row):
    return df_aisles.loc[row.aisle_id, 'aisle']
aisles['aisle_name'] = aisles.apply(get_aisle_name, axis = 1)
aisles = aisles.set_index(aisles['aisle_id'].values)
aisles = aisles.drop(['aisle_id', 'department_id', 'product_name'], axis = 1)
aisles.columns = ['number_of_products', 'aisle_name']

#%% Select the combination of the 3 aisles
df_fruits = df_products[df_products['aisle_id'].isin([24,83,123])]
products_id = df_fruits['product_id'].values
df_fruits = df_fruits.set_index(products_id)

aisle_orders = df_allOrdersProducts[df_allOrdersProducts['product_id'].isin(products_id)]
aisle_ordersproducts = aisle_orders.drop(['add_to_cart_order', 'reordered', 'index'], axis=1)

aisle_orders = aisle_ordersproducts.groupby(['order_id']).agg({'product_id': lambda x: tuple(x)})
aisle_orders.columns = ['products']

#%%
NUMBER_ORDERS_AISLE = len(aisle_orders)

#%%
aisle_products = aisle_ordersproducts.groupby(['product_id']).count()
aisle_products.columns = ['occurences']
aisle_products = aisle_products.sort_values(by=['occurences'], ascending=False)

#%%
SUPPORT_THRESHOLD = 0.001
OCCURENCES_THRESHOLD = SUPPORT_THRESHOLD * NUMBER_ORDERS_AISLE
#%%
def frequency(row):
    return row.occurences/NUMBER_ORDERS_AISLE
aisle_products['frequency'] = aisle_products.apply(frequency, axis=1)

#%%
frequent_products = aisle_products[aisle_products['frequency']>= SUPPORT_THRESHOLD]
#list_frequent_items = [set([item]) for item in frequent_products.index]
dict_frequent_items = {(i,): aisle_products.loc[i, 'occurences'] for i in frequent_products.index}

#%%
order_array = np.array(aisle_ordersproducts, dtype=object)

#%%
def get_candidates_subsets(frequent_itemsets):
    if len(frequent_itemsets) == 0:
        return[]
    k = len(frequent_itemsets[0])
    for i, pair1 in enumerate(frequent_itemsets[:-1]):
        for pair2 in frequent_itemsets[i+1:]:
            if pair1[:k-1] == pair2[:k-1]:
                yield tuple(sorted(pair1 + pair2[k-1:]))
                
#%%
def get_itemsets(order_item, candidates,k):
    
    # For each order, generate a list of items in that order
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]      
    
        # For each item list, generate item pairs, one at a time
        for itemset in combinations(item_list, k):
            candidate = tuple(sorted(itemset))
            if candidate in candidates.keys():
                yield candidate

#%%
def get_all_frequent_itemsets(order_array, frequent_items):
    flag = True
    frequent_itemsets_k = list(frequent_items.keys())
    frequent_itemsets_all = {1: frequent_items}
    frequent_orders = np.array([item for item in order_array if (item[1],) in frequent_itemsets_k])
    k = 1
    while flag:
        k += 1
        candidates = Counter(get_candidates_subsets(frequent_itemsets_k))
        C = Counter(get_itemsets(frequent_orders, candidates,k))
        C_frequent = C.copy()
        for itemset, occurences in C.items():
            if occurences < OCCURENCES_THRESHOLD:
                del C_frequent[itemset]
        frequent_itemsets_k = list(C_frequent.keys())
        if len(frequent_itemsets_k) == 0:
            flag = False
        else:
            frequent_itemsets_all[k] = {itemset: occurence for itemset, occurence in C_frequent.items()}
    return frequent_itemsets_all

#%%
F = get_all_frequent_itemsets(order_array, dict_frequent_items)

#%%
#%%effiecient apriori:
#from mlxtend.frequent_patterns import apriori
#thresholds = [0.0001, 0.0025, 0.005, 0.0075, 0.001, 0.0025, 0.005, 0.0075, 0.01]
#times = []
#max_lengths = []
#numbers = []
#for threshold in thresholds:
#    start = time.time()
#    apriori(aisle_ordersproducts, min_support=0.6)


 #%%
def rule_generation(F, thresh):
    rulelist=[]                
    for k, dic in F.items():
        for itemset in dic.keys():
            cand=[(item,) for item in itemset]
            subsets = cand
            i = 1
            while len(cand) > 0 and i < k:
                for conseq in subsets:
                    antece=[x for x in itemset if x not in conseq]
                    antece=tuple(antece)
                    confi=confget(antece,itemset)
                    if confi<thresh:
                        cand.remove(conseq)
                    else:
                        rulelist.append((antece,conseq,confi))
                
                subsets = list(get_candidates_subsets(cand))
                cand = subsets
                i +=1
    return rulelist

thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
times = []
max_lengths = []
numbers = []
for t in thresholds:
    start = time.time()          
    rules = rule_generation(F, t)
    execution_time = time.time() - start
    times.append(execution_time)
    max_lengths.append(max([len(i[1]) for i in rules]))
    numbers.append(len(rules))
    
#%%
plt.plot(thresholds, times, 'r+-')
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Confidence threshold')
plt.ylabel('Running time')
    

#%%
sorted_rules = sorted(rules, key=lambda tup: tup[2], reverse=True)
sorted_rules

top_rules = sorted_rules[:10]

def get_product_names(list_ids):
    return tuple([df_fruits.loc[id, 'product_name'] for id in list_ids])

top_rules_names = [tuple(list(map(get_product_names, rule[:2]))+ [rule[2]]) for rule in top_rules]
#%% Fast implemented

#This takes long
from pymining import itemmining, assocrules
transactions=aisle_orders['products']
item_sets = itemmining.get_relim_input(transactions)
#%%

min_supp= SUPPORT_THRESHOLD * NUMBER_ORDERS_AISLE
item_sets = itemmining.relim(item_sets, min_support=min_supp)

#%%
thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
times = []
max_lengths = []
numbers = []
for t in thresholds:
    start = time.time()          
    rules = assocrules.mine_assoc_rules(item_sets, min_support=min_supp, min_confidence=t)
    execution_time = time.time() - start
    times.append(execution_time)
    max_lengths.append(max([len(i[1]) for i in rules]))
    numbers.append(len(rules))


      
    
