# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:34:34 2018

@author: caspe
"""
import numpy as np
import itertools
def findsubsets(S,m):
    return list(itertools.combinations(S, m))
def tempfix(f,key):
    r=np.array(f)
    a=np.zeros(shape=(len(f),key))
    r1=np.zeros(shape=(len(f),1))
    b=[]
    for k in range(np.shape(a)[0]):
        r1[k]=r[k][1]
        b.append(r[k][0])
        for i in range(np.shape(a)[1]):
            a[k,i]=r[k,0][i]
    
    return(a,r1,b)
def ConfGet(superset,subset,superitem,subitem,occsuper,occsub):
    #Antecedent +consequent, antecedent, frequent itemset for superset, freq items for ant. ,
    #occurence for ant + cons,  
    antsub=occsub[subitem.index(subset)][0]
    conssuper=occsuper[superitem.index(superset)][0]
    conf=conssuper/float(antsub)
    return conf

def ConfGet(superset,subset,superitem,subitem,occsuper,occsub):
    #Antecedent +consequent, antecedent, frequent itemset for superset, freq items for ant. ,
    #occurence for ant + cons,  
    antsub=occsub[subitem.index(subset)][0]
    conssuper=occsuper[superitem.index(superset)][0]
    conf=conssuper/float(antsub)
    return conf

def confget(ante,conseq):
    #Antecedent +consequent, antecedent, frequent itemset for superset, freq items for ant. ,
    #occurence for ant + cons, 
    rulelen=len(conseq)
    antsup=F[len(ante)][ante]
    conseqsup=F[rulelen][conseq]
    conf=conseqsup/float(antsup)
    return conf

def ConfGet1(superset,subset,superitem,occsuper):
    
    antsub=o[0].loc[subset]
    conssuper=occsuper[superitem.index(superset)][0]
    conf=conssuper/float(antsub)
    return conf

def updateitems(ants,conseq,count):
    conslist=[]
    antlist=[]
    if type(conseq[0]==int):
        for k in range(len(ants)):
            for s in range(k+1,len(ants)):
                if len(np.intersect1d(list(ants[k]),list(ants[s])))==len(ants[k])-1:
                    newcons= (conseq[k], np.setdiff1d(ants[k],ants[s])[0])
                    newant=tuple(np.intersect1d(ants[k],ants[s]),*newcons)
                    antlist.append(newant)
                    conslist.append(newcons)    
    else:
        for k in range(len(ants)):
            for s in range(k+1,len(ants)):
                if len(np.intersect1d(list(ants[k]),list(ants[s])))==len(ants[k])-1:
                    
                    newcons= (*conseq[k], np.setdiff1d(ants[k],ants[s])[0])
                    newant=tuple(np.intersect1d(ants[k],ants[s]),*newcons)
                    antlist.append(newant)
                    conslist.append(newcons)
    return (antlist,conslist)
                    

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
list_frequent_items = [[i] for i in frequent_products.index]

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
    frequent_itemsets_k = frequent_items
    frequent_itemsets_all = {1: frequent_items}
    frequent_orders = np.array([item for item in order_array if [item[1]] in frequent_items])
    k = 1
    while flag:
        k += 1
        print(k)
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
            frequent_itemsets_all[k] = [(itemset, occurence) for itemset, occurence in C_frequent.items()]
    return frequent_itemsets_all

#%%
F = get_all_frequent_itemsets(order_array, list_frequent_items)



#%%
maxrulelen=len(F)
r2=np.array(F[2])
r3=np.array(F[3])
r4=np.array(F[4])
#%%

#%%
#r1,o1,b1=tempfix(F[1],1)
b1=F[1]
b1=[tuple(l) for l in b1]

r2,o2,b2=tempfix(F[2],2)
r3,o3,b3=tempfix(F[3],3)
r4,o4,b4=tempfix(F[4],4)
o=[aisle_products['occurences'],o2,o3,o4]
b=[b1,b2,b3,b4]
threshold=0.25      
#%%
    


#%%   
 
thresh=0.05
rulelist=[]
        #%%
for itemset,occ in F[4].items():
    cand=list(itemset)
    for conseq in itemset:
        antece=list(itemset)
        antece.remove(conseq)
        antece=tuple(antece)
        confi=confget(antece,tuple(itemset))
        if confi<thresh:
            cand.remove(conseq)
        else:
            rulelist.append((antece,(conseq),confi))

    pairs=findsubsets(cand,2)
    cand=pairs
    
    for conseq in pairs:
        antece=list(itemset)
        antece=[x for x in antece if x not in conseq]
        antece=tuple(antece)
        confi=confget(antece,itemset)
        if confi<thresh:
            cand.remove(conseq)
        else:
            rulelist.append((antece,(conseq),confi))
                #print(antece,conseq)
#%%effiecient apriori:
from mlxtend.frequent_patterns import apriori
apriori(aisle_ordersproducts, min_support=0.6)


 #%%
#def update(items,consequents,count)
maxrule=4
rulelist=[]                
for k, dic in F.items():
    for itemset, occ in dic.items():
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
                    rulelist.append((antece,(conseq,),confi))
            
            subsets = list(get_candidates_subsets(cand))
            cand = subsets
            i +=1
      






    
