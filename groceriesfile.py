# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:06:25 2020

@author: Vimal PM
"""

#importing necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

#importing the datasets using pd.read_csv function

groceries = [] 
with open("D:\\DATA SCIENCE\\Data sets\\groceries.csv","r") as f:
    groceries = f.read()



# splitting the data's using "\n"
groceries = groceries.split("\n")
#splitting each and every data's using ","  to convert to list format
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
    
all_groceries_list = []
#here taking out each and every item  from that groceries_list and assign it based on index values
all_groceries_list = [i for item in groceries_list for i in item]
#to count the data's for that importing counter function from collection module 
from collections import Counter

item_frequencies = Counter(all_groceries_list)
#sorting the data's using lambda function  to have numbers of argument
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])


# Getting the frequency of each and  every data's
frequencies = list(reversed([i[1] for i in item_frequencies]))
#sorting the data's in string fromat from high to low
items = list(reversed([i[0] for i in item_frequencies]))
#from this frequencies  i can see the data called "whole milk" and "other vagitable" are most repeated

#getting the visualization for first 11 data's from highest values using bar plot
plt.bar(height = frequencies[:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[:11]);plt.xlabel("items")
plt.ylabel("Count")


# Creating Data Frame 

groceries_series  = pd.DataFrame(pd.Series(groceries_list))
# removing the last empty transaction
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction
#Next i would like to give a name for the columns which having data's
groceries_series.columns = ["transactions"]

# creating a dummy variable or in a binary matrix format
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')
#applying the apriori
frequent_itemsets = apriori(X,min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape
#(989, 2)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.shape
#(2700, 9)

rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
           antecedents         consequents  ...  leverage  conviction
0   (other vegetables)        (whole milk)  ...  0.025394    1.214013
1         (whole milk)  (other vegetables)  ...  0.025394    1.140548
2         (rolls/buns)        (whole milk)  ...  0.009636    1.075696
3         (whole milk)        (rolls/buns)  ...  0.009636    1.048452
4             (yogurt)        (whole milk)  ...  0.020379    1.244132
5         (whole milk)            (yogurt)  ...  0.020379    1.102157
6         (whole milk)   (root vegetables)  ...  0.021056    1.101913
7    (root vegetables)        (whole milk)  ...  0.021056    1.350401
8   (other vegetables)   (root vegetables)  ...  0.026291    1.179941
9    (root vegetables)  (other vegetables)  ...  0.026291    1.426693
10            (yogurt)  (other vegetables)  ...  0.016424    1.170929
11  (other vegetables)            (yogurt)  ...  0.016424    1.109436
12        (rolls/buns)  (other vegetables)  ...  0.007013    1.049620
13  (other vegetables)        (rolls/buns)  ...  0.007013    1.046477
14    (tropical fruit)        (whole milk)  ...  0.015486    1.247252
15        (whole milk)    (tropical fruit)  ...  0.015486    1.072631
16              (soda)        (rolls/buns)  ...  0.006258    1.046003
17        (rolls/buns)              (soda)  ...  0.006258    1.042983
18    (tropical fruit)  (other vegetables)  ...  0.015589    1.225796
19  (other vegetables)    (tropical fruit)  ...  0.015589    1.098913

#Negleting the redundancy
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)
                              antecedents  ... conviction
2558                                (ham)  ...   1.190407
2023                 (whipped/sour cream)  ...   1.066171
1313                    (root vegetables)  ...   1.051406
2041  (other vegetables, root vegetables)  ...   1.101338
2206                     (tropical fruit)  ...   1.041688
776                             (berries)  ...   1.275461
1469     (whipped/sour cream, whole milk)  ...   1.192963
2434               (curd, tropical fruit)  ...   1.773680
1038             (beef, other vegetables)  ...   1.490123
1871          (domestic eggs, whole milk)  ...   1.180732
#changing the support value and lift
frequent_itemsets = apriori(X,min_support=0.02, max_len=3,use_colnames = True)
frequent_itemsets.shape

#(122, 2)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2)
rules.shape
#(20, 9)

rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True) 
                       antecedents  ... conviction
0               (other vegetables)  ...   1.179941
1                (root vegetables)  ...   1.426693
2                         (yogurt)  ...   1.132873
3                 (tropical fruit)  ...   1.193594
4               (other vegetables)  ...   1.091160
5             (whipped/sour cream)  ...   1.350565
6   (other vegetables, whole milk)  ...   1.290900
7    (root vegetables, whole milk)  ...   1.533320
8               (other vegetables)  ...   1.080555
9                (root vegetables)  ...   1.175091
10      (yogurt, other vegetables)  ...   1.528340
11            (yogurt, whole milk)  ...   1.338511
12  (other vegetables, whole milk)  ...   1.225003
13                        (yogurt)  ...   1.100890
14              (other vegetables)  ...   1.066737
15                    (whole milk)  ...   1.047905
16                        (yogurt)  ...   1.090455
17            (whipped/sour cream)  ...   1.210881
18                     (pip fruit)  ...   1.226392
19                (tropical fruit)  ...   1.147931
#Negleting the redundancy
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)
                       antecedents           consequents  ...  leverage  conviction
6   (other vegetables, whole milk)     (root vegetables)  ...  0.015026    1.290900
19                (tropical fruit)           (pip fruit)  ...  0.012499    1.147931
0               (other vegetables)     (root vegetables)  ...  0.026291    1.179941
12  (other vegetables, whole milk)              (yogurt)  ...  0.011828    1.225003
5             (whipped/sour cream)    (other vegetables)  ...  0.015006    1.350565
16                        (yogurt)  (whipped/sour cream)  ...  0.010742    1.090455
3                 (tropical fruit)              (yogurt)  ...  0.014645    1.193594