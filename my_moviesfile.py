# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:25:48 2020

@author: Vimal PM
"""
#importing the necessary libraies 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
#loading the dataset using pd.read_csv function
my_movie=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\Assignment8\my_movie.csv")
my_movie.shape
#(10, 10)
my_movie.columns
#Index(['Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
  #     'LOTR2', 'Harry Potter2', 'LOTR', 'Braveheart', 'Green Mile'],
  #    dtype='object')

#going for pairplot for every variables inside my dataset
sns.pairplot(my_movie.iloc[:,0:9])

#applying the apriori
movie_itemset=apriori(my_movie,min_support=0.005,max_len=3,use_colnames=True)
movie_itemset.shape
#(46, 2)
#most frequent item based on support
plt.bar(x=list(range(1,11)),height=movie_itemset.support[1:11],color="rgmyk");plt.xticks(list(range(1,11)),movie_itemset.itemset[1:11])
#applying the association rule
rules=association_rules(movie_itemset,metric="lift",min_threshold=1)
rules.shape
# (124, 9)
print(rules.lift)
0      1.190476
1      1.190476
2      1.111111
3      1.111111
4      1.666667
  
119    5.000000
120    5.000000
121    5.000000
122    5.000000
123    5.000000
#neglecting the redundancy
def to_list(i):
    return(sorted(list(i)))
    
max=rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)    
max=max.apply(sorted)
rules_sets = list(max)
unique_rules_sets=[list (m) for m in set(tuple(i)for i in rules_sets)]
index_rules=[]
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

#getting the rules without redundancy
rules_no_redundancy = rules.iloc[index_rules,:]
#sorting the data with respect to lift and getting top 10 rules
rules_no_redundancy.sort_values("lift",ascending=False).head
<bound method NDFrame.head of                       antecedents      consequents  ...  leverage  conviction
82            (LOTR, Sixth Sense)     (Green Mile)  ...      0.08         inf
54           (LOTR2, Sixth Sense)          (LOTR1)  ...      0.08         inf
106   (Harry Potter1, Green Mile)          (LOTR1)  ...      0.08         inf
48   (Harry Potter1, Sixth Sense)          (LOTR1)  ...      0.08         inf
94              (LOTR, Gladiator)     (Green Mile)  ...      0.08         inf
22                (Harry Potter2)  (Harry Potter1)  ...      0.08         inf
70   (Harry Potter1, Sixth Sense)     (Green Mile)  ...      0.08         inf
76           (LOTR2, Sixth Sense)     (Green Mile)  ...      0.08         inf
30                         (LOTR)     (Green Mile)  ...      0.08         inf
100        (LOTR2, Harry Potter1)          (LOTR1)  ...      0.08         inf
112           (LOTR2, Green Mile)          (LOTR1)  ...      0.08         inf
16                        (LOTR2)          (LOTR1)  ...      0.16         inf
118        (LOTR2, Harry Potter1)     (Green Mile)  ...      0.08         inf
28                        (LOTR2)     (Green Mile)  ...      0.06        1.60
24                (Harry Potter1)     (Green Mile)  ...      0.06        1.60
20                        (LOTR2)  (Harry Potter1)  ...      0.06        1.60
58      (Sixth Sense, Green Mile)          (LOTR1)  ...      0.06        1.60
18                   (Green Mile)          (LOTR1)  ...      0.06        1.60
14                (Harry Potter1)          (LOTR1)  ...      0.06        1.60
4                          (LOTR)    (Sixth Sense)  ...      0.04         inf
6                   (Sixth Sense)     (Green Mile)  ...      0.08        1.20
38              (LOTR, Gladiator)    (Sixth Sense)  ...      0.04         inf
26                      (Patriot)     (Braveheart)  ...      0.04        1.08
88           (Gladiator, Patriot)     (Braveheart)  ...      0.04        1.08
64         (LOTR2, Harry Potter1)    (Sixth Sense)  ...      0.04         inf
8                     (Gladiator)        (Patriot)  ...      0.18        2.80
10                         (LOTR)      (Gladiator)  ...      0.03         inf
12                    (Gladiator)     (Braveheart)  ...      0.03        1.05
0                     (Gladiator)    (Sixth Sense)  ...      0.08        1.40
2                       (Patriot)    (Sixth Sense)  ...      0.04        1.20
32           (Gladiator, Patriot)    (Sixth Sense)  ...      0.04        1.20
44       (Gladiator, Sixth Sense)     (Green Mile)  ...      0.00        1.00

#changing the suppot value and min threshold
#applying the apriori
movie_itemset=apriori(my_movie,min_support=0.002,max_len=3,use_colnames=True)
movie_itemset.shape
#(46, 2)
#most frequent item based on support
plt.bar(x=list(range(1,11)),height=movie_itemset.support[1:11],color="rgmyk");plt.xticks(list(range(1,11)),movie_itemset.itemset[1:11])
#applying the association rule
rules=association_rules(movie_itemset,metric="lift",min_threshold=2)
rules.shape
# (74, 9)
print(rules.lift)
0     2.5
1     2.5
2     5.0
3     5.0
4     2.5

69    5.0
70    5.0
71    5.0
72    5.0
73    5.0
#neglecting the redundancy
def to_list(i):
    return(sorted(list(i)))
    
max=rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)    
max=max.apply(sorted)
rules_sets = list(max)
unique_rules_sets=[list (m) for m in set(tuple(i)for i in rules_sets)]
index_rules=[]
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

#getting the rules without redundancy
rules_no_redundancy = rules.iloc[index_rules,:]
#sorting the data with respect to lift and getting top 10 rules
rules_no_redundancy.sort_values("lift",ascending=False).head
<bound method NDFrame.head of                         antecedents      consequents  ...  leverage  conviction
8                (Harry Potter2)  (Harry Potter1)  ...      0.08         inf
30          (LOTR2, Sixth Sense)  (Harry Potter1)  ...      0.08         inf
18  (Harry Potter1, Sixth Sense)          (LOTR1)  ...      0.08         inf
34  (Harry Potter1, Sixth Sense)     (Green Mile)  ...      0.08         inf
68        (LOTR2, Harry Potter1)     (Green Mile)  ...      0.08         inf
38          (LOTR2, Sixth Sense)     (Green Mile)  ...      0.08         inf
14                        (LOTR)     (Green Mile)  ...      0.08         inf
22          (LOTR2, Sixth Sense)          (LOTR1)  ...      0.08         inf
46             (LOTR, Gladiator)     (Green Mile)  ...      0.08         inf
2                        (LOTR2)          (LOTR1)  ...      0.16         inf
62           (LOTR2, Green Mile)          (LOTR1)  ...      0.08         inf
50        (LOTR2, Harry Potter1)          (LOTR1)  ...      0.08         inf
42           (LOTR, Sixth Sense)     (Green Mile)  ...      0.08         inf
56   (Harry Potter1, Green Mile)          (LOTR1)  ...      0.08         inf
12                       (LOTR2)     (Green Mile)  ...      0.06       1.600
0                (Harry Potter1)          (LOTR1)  ...      0.06       1.600
6                        (LOTR2)  (Harry Potter1)  ...      0.06       1.600
26     (Sixth Sense, Green Mile)          (LOTR1)  ...      0.06       1.600
4                   (Green Mile)          (LOTR1)  ...      0.06       1.600
10               (Harry Potter1)     (Green Mile)  ...      0.06       1.600
16      (Gladiator, Sixth Sense)           (LOTR)  ...      0.05       1.125