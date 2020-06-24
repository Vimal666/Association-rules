# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:58:41 2020

@author: intel
"""

#importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

#loading the datasets using pd.read_csv function

book=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//Assignment8//book.csv")
book.columns
#['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
     #  'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence'],)
     


#going for visualizations
import seaborn as sns      
sns.pairplot(book.iloc[:,0:10])
# removing the last empty transaction
books = book.iloc[:1999,:]
#applying the apriori
book_itemset = apriori(books,min_support=0.005, max_len=3,use_colnames = True)
book_itemset.shape
#(224, 2)
# Most Frequent item sets based on support 
book_itemset.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = book_itemset.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),book_itemset.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(book_itemset, metric="lift", min_threshold=1)
rules.shape
# rules.shape
#Out[107]: (1054, 9)

rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
   antecedents consequents  antecedent support  ...      lift  leverage  conviction
0    (CookBks)  (ChildBks)            0.431216  ...  1.403476  0.073633    1.420547
1   (ChildBks)   (CookBks)            0.423212  ...  1.403476  0.073633    1.440693
2    (GeogBks)  (ChildBks)            0.276138  ...  1.669429  0.078233    1.965353
3   (ChildBks)   (GeogBks)            0.423212  ...  1.669429  0.078233    1.342954
4    (CookBks)   (GeogBks)            0.431216  ...  1.617436  0.073521    1.308111
5    (GeogBks)   (CookBks)            0.276138  ...  1.617436  0.073521    1.880054
6   (DoItYBks)   (CookBks)            0.282141  ...  1.541905  0.065930    1.697325
7    (CookBks)  (DoItYBks)            0.431216  ...  1.541905  0.065930    1.270625
8   (DoItYBks)  (ChildBks)            0.282141  ...  1.541740  0.064687    1.659738
9   (ChildBks)  (DoItYBks)            0.423212  ...  1.541740  0.064687    1.270520
10   (CookBks)    (ArtBks)            0.431216  ...  1.606960  0.063109    1.238928
11    (ArtBks)   (CookBks)            0.241121  ...  1.606960  0.063109    1.852392
12  (YouthBks)  (ChildBks)            0.247624  ...  1.575256  0.060285    1.730365
13  (ChildBks)  (YouthBks)            0.423212  ...  1.575256  0.060285    1.233547
14    (ArtBks)  (ChildBks)            0.241121  ...  1.593231  0.060536    1.770777
15  (ChildBks)    (ArtBks)            0.423212  ...  1.593231  0.060536    1.232269
16   (CookBks)  (YouthBks)            0.431216  ...  1.517908  0.055302    1.205480
17  (YouthBks)   (CookBks)            0.247624  ...  1.517908  0.055302    1.646481
18   (CookBks)    (RefBks)            0.431216  ...  1.648724  0.060034    1.215455
19    (RefBks)   (CookBks)            0.214607  ...  1.648724  0.060034    1.967811

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
antecedents            consequents  ...  leverage  conviction
751             (ItalAtlas)      (RefBks, ItalArt)  ...  0.015768    1.768762
742     (ItalAtlas, ArtBks)              (ItalArt)  ...  0.015634   11.417709
350      (ItalCook, ArtBks)              (ItalArt)  ...  0.034776    2.829388
561      (ItalCook, RefBks)            (ItalAtlas)  ...  0.021289    1.905474
920    (ItalAtlas, GeogBks)              (ItalArt)  ...  0.010511    2.167250
896   (ItalCook, ItalAtlas)              (ItalArt)  ...  0.011390    2.084185
963    (ItalCook, Florence)              (ItalArt)  ...  0.008655    2.081353
812     (ItalArt, ChildBks)            (ItalAtlas)  ...  0.013174    1.612434
968   (DoItYBks, ItalAtlas)              (ItalArt)  ...  0.008582    1.902951
1009              (ItalArt)  (ItalAtlas, YouthBks)  ...  0.007655    1.191271

#changing the support value and lift
book_itemlist2 = apriori(book,min_support=0.02, max_len=3,use_colnames = True)
book_itemlist2.shape
#(151, 2)
# Most Frequent item sets based on support 
book_itemlist2.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = book_itemlist2.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('book_itemlist2');plt.ylabel('support')

rules = association_rules(book_itemlist2, metric="lift", min_threshold=2)
rules.shape
#(398, 9)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True) 
             antecedents           consequents  ...  leverage  conviction
0    (CookBks, ChildBks)             (GeogBks)  ...  0.078844    1.740319
1              (GeogBks)   (CookBks, ChildBks)  ...  0.078844    1.623273
2    (CookBks, ChildBks)            (DoItYBks)  ...  0.073808    1.670982
3             (DoItYBks)   (CookBks, ChildBks)  ...  0.073808    1.542706
4    (CookBks, ChildBks)            (YouthBks)  ...  0.065640    1.516850
5             (YouthBks)   (CookBks, ChildBks)  ...  0.065640    1.553924
6    (CookBks, ChildBks)              (ArtBks)  ...  0.064804    1.500417
7               (ArtBks)   (CookBks, ChildBks)  ...  0.064804    1.565974
8    (CookBks, ChildBks)              (RefBks)  ...  0.067588    1.506277
9               (RefBks)   (CookBks, ChildBks)  ...  0.067588    1.734652
10            (ItalCook)             (CookBks)  ...  0.064582         inf
11             (CookBks)            (ItalCook)  ...  0.064582    1.203406
12   (DoItYBks, CookBks)             (GeogBks)  ...  0.056750    1.718354
13             (GeogBks)   (DoItYBks, CookBks)  ...  0.056750    1.338806
14  (DoItYBks, ChildBks)             (GeogBks)  ...  0.053716    1.675673
15             (GeogBks)  (DoItYBks, ChildBks)  ...  0.053716    1.313213
16     (CookBks, ArtBks)             (GeogBks)  ...  0.057408    1.904063
17    (CookBks, GeogBks)              (ArtBks)  ...  0.057107    1.641657
18              (ArtBks)    (CookBks, GeogBks)  ...  0.057107    1.415327
19             (GeogBks)     (CookBks, ArtBks)  ...  0.057408    1.332800
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
212   (ItalCook, ArtBks)             (ItalArt)  ...  0.034760    2.829461
361   (ItalCook, RefBks)           (ItalAtlas)  ...  0.021279    1.905511
330            (ItalArt)  (DoItYBks, ItalCook)  ...  0.022163    1.943096
208   (CookBks, ItalArt)            (ItalCook)  ...  0.032847   10.384714
334  (ItalCook, GeogBks)             (ItalArt)  ...  0.020896    1.522400
357           (ItalCook)  (ItalAtlas, CookBks)  ...  0.019765    1.218401
301  (ItalArt, ChildBks)            (ItalCook)  ...  0.024414    4.255200
218           (ItalCook)             (ItalArt)  ...  0.031995    1.420990
396          (ItalAtlas)  (ItalCook, ChildBks)  ...  0.016855    1.991471
349          (ItalAtlas)            (ItalCook)  ...  0.018800    2.342893