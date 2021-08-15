# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:13:07 2021

@author: LAB1-PC9
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#############################################################
data = pd.read_csv('train.csv',parse_dates = ['Occurrence Local Date Time'])
data.head()
############################################################

x=data['Reporting Agency'].value_counts()

for i in range(len(data['Reporting Agency'])):
    if 'ca' in  str(data['Reporting Agency'][i]).lower():
        data['Reporting Agency'][i]='camera'
    elif 'tr' in str(data['Reporting Agency'][i]).lower() or 'coc' in str(data['Reporting Agency'][i]).lower():
        data['Reporting Agency'][i]='Traffic'
    elif 'la' in str(data['Reporting Agency'][i]).lower():
        data['Reporting Agency'][i]='LAW'
    else:
        data['Reporting Agency'][i]='other'
        


x=data['Reporting Agency'].value_counts()
#################################################
data.set_index(data['EventId'],inplace=True)
data.drop('EventId',inplace=True,axis=1)
##################################################
x=data['Cause'].value_counts()
ind=x.index
filter1=[]
for i in range(len(x)):
    if x[i]>50:
        filter1.append(ind[i])
data=data.loc[data['Cause'].isin( filter1)]
#########################################################
x=data['road_segment_id'].value_counts()
ind=x.index
filter2=[]
for i in range(len(x)):
    if x[i]>100:
        filter2.append(ind[i])
data=data.loc[data['road_segment_id'].isin( filter2)]

################################################

x=data['Subcause'].value_counts()
ind=x.index
filter3=[]
for i in range(len(x)):
    if x[i]>25:
        filter3.append(ind[i])
data=data.loc[data['Subcause'].isin( filter3)]

####################################################
x=data['Status'].value_counts()
data.drop('Status',inplace=True,axis=1)

#####################################################
data.to_csv('after pro.csv')

####################################################
data.dropna(inplace=True)
###################
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
data=data
data.iloc[:,1:-3]=data.iloc[:,1:-3].apply(LabelEncoder().fit_transform)
###################################################
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
#######################################
'''from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x, y)
np.set_printoptions(precision=3)
print(fit.scores_)'''
#######################
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
############################################


