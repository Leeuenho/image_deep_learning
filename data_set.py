# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 15:43:40 2021

@author: pixee
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

path = 'D:/한림대학교/4학년/X-ray_data_set/CheXpert-v1.0-small/'

train_data = pd.read_csv(path+'CheXpert-v1.0-small//train.csv')
disease = ['Cardiomegaly']


for _,name in enumerate(disease):
    globals()['non_{}'.format(name)] = pd.DataFrame(columns = ("Path", 'Ans'))
    globals()['{}'.format(name)] = pd.DataFrame(columns = ["Path", 'Ans'])
    

     
for i in range(len(disease)):
    train_data = train_data[['Path', disease[i]]] #경로와 질병 불러오기
    
    #정상(0) 혹은 질병(1) 이면 저장, 나머지는 패스
    
    for ii in range(len(train_data)):
        
        try : 

            if((train_data[disease[i]][ii]) == 0):
                globals()['non_{}'.format(disease[i])] = globals()['non_{}'.format(disease[i])].append({"Path" : (train_data['Path'][ii]), 
                                                         'Ans': (train_data[disease[i]][ii])}, ignore_index = True)
            if((train_data[disease[i]][ii]) == 1):
                globals()['{}'.format(disease[i])] = globals()['{}'.format(disease[i])].append({"Path" : (train_data['Path'][ii]), 
                                                         'Ans': (train_data[disease[i]][ii])}, ignore_index = True)
                
        except :
            continue
                     

