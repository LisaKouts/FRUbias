import numpy as np
from tables import *
from tqdm import tqdm
from itertools import product
import pandas as pd

class Writearray:
    '''
    This class computes the similarity matrix based on which the fuzzy rough sets are later computed
    '''

    def __init__(self, df, alpha, variable, method, target):
        '''
        Preprocessing steps, the numeric variables are normalized in the interval [0,1]

        Attributes
        ----------
        df : pandas DataFrame
            a dataset consisting of several variables, note that no decision / outcome feature should be present
        
        alpha : float
            this variable in the interval [0,1] helps separating the fuzzy-rough regions, 
            the larger it is the more separated the regions
        
        variable : string
            name of variable that is uppressed

        Returns
        -------
        Creates a h5file in the specified location, the file contains the square distance matrix
        '''
        self.variable = variable
        self.numeric = [False if df[col].dtype == 'object' else True for col in df]
        self.nominal = [True if df[col].dtype == 'object' else False for col in df]
        self.distance = method
        self.alpha = alpha

        # normalize numeric features
        num = df.loc[:,self.numeric]
        scaled=np.subtract(num,np.min(num,axis=0))/np.subtract(np.max(num,axis=0),np.min(num,axis=0))
        df.loc[:,df.columns[self.numeric]] = scaled.round(3).astype('float32')

        if self.distance == 'HVDM':
           beta_f_x_k, beta_f_x, nominal_columns, K = self.nominal_probabilities(df, target)
           self.hvdm_nom = self.HVDM_nominal(df, beta_f_x_k, beta_f_x, nominal_columns, K)
        
        self.df = df.values
        

    def sim_array(self, h5file, group, hide_progress = False):
       for instance in tqdm(range(0,len(self.df)), desc=self.variable+' building similarity matrix', disable=hide_progress, miniters=1000):
          sim = self.similarity(instance)
          h5file.create_array(group, 'col'+str(instance), sim, 'Distance instance '+str(instance))

    def similarity(self, i):
        '''
        See here for the equations: https://jair.org/index.php/jair/article/view/10182/24168
        '''
        if self.distance == 'HMOM':
           d = np.sum(np.abs(np.subtract(self.df[i][self.numeric], self.df[:,self.numeric])), axis=1) + np.sum(self.df[i][self.nominal] != self.df[:,self.nominal],axis=1)
        
        elif self.distance == 'HEOM':
           d_num = np.sum((np.subtract(self.df[i][self.numeric], self.df[:,self.numeric])**2), axis=1)
           d_nom = np.sum(self.df[i][self.nominal] != self.df[:,self.nominal],axis=1)
           d = (d_num + d_nom)**0.5

        elif self.distance == 'HVDM':
            d = (np.sum(np.subtract(self.df[i,self.numeric].T,self.df[:,self.numeric])**2,axis=1) + np.sum(self.hvdm_nom.loc[i],axis=1))**0.5
            d = d.values

        else:
           print('Distance function not specified')
        
        return np.exp(-self.alpha * d.astype('float32'))
    
    def nominal_probabilities(self, df1, target_values):
        beta_f_x_k = {}
        beta_f_x = {}

        # select nominal values and target feature
        nom_df = df1[df1.columns[self.nominal]].copy()
        nom_df['y'] = target_values.values

        for i in df1.columns[self.nominal]:
            beta_f_x_k[i] = pd.DataFrame(product(list(nom_df[i].unique()), list(nom_df['y'].unique())), columns=[i,'y'])
    
            # add empty frequency column
            beta_f_x_k[i]['frequency'] = np.zeros((len(beta_f_x_k[i])))

            # set index
            beta_f_x_k[i] = beta_f_x_k[i].set_index([i,'y'])

            # populate frequency column with number of occurances of category x output class
            freq_f_k = nom_df[[i,'y']].groupby([i,'y'])['y'].count()
            for j in range(len(freq_f_k)):
                beta_f_x_k[i].loc[(freq_f_k.index[j])] = freq_f_k.iloc[j]
            
            # number of instances per category in the nominal feature
            beta_f_x[i] = nom_df[i].value_counts()
        
        return beta_f_x_k, beta_f_x, df1.columns[self.nominal], len(target_values.unique())
    
    def HVDM_nominal(self, df, beta_f_x_k, beta_f_x, nom_cols, K):

        nom_d = np.empty((1,len(nom_cols)+2)) # two extra columns for the indices of x and y
        cols = ['x','y']
        [cols.append(f) for f in nom_cols]

        for x in range(len(df)):
            for y in range(len(df)):
                tau_nominal_F = [x,y]
                for f in nom_cols:
                    tau_nominal = 0
                    for k in range(K):
                        b_f_x_k = beta_f_x_k[f].loc[(df.loc[x,f],k)].values[0]
                        b_f_y_k = beta_f_x_k[f].loc[(df.loc[y,f],k)].values[0]
                        b_f_x = beta_f_x[f][df.loc[x,f]]
                        b_f_y = beta_f_x[f][df.loc[y,f]]
                        tau_nominal = tau_nominal + ((b_f_x_k/b_f_x) - (b_f_y_k/b_f_y))**2
                    tau_nominal = (tau_nominal / K)
                    tau_nominal_F.append(tau_nominal)
                nom_d = np.append(nom_d,[tau_nominal_F],axis=0)
        nom_d = pd.DataFrame(nom_d[1:],columns=cols) # first row is removed since it was only to initialize
        nom_d = nom_d.set_index(['x','y'])

        return nom_d
    
import sys
if __name__=="__main__":
  args = Writearray(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]).sim_array(sys.argv[5], sys.argv[6], sys.argv[7])
  print("In mymodule:",args)