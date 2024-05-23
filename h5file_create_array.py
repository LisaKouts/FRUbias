import numpy as np
from tables import *
from tqdm import tqdm

class Writearray:
    '''
    This class computes the similarity matrix based on which the fuzzy rough sets are later computed
    '''

    def __init__(self, df, alpha, variable):
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
        numeric : list of bool
            each entry in the list represents a the data type of a feature in the dataset. if True, this 
            feature is numeric, if False, the feature is nominal. it is needed to compute the Heterogeneous
            Manhattan Overal distance metric
        
        nominal : list of bool
            each entry in the list represents a the data type of a feature in the dataset. if True, this 
            feature is nominal, if False, the feature is numeric. it is needed to compute the Heterogeneous
            Manhattan Overal distance metric
        '''
        self.variable = variable
        self.numeric = [False if df[col].dtype == 'object' else True for col in df]
        self.nominal = [True if df[col].dtype == 'object' else False for col in df]

        num = df.loc[:,self.numeric]
        scaled=np.subtract(num,np.min(num,axis=0))/np.subtract(np.max(num,axis=0),np.min(num,axis=0))
        df.loc[:,df.columns[self.numeric]] = scaled.round(3).astype('float32')

        self.df = df.values
        self.alpha = alpha

    def sim_array(self, h5file, group, hide_progress = False):
       for instance in tqdm(range(0,len(self.df)), desc=self.variable+' building similarity matrix', disable=hide_progress):
          sim = self.similarity(instance)
          h5file.create_array(group, 'col'+str(instance), sim, 'Distance instance '+str(instance))

    def similarity(self, i):
        d = np.sum(np.abs(np.subtract(self.df[i][self.numeric], self.df[:,self.numeric])), axis=1) + np.sum(self.df[i][self.nominal] != self.df[:,self.nominal],axis=1)
        return np.exp(-self.alpha * d.astype('float32'))
    
import sys
if __name__=="__main__":
  args = Writearray(sys.argv[1], sys.argv[2]).sim_array(sys.argv[3], sys.argv[4], sys.argv[5])
  print("In mymodule:",args)