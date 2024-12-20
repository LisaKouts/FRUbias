from tables import *
import re
import pandas as pd
from .FRSpy import FRSpy
from .h5file_create_array import Writearray
import h5py
import pickle
import os

class FRU:

  def __init__(self, df, path):
    '''
    Attributes
    ------------
    df : pandas DataFrame
      Contains all variables in the dataset plus the decision variable
 
    path : str
      path to file where the membership values to fuzzy-rough regions should be saved in pickle form. Pickle format is used for storage efficiency in case of large datasets
      can also be empty and the file will be saved in the current directory
    '''


    self.df = df
    self.path = path

  def membership_values(self, 
                        columns, 
                        target,
                        h5_name, 
                        similarity = 'HMOM', 
                        alpha = 0.5, 
                        implication = 'Luka',
                        conjunction = 'Luka',
                        frs_method = 'Inuiguchi',
                        complete = False, 
                        hide_progress = False):
    '''
    A function that computes the membership values to fuzzy-rough regions for larger datasets. It computes a similarity matrix and stores it in a h5 file. 
    It also computes the fuzzy-rough regions and saves them in a pickle file.

    Attributes
    ------------
    columns : list of strings
      a list of the column names to be suppressed or removed from the dataset. after removal, the fuzzy-rough
      set regions will be computed using the rest of the data
    
    target : string
      name of target variable, outcome, label
      
    h5_name : str
      name of the h5 file where the similarity values are saved
    
    similarity: string
      Heterogeneus distance functions able to process both numeric and nominal features
      Cane be either 'HMOM', 'HEOM' or 'HVDM'; see journal paper Fuzzy-Rough Cognitive Networks (Nápoles et al., 2018) for the equations

    alpha: float
      parameter to separate the fuzzy-rough regions

    implication: string
      fuzzy implication, options are 'Luka', 'Godel', 'Fodor', 'Goguen' (see publication for details)

    conjunction: string
      fuzzy conjunction, options are 'Standard', 'Algebraic', 'Luka', 'Drastic' (see publication for details)

    frs_method: string
      definition of fuzzy-rough sets, options are either Inuiguchi or Radzikowska

    complete: boolean
      if True, the membership values to the fuzzy-rought regions are created using the complete set of data. 
      if this object already exists, you can set it to False to save time if dealing with large dataset
    
    '''

    # rename the column names cause they contain illegal characters and h5 cannot index columns properly
    self.df.columns = list(map(lambda x: re.sub('=<', 'less',x), list(self.df.columns)))
    self.df.columns = list(map(lambda x: re.sub('=>', 'more',x), list(self.df.columns)))
    self.df.columns = list(map(lambda x: re.sub('/|\(|\)|>|<|=| ', '',x), list(self.df.columns)))

    # fr values to fuzzy rough regions
    membership = pd.get_dummies(self.df[target])
    file_name = os.path.join(self.path, "matrix_"+h5_name+".h5")

    if not complete:
      h5file = open_file(file_name, mode="w", title=h5_name) # create h5 file to store distance matrix

      group = h5file.create_group("/", 'full', 'Distances after removing full') # full
      Writearray(self.df[columns], alpha, 'full', method=similarity, target=self.df[target]).sim_array(h5file = h5file, group = group, hide_progress = hide_progress)

      h5file.close()

      h5file = open_file(file_name, mode="r")
      frregions = FRSpy(membership, implication = implication, conjunction = conjunction, frs_method = frs_method).regions(file_name,'full', hide_progress = hide_progress)

      file_name_pickle = os.path.join(self.path, 'full_mem.pickle')
      with open(file_name_pickle, 'wb') as handle:
        pickle.dump(frregions, handle, protocol=pickle.HIGHEST_PROTOCOL)
      h5file.close()

      h5file = h5py.File(file_name, mode="a")
      del h5file['full']
      h5file.close()

    for s_attr in columns:
      # if hide_progress:
      #   print(s_attr)

      h5file = open_file(file_name, mode="a")
      dataset = self.df[columns].drop(s_attr, axis=1) # remove protected
      group = h5file.create_group("/", s_attr, 'Distances after removing '+s_attr)
      Writearray(dataset, alpha, s_attr, method=similarity, target=self.df[target]).sim_array(h5file = h5file, group = group, hide_progress = hide_progress)

      h5file.close()
      h5file = open_file(file_name, mode="r")

      frregions = FRSpy(membership, implication = implication, conjunction = conjunction, frs_method = frs_method).regions(file_name,s_attr, hide_progress = hide_progress)
      file_name_pickle = os.path.join(self.path, s_attr+'_mem.pickle')
      with open(file_name_pickle, 'wb') as handle:
        pickle.dump(frregions, handle, protocol=pickle.HIGHEST_PROTOCOL)
      h5file.close()

      h5file = h5py.File(file_name, mode="a")
      del h5file[s_attr]
      h5file.close()

  def load_membership_values(self):
    '''
    Loads the membership values to fuzzy-rough regions for larger datasets from the pickle files. It should be called after function membership_values()

    Returns
    ------------
    Dictionary object: each entry in the dictionary contains the membership values to the fuzzy rough regions of each pickle file in the folder
    The values are stored in numpy arrays for each decision class 
    '''
    mem_dic = {}
    for root, _, files in os.walk(self.path):
        for name in files:
            if 'pickle' in name:
                att_name = re.findall(r"(.*?)_mem.pickle", name)[0]
                file_path_name = os.path.join(root, name) 
                with open(file_path_name, 'rb') as handle: 
                    mem_dic_att = pickle.load(handle)
                    mem_dic[att_name] = mem_dic_att

    return mem_dic


import sys
if __name__=="__main__":
  args = FRU(sys.argv).membership_values(sys.argv)
  print("In mymodule:",args)