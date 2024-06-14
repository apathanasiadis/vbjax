import numpy as np
import pandas as pd
import pickle
import itertools
from pathlib import Path
import os
import sys

### pickles

def pickle_sth(path, option, object=None):
    '''
    save or load a pickled file.

    Inputs
    -----------------
    path: str
        string that tells the full path (file name and extension, i.e. .pickle)
        to save there the object
    option: str
        input either 'save', or 'load'
    object: a variable
        default = None.
    Outputs
    -----------------
    if option=='save': it just saves the file and prints success
    elif option=='load': it loads the file to the `object` variable
    '''
    if option == 'save':
        with open(path, 'wb') as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('file was successfully saved to the predetermined path')
    if option == 'load':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            return obj
        

### lists utils
        
def liststuple(number):
    b = [[] for _ in range(number)]
    return tuple(b)

def listoflists(length, number):
    '''
    inputs:
        length = length of parent list, i.e. how many nested lists
        number = how many parent lists
    outputs:
        tuple of lists
    '''
    if number == 1:
        return [[] for _ in range(length)]
    else:
        a = []
        for i in range(number):
            a.append([[] for _ in range(length)])
        return tuple(a)

def listoflists_dyn(length_list, number):
    a = []
    for i in range(number):
        length = length_list[i]
        a.append([[] for _ in range(length)])
    return tuple(a)

def max_nested_length(list_name, nestedarrays=False):
    '''
    input:
        list_name: a list of lists
    output:
        maximum length number of nested lists
    '''
    if nestedarrays == False:
        return max([len(list_name[i]) for i in range(len(list_name))])
    else:
        return max([list_name[i].size for i in range(len(list_name))])

def append_(times, what):
    list_name = []
    for i in range(times):
        list_name.append(what)
    return list_name

def flatten_list(xss):
    return list(itertools.chain.from_iterable(xss))


### pd dataframes

def indexed_df(data, index):
    '''
    inputs:
        data: dataset
        index: index variable of dataset
    output:
        dataframe with index
    '''
    df = pd.DataFrame(data, index = index)
    return df


### numpy extenstions

def mat_extension(mat, desired_length):
    '''
    matrix extension by replicating the last row and padding with zeros the rest of elements
    '''
    l = desired_length - mat.shape[0]
    aa = np.zeros((desired_length,desired_length))
    aa[:mat.shape[0], :mat.shape[1]] = mat
    for i in range(l):
        aa[mat.shape[0]+i,:mat.shape[1]+i+1] = np.append(np.zeros(i+1), mat[:,-1])
        aa[:mat.shape[0]+i+1,mat.shape[1]+i] = np.append(np.zeros(i+1), mat[:,-1])

    return aa

def zero_the_most_frequent(labels, classes2 = False):
    '''
    Inputs:
    labels: labels from clusters, np.array, same shape with your data
    classes2: boolean, if True, labels are returned with only 0s and 1s

    Outputs:
    labels: updated labels, where the most frequent label takes the value 0

    '''
    upd_labels = labels.copy()
    mostfrequentlabel = np.bincount(upd_labels.astype(int)).argmax()
    if classes2 == True:
        upd_labels[upd_labels != mostfrequentlabel] = 99
        upd_labels[upd_labels == mostfrequentlabel] = 0
        upd_labels[upd_labels != 0] = 1
    else:
        mfl_idx = np.where(upd_labels == mostfrequentlabel)[0]
        zero_idx = np.where(upd_labels == 0)[0]
        upd_labels[zero_idx] = 99
        upd_labels[mfl_idx] = 0
        upd_labels[zero_idx] = mostfrequentlabel

    return upd_labels


### dictionaries

def get_key_given_value(dictionary, value):
    '''given a value of a dictionary, find the key'''
    return next((k for k, v in dictionary.items() if v == value), None)


### low-level functions

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    except FileExistsError:
        print(f"Folder already exists at {folder_path}")

def add_modules_folder(parent_folder_name, modules_folder_name):
    cwd = os.getcwd()
    i = 0; head_tail = '';
    while head_tail != parent_folder_name:
        head = Path(cwd).parents[i]
        head_tail = os.path.split(head)[1]
        i += 1
    sys.path.append(os.path.join(head, modules_folder_name))
