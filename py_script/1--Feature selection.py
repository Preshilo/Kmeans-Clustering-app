import os
import pandas as pd
from warnings import simplefilter
import pickle
os.chdir('/home/presh/Downloads/Christwin/hhblock_dataset')
simplefilter(action='ignore', category=FutureWarning)
ect_data0 = pd.read_csv('block_0.csv')
ect_data0 = ect_data0.groupby('LCLid').median()
for i in range(1, 20):
    ect_data_i = pd.read_csv(f'block_{i}.csv') 
    ect_data0 = pd.concat([ect_data0,  ect_data_i.groupby('LCLid').median()], ignore_index=False, axis=0)
for i in range(20, 40):
    ect_data_i = pd.read_csv(f'block_{i}.csv') 
    ect_data0 = pd.concat([ect_data0,  ect_data_i.groupby('LCLid').median()], ignore_index=False, axis=0)
for i in range(40, 60):
    ect_data_i = pd.read_csv(f'block_{i}.csv') 
    ect_data0 = pd.concat([ect_data0,  ect_data_i.groupby('LCLid').median()], ignore_index=False, axis=0)
for i in range(60, 80):
    ect_data_i = pd.read_csv(f'block_{i}.csv') 
    ect_data0 = pd.concat([ect_data0,  ect_data_i.groupby('LCLid').median()], ignore_index=False, axis=0)
for i in range(80, 100):
    ect_data_i = pd.read_csv(f'block_{i}.csv') 
    ect_data0 = pd.concat([ect_data0,  ect_data_i.groupby('LCLid').median()], ignore_index=False, axis=0)
for i in range(100, 112):
    ect_data_i = pd.read_csv(f'block_{i}.csv') 
    ect_data0 = pd.concat([ect_data0,  ect_data_i.groupby('LCLid').median()], ignore_index=False, axis=0)
ect_data0.ffill(inplace = True)
User = ect_data0.copy()
User.reset_index(inplace = True)
os.chdir('/home/presh/Downloads/Christwin/hhblock_dataset/pickles')
with open('data.pickle', 'wb') as fh:
    pickle.dump(User, fh)

