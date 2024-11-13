
#%%
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# %%

with open('return_predictability.pkl', 'rb') as f:
    df = pickle.load(f)
    print(df.head())


ff_research = pd.read_csv('data/ff_factors.csv')
ff_research.head()

df['Date'] = [''.join(x.split('-')[0:2]) for x in df.DATE]
# %%
