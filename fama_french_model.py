
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

df.interpolate(method='linear', limit_direction='backward', inplace=True)

df = df.merge(ff_research[['Date','macro_smb', 'macro_hml', 'macro_mkt-rf']], on = 'Date', how ='inner' )

# %%
from sklearn.model_selection import train_test_split

X = df[['macro_smb', 'macro_hml', 'macro_mkt-rf', 'macro_tbl']]
y = df[['risk_premium']]



# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)


# %%

cv_train_score =  []
cv_test_score = []
for train_index, test_index in cv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    lm = LinearRegression()
    y_pred_lr_train = lm.predict(X_train)
    y_pred_lr_test = lm.predict(y_test)
    
    cv_train_score.append(r2_linear_train)
    cv_test_score.append(cv_test_score)
    print(cv_train_score.mean())
    print(cv_test_score.mean())

    

# %%
