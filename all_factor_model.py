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

df_copy = df.copy()

train, test = train_test_split(df_copy, test_size=0.2, random_state=42)

train.drop(columns=['DATE','Date','permno', 'age', 'macro_smb', 'realestate', 'name','bm_ia', 'macro_hml','sic2', 'macro_mkt-rf'], inplace=True)
test.drop(columns=['DATE','Date','permno', 'age', 'macro_smb', 'realestate', 'name', 'bm_ia', 'macro_hml', 'sic2', 'macro_mkt-rf'], inplace=True)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# %%

from sklearn.preprocessing import StandardScaler

X_train = train.drop(columns=['risk_premium'])
X_test = test.drop(columns=['risk_premium'])

y_train = train['risk_premium']
y_test = test['risk_premium']



stdSc = StandardScaler()
X_train= stdSc.fit_transform(X_train)
X_test = stdSc.fit_transform(X_test)


# %%

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, make_scorer, mean_squared_error 

lm = LinearRegression()

cv = KFold(n_splits=5, shuffle=True, random_state=45)
r2 = make_scorer(r2_score)

r2_val_score = cross_val_score(lm, X_train, y_train, cv=cv, scoring=r2)
scores = [r2_val_score.mean()]
scores

# %%
from sklearn.cross_decomposition import PLSRegression

def optimise_pls_cv(X, y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components=n_comp)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)

    # Calculate scores
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    
    return (y_cv, r2, mse, rpd)

r2s = []
mses = []
rpds = []
xticks = np.arange(1, 41)
for n_comp in xticks:
    y_cv, r2, mse, rpd = optimise_pls_cv(X_train, y_train, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)

def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('PLS')

    plt.show()
    

plot_metrics(mses, 'MSE', 'min')

# %%
