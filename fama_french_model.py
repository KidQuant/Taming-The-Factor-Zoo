
#%%
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# %%

# df = pd.read_csv('data/return_predictability_data.csv')

# with open('return_predictability.pkl', 'wb') as f:
#     pickle.dump(df, f)

with open('return_predictability.pkl', 'rb') as f:
    df = pickle.load(f)
    print(df.head())


ff_research = pd.read_csv('data/ff_factors.csv')
ff_research.head()

df['Date'] = [''.join(x.split('-')[0:2]) for x in df.DATE]

# df.interpolate(method='linear', limit_direction='backward', inplace=True)

df = df.merge(ff_research[['Date', 'macro_mkt-rf']], on = 'Date', how ='inner' )

ff3= df[~np.isnan(df['bm'])]
ff3=ff3[~np.isnan(df['mvel1'])]


# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = ff3[['macro_mkt-rf', 'macro_tbl', 'mvel1', 'bm']]
y = ff3[['risk_premium']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

stdSc = StandardScaler()
X_train = stdSc.fit_transform(X_train.astype(float))
X_test = stdSc.fit_transform(X_test.astype(float))


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
xticks = np.arange(1, 5)
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
plot_metrics(rpds, 'RPD', 'max')


# %%
##############################################
# Fama French 5 Factor Model
###############################################

df = df[['macro_mkt-rf', 'macro_tbl', 'mvel1', 'bm', 'operprof', 'grcapx', 'risk_premium']]

df= df[~np.isnan(df['bm'])]
df=df[~np.isnan(df['mvel1'])]
df=df[~np.isnan(df['operprof'])]
df=df[~np.isnan(df['grcapx'])]

X = df[['macro_mkt-rf', 'macro_tbl', 'mvel1', 'bm', 'operprof', 'grcapx']]
y = df['risk_premium'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=45)
r2 = make_scorer(r2_score)

r2_val_score = cross_val_score(lm, X_train, y_train, cv=cv, scoring=r2)
scores = [r2_val_score.mean()]
scores
# %%

r2s = []
mses = []
rpds = []
n_comp = 4
xticks = np.arange(1, 5)
for n_comp in xticks:
    y_cv, r2, mse, rpd = optimise_pls_cv(X_train, y_train, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)


plot_metrics(mses, 'MSE', 'min')
plot_metrics(rpds, 'RPD', 'max')

# %%

