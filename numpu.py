import numpy as np

def asset_metrics(portfolio_df):
    return_asset = portfolio_df/portfolio_df.shift(1)-1
    mean_asset = return_asset.mean() * 252
    std_asset = return_asset.std() * np.sqrt(252)
    cov_asset = return_asset.cov() * 252
    corr_asset = return_asset.corr()
    return return_asset, mean_asset, std_asset, cov_asset, corr_asset
  import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
from metrics import asset_metrics

def calc_portfolio_ret(w, mean_return):
    return np.sum(w*mean_return)

def calc_portfolio_std(w, cov_matrix):
    return np.sqrt(np.dot(np.dot(w, cov_matrix), w.transpose()))

def sharpe_rate2(w, portfolio_df, riskfree, sign=-1.0):
    return_asset = portfolio_df/portfolio_df.shift(1)-1
    cov_matrix = return_asset.cov() * 252
    mean_return = return_asset.mean() * 252

    ret = calc_portfolio_ret(w, mean_return)
    excess_ret = ret - riskfree
    std = calc_portfolio_std(w, cov_matrix)
    return sign*(excess_ret/std)

def drawdown(w, ret_asset, sign=-1.0):
    portfolio_hozam = pd.DataFrame(np.dot(ret_asset, w)+1)
    portfolio_hozam.columns = ["Return"]
    portfolio_ertek =1000*portfolio_hozam["Return"].cumprod()
    max_portfolio = portfolio_ertek.rolling(window=len(portfolio_ertek), min_periods=1).max()
    dd = portfolio_ertek / max_portfolio - 1
    mdd = dd.rolling(window=len(portfolio_ertek), min_periods=1).min()
    mmdd = mdd.min()
    return sign*mmdd

df_UNG = pd.read_csv("UNG.csv")
df_DBA = pd.read_csv("DBA.csv")
df_DBB = pd.read_csv("DBB.csv")
df_LQD = pd.read_csv("LQD.csv")
df_XLI = pd.read_csv("XLI.csv")
df_USD = pd.read_csv("DTB3.csv")

df_UNG = df_UNG.drop(df_UNG.columns[[1, 2, 3, 4, 6]], axis=1)
df_DBA = df_DBA.drop(df_DBA.columns[[1, 2, 3, 4, 6]], axis=1)
df_DBB = df_DBB.drop(df_DBB.columns[[1, 2, 3, 4, 6]], axis=1)
df_LQD = df_LQD.drop(df_LQD.columns[[1, 2, 3, 4, 6]], axis=1)
df_XLI = df_XLI.drop(df_XLI.columns[[1, 2, 3, 4, 6]], axis=1)
df_USD.set_index("DATE", inplace=True)
df = df_USD

df = df.drop(df[df.DTB3 == "."].index)
df.columns=["Risk free"]
df["Risk free"] = df["Risk free"].astype(float)

atlag = df["Risk free"].mean()
risk_free=atlag*4/100

portfolio_df = pd.merge(df_UNG, df_XLI, on = "Date")
portfolio_df = pd.merge(portfolio_df, df_LQD, on = "Date")
portfolio_df = pd.merge(portfolio_df, df_DBB, on = "Date")
portfolio_df = pd.merge(portfolio_df, df_DBA, on = "Date")
portfolio_df.columns=["Date", "UNG", "XLI", "LQD", "DBB", "DBA"]

portfolio_df.set_index("Date", inplace=True)

ret_asset, mean_asset, std_asset, cov_asset, corr_asset = asset_metrics(portfolio_df)


#sulyok1

cons = ({"type": "eq", "fun": lambda weight: np.sum(weight) - 1})

#sharpe maximalizálás

sharpe_max = sp.optimize.minimize(sharpe_rate2,
                           np.array([0.5,0.5,0,0,0]),
                           args=(portfolio_df, risk_free),
                           constraints=cons)

sharpe_weight = sharpe_max.x
sharpe_rate = -sharpe_max.fun

#drawdown minimalizálás

drawdown_min = sp.optimize.minimize(drawdown,
                            np.array([0, 0.3, 0.7, 0, 0]),
                            args=(ret_asset),
                            constraints=cons)

drawdown_weight = drawdown_min.x
drawdown_perc = -drawdown_min.fun
