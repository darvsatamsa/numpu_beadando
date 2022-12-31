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

#rolling Sharpe

sharpe_weight_rolling = []
sharpe_fun_rolling = []

for i in range(1260, 3774):
    portfolio_df_list = portfolio_df.values.tolist()
    portfolio_df_list_windowed = portfolio_df_list[i-1260:i]
    portfolio_df_windowed = pd.DataFrame(portfolio_df_list_windowed)
    print(i)
    rolling = scipy.optimize.minimize(sharpe_rate2,
                            np.array([0,0.5,0.5,0,0]),
                            args=(portfolio_df_windowed, risk_free),
                            constraints=cons)

    rw = rolling.x
    rs = rolling.fun
    sharpe_weight_rolling.append(rw)
    sharpe_fun_rolling.append(rs)

sharpe_fun_rolling_df = pd.DataFrame(sharpe_fun_rolling)
sharpe_weight_rolling_df = pd.DataFrame(sharpe_weight_rolling)



#rolling drawdown

drawdown_weight_rolling = []
drawdown_fun_rolling = []

for i in range(1260, 3774):
    portfolio_df_list = portfolio_df.values.tolist()
    portfolio_df_list_windowed = portfolio_df_list[i-1260:i]
    portfolio_df_windowed = pd.DataFrame(portfolio_df_list_windowed)
    print(i)

    return_asset_windowed = portfolio_df_windowed / portfolio_df_windowed.shift(1) - 1

    rolling = scipy.optimize.minimize(drawdown,
                            np.array([0,0.5,0.5,0,0]),
                            args=(return_asset_windowed),
                            constraints=cons)

    rwdd = rolling.x
    rdd = rolling.fun
    drawdown_weight_rolling.append(rwdd)
    drawdown_fun_rolling.append(rdd)

drawdown_weight_rolling_df = pd.DataFrame(drawdown_weight_rolling)
drawdown_fun_rolling_df = pd.DataFrame(drawdown_fun_rolling)


#Sharpe plotok

ret_asset_select = pd.DataFrame(ret_asset[1260:])
ret_asset_select = ret_asset_select.reset_index(drop=True)
sharpe_weight_rolling_df.columns = ["UNG", "XLI", "LQD", "DBB", "DBA"]
df3 = ret_asset_select.mul(sharpe_weight_rolling_df)
df3 = df3.sum(axis=1)+1

df3 = pd.DataFrame(df3)
df3.columns=["Napi hozam"]

sharpe_value = 1000*df3["Napi hozam"].cumprod()

plt.plot(sharpe_value)
plt.title("Portfólió értékének alakulása Sharpe mutató szerint optimalizált súlyokkal")
plt.ylabel("USD")
plt.show()

plt.plot(sharpe_weight_rolling_df)
lineObjects = plt.plot(sharpe_weight_rolling_df)
plt.title("Optimális súlyok alakulása")
plt.legend(iter(lineObjects), ("UNG","XLI","LQD","DBB","DBA"))
plt.show()


#drawdown plotok

ret_asset_select = pd.DataFrame(ret_asset[1260:])
ret_asset_select = ret_asset_select.reset_index(drop=True)
drawdown_weight_rolling_df.columns = ["UNG", "XLI", "LQD", "DBB", "DBA"]
df3 = ret_asset_select.mul(drawdown_weight_rolling_df)
df3 = df3.sum(axis=1)+1

df3 = pd.DataFrame(df3)
df3.columns=["Napi hozam"]

drawdown_value = 1000*df3["Napi hozam"].cumprod()

plt.plot(drawdown_value)
plt.title("Portfólió értékének alakulása Maximum Drawdown szerint optimalizált súlyokkal")
plt.ylabel("USD")
plt.show()

plt.plot(drawdown_weight_rolling_df)
lineObjects = plt.plot(drawdown_weight_rolling_df)
plt.title("Optimális súlyok alakulása")
plt.legend(iter(lineObjects), ("UNG","XLI","LQD","DBB","DBA"))
plt.show()


pass
