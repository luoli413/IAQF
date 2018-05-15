"""
Function list
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Function list******************************************************************************
# import data
def input_data(file_name):
    # df = pd.DataFrame()
    df = pd.read_csv(file_name)
    df.sort_values(by = ['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    # rename the price column by the data_list name.
    col_name = file_name[file_name.rfind('/')+1:-4]
    df.rename(columns={'Adj Close': col_name}, inplace=True)
    df.ffill()
    if col_name !='S&P500':
        if col_name =='Interest_rate':
            df.loc[df[col_name]=='.',col_name] = np.nan
        df[col_name] = pd.to_numeric(df[col_name]) * 0.01
    return df[col_name]

# producing signals and positions
def signals(df,ma_short,ma_long,short_deposit,long_leverage):
    df['ma_short'] = df.iloc[:,0].rolling(ma_short, min_periods=ma_short).mean()
    df['ma_long'] = df.iloc[:,0].rolling(ma_long, min_periods=ma_long).mean()
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = long_leverage
    df.loc[df['ma_short'] < df['ma_long'], 'signal'] = - short_deposit
    return df[~df['signal'].isna()]

# options prices
'''
variable clarification:
    name:'c' call option; 'p' put option
output:
    first one: price
    second one : delta    
    third one : gamma 
'''
def callandput(S,K,sigma,r,T,name):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    if name =='c':
        return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2), norm.cdf(d1), \
               norm.pdf(d1)/(S*sigma*(T**0.5))
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1), norm.cdf(d1)-1, \
           norm.pdf(d1)/(S*sigma*(T**0.5))

# computing indicators
def comput_idicators(df,trading_days,save_file,save_address, whole=1):
    # TODO:net_value has some problem.
    df['net_value'] = (df['return'] + 1).cumprod()
    df['net_value'].ix[0] = 1
    df['cum_profit'] = df['profit'].cumsum()
    df['cum_profit'].ix[0] = 0
    # columns needed
    col = ['return','S&P500','Interest_rate','net_value','cum_profit']
    df_valid = df.ix[:,col]
    # benchmark_net_value
    df_valid['benchmark'] = df_valid['S&P500']/df_valid['S&P500'].ix[0]
    # benchmark_return
    df_valid['benchmark_return'] = (df_valid['benchmark']-
                                           df_valid['benchmark'].shift(1))/\
                                   df_valid['benchmark'].shift(1)
    # Annualized return
    df_valid['Annu_return'] = pd.expanding_mean(df_valid['return'])\
                                *np.sqrt(trading_days)
    # Volatility
    df_valid.loc[:, 'algo_volatility'] = pd.expanding_std(df_valid
                                                          ['return']) * np.sqrt(trading_days)
    df_valid.loc[:, 'xret'] = df_valid['return'] - df_valid[
        'Interest_rate'] / trading_days
    df_valid.loc[:,'ex_return'] = df_valid['return'] - df_valid['benchmark_return']
    def ratio(x):
        return np.nanmean(x)/np.nanstd(x)
    # sharpe ratio
    df_valid.loc[:, 'sharpe'] = pd.expanding_apply(df_valid['xret'], ratio)\
                                * np.sqrt(trading_days)
    # information ratio
    df_valid.loc[:, 'IR'] = pd.expanding_apply(df_valid['ex_return'], ratio)\
                                * np.sqrt(trading_days)
    # Transfer infs to NA
    df_valid.loc[np.isinf(df_valid.loc[:, 'sharpe']), 'sharpe'] = np.nan
    df_valid.loc[np.isinf(df_valid.loc[:, 'IR']), 'IR'] = np.nan
    # hit_rate
    wins = np.where(df_valid['return'] >= df_valid[
        'benchmark_return'],  1.0, 0.0)
    df_valid.loc[:, 'hit_rate'] = wins.cumsum()/pd.expanding_apply(wins, len)
    # 95% VaR
    df_valid['VaR'] = -pd.expanding_quantile(df_valid['return'], 0.05)*\
                      np.sqrt(trading_days)
    # 95% CVaR
    df_valid['CVaR'] = -pd.expanding_apply(df_valid['return'],
                                          lambda x: x[x < np.nanpercentile(x, 5)].mean())\
                       * np.sqrt(trading_days)
    if whole ==1:
    # max_drawdown
        def exp_diff(x,type):
            if type == 'dollar':
                xret = pd.expanding_apply(x, lambda xx:
                (xx[-1] - xx.max()))
            else:
                xret = pd.expanding_apply(x, lambda xx:
                (xx[-1] - xx.max())/xx.max())
            return xret
    # dollar
        xret = exp_diff(df_valid['cum_profit'],'dollar')
        df_valid['max_drawdown_profit'] = abs(pd.expanding_min(xret))
    # percentage
        xret = exp_diff(df_valid['net_value'], 'percentage')
        df_valid['max_drawdown_ret'] = abs(pd.expanding_min(xret))
    # max_drawdown_duration:
    # drawdown_enddate is the first time for restoring the max
        def drawdown_end(x,type):
                xret= exp_diff(x,type)
                minloc = xret[xret == xret.min()].index[0]
                x_sub = xret[xret.index > minloc]
            # if never recovering,then return nan
                try:
                    return x_sub[x_sub==0].index[0]
                except:
                    return np.nan
        def drawdown_start(x,type):
                xret = exp_diff(x, type)
                minloc = xret[xret == xret.min()].index[0]
                x_sub = xret[xret.index < minloc]
                try:
                    return x_sub[x_sub==0].index[-1]
                except:
                    return np.nan
        df_valid['max_drawdown_profit_start'] = pd.Series()
        df_valid['max_drawdown_profit_end'] = pd.Series()
        df_valid['max_drawdown_profit_start'].ix[-1] = drawdown_start(
            df_valid['cum_profit'],'dollar')
        df_valid['max_drawdown_profit_end'].ix[-1] = drawdown_end(
            df_valid['cum_profit'], 'dollar')
    df_valid.to_csv(save_address)
    # =====result visualization=====
    plt.figure(1)
    if whole==1:
        plt.subplot(224)
        plt.plot(df_valid['net_value'],label = 'strategy')
        plt.plot(df_valid['benchmark'],label = 'S&P500')
    plt.xlabel('Date')
    plt.legend(loc=0, shadow=True)
    plt.ylabel('Net_value')
    plt.title('Net_value of '+ save_file +' & SP500')

    plt.subplot(223)
    plt.plot(df_valid['cum_profit'],label = 'strategy')
    plt.xlabel('Date')
    plt.ylabel('Cum_profit')
    plt.title('Cum_profit of ' + save_file)

    plt.subplot(221)
    plt.plot(df_valid['return'], label='strategy')
    plt.xlabel('Date')
    plt.ylabel('Daily_return')
    plt.title('Daily Return of ' + save_file)

    plt.subplot(222)
    x_return = df_valid[df_valid['return'].notna()].loc[:,'return']
    y_return = df_valid[df_valid[
        'benchmark_return'].notna()].loc[:,'benchmark_return']
    mu = x_return.mean()
    sigma = x_return.std()
    mybins = np.linspace(mu-3*sigma,mu+3*sigma,100)
    count_x,_,_ = plt.hist(x_return,mybins,normed=1,alpha=0.5,label = 'strategy')
    count_y,_,_ = plt.hist(y_return,mybins,normed =1,alpha=0.5,label = 'S&P500')
    plt.ylabel('density')
    plt.xlabel('daily_return')
    plt.title('Histogram of Daily Return for ' +
              save_file+' & SP500')
    plt.grid(True)
    # add normal distribution line
    y = mlab.normpdf(mybins, mu, sigma)
    plt.plot(mybins, y, 'r--', linewidth = 1,label = 'Normal of strategy')
    plt.legend(loc=0, shadow=True)
    plt.tight_layout()
    plt.show()
    return df_valid

def data_processing(file_address,data_list,ma_long,ma_short,deposit_percent,
                    long_leverage,T_new,T_old,Year):
    # import data
    df = pd.DataFrame()
    s = 0
    for st in data_list:
        # print(input_data(file_address + st + '.csv'))
        if s==0:
            df = input_data(file_address+st+'.csv')
        else:
            df = pd.concat([df, input_data(file_address+st+'.csv')], axis = 1)
        s+=1
    df = df.dropna(axis=0 , how='any')
    # ******************************************************************************
    # signal
    df = signals(df,ma_short,ma_long,deposit_percent,long_leverage)
    # ============for portfolio 2&3=======================
    # vectorized function
    vcallandput = np.vectorize(callandput)
    # at-the-money
    df['call_90'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['S&P500']),
                                             np.array(df['VXV']), np.array(df['Interest_rate']),
                                             T_new/Year, 'c')[0], index=df.index)
    # when moving to the next day, strike price is the old one.
    df['call_89'] = pd.DataFrame(vcallandput(np.array(df['S&P500']),np.array(df['S&P500'].shift(1)),
                                             np.array(df['VXV']),np.array(df['Interest_rate']),
                                             T_old/Year,'c')[0],index=df.index)
    df['put_90'] = pd.DataFrame(vcallandput(np.array(df['S&P500']),np.array(df['S&P500']),
                                            np.array(df['VXV']),np.array(df['Interest_rate']),
                                            T_new/Year,'p')[0],index=df.index)
    df['put_89'] = pd.DataFrame(vcallandput(np.array(df['S&P500']),np.array(df['S&P500'].shift(1)),
                                            np.array(df['VXV']),np.array(df['Interest_rate']),
                                            T_old/Year,'p')[0],index=df.index)
    df['straddle_90'] = df['call_90']+df['put_90']
    df['straddle_89'] = df['call_89']+df['put_89']
    return df
def cut_position(df,signal_col,trading_days,save_file,save_address,long =1):
    if long==1:
        return comput_idicators(df[df[signal_col]>0],trading_days,save_file,save_address,whole=0)
    return comput_idicators(df[df[signal_col]<0],trading_days,save_file,save_address,whole=0)



