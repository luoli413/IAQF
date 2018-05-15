'''
portfolio4
'''
import pandas as pd
import numpy as np
pd.set_option('expand_frame_repr', False)
import IAQF_Functions as af

# parameter initialization
file_address = 'D:/Work&Study/NYU/PythonScripts/IQAF/data/'
data_list = ('S&P500','VXV','Interest_rate')
# momentum strategy parameters
ma_short = 60
ma_long = 120
# trading day
Year = 250.0
# option expiration
T_new = 90.0
T_old = 89.0
# leverage
short_leverage = 1.0
long_leverage = 1.0
save_file = 'strategy4'

def reprocessing(T, Year, df):
    straddle_time = np.tile(np.arange(T, 0, -1) * 1.0 / Year, int(df.shape[0] / T) + 1)
    df['straddle_time'] = straddle_time[:df.shape[0]]
    # add column
    df['strike_price'] = pd.Series()
    df.loc[df['straddle_time'] == T / Year, 'strike_price'] = \
        df.loc[df['straddle_time'] == T / Year, 'S&P500']
    df['strike_price'] = df['strike_price'].ffill()
    vcallandput = np.vectorize(af.callandput)
    df['put_holding'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['strike_price']),
                                                 np.array(df['VXV']), np.array(df['Interest_rate']),
                                                 np.array(df['straddle_time']), 'p')[0], index=df.index)
    df['call_holding'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['strike_price']),
                                                  np.array(df['VXV']), np.array(df['Interest_rate']),
                                                  np.array(df['straddle_time']), 'c')[0], index=df.index)
    df['straddle_holding'] = df['call_holding'] + df['put_holding']
    # stock shares
    df['delta_old'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['strike_price']),
                                         np.array(df['VXV']), np.array(df['Interest_rate']),
                                         np.array(df['straddle_time']), 'c')[1], index=df.index)
    df['gamma_old'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['strike_price']),
                                         np.array(df['VXV']), np.array(df['Interest_rate']),
                                         np.array(df['straddle_time']), 'c')[2], index=df.index)
    df['delta_new'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['S&P500']),
                                         np.array(df['VXV']), np.array(df['Interest_rate']),
                                         T / Year, 'c')[1], index=df.index)
    df['gamma_new'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['S&P500']),
                                         np.array(df['VXV']), np.array(df['Interest_rate']),
                                         T / Year, 'c')[2], index=df.index)
    df['gamma'] = 2.0 * ((df['gamma_new'] - df['gamma_old']) / df['gamma_new'])
    df['delta'] = 2.0 * (df['delta_new'] - df['delta_old'])-df['gamma']*(df['delta_new']-1.0)
    df['cash'] = df['straddle_90']-df['delta']*df['S&P500']-df['gamma']*df['put_90']-df['straddle_holding']
    return df

if __name__ == "__main__":
    df = af.data_processing(file_address,data_list,ma_long,ma_short,short_leverage,
                    long_leverage,T_new,T_old,Year)
    df = reprocessing(T_new, Year, df)
    # calculating profit
    df['profit'] = (df['straddle_holding']+df['delta'].shift(1)*df['S&P500']+df['gamma'].shift(1)*df['put_89']
                    + (1+df['Interest_rate'].shift(1)*(1/Year))*df['cash'].shift(1)-df['straddle_90'].shift(1))
    df = df.reset_index()
    for i in range(1, df.shape[0], 1):
        if df.loc[i, 'straddle_time'] == T_new/Year:
            df.loc[i, 'profit'] = max(0, df.loc[i, 'S&P500']-df.loc[i-1, 'strike_price'])\
                                   + max(0, df.loc[i-1, 'strike_price']-df.loc[i, 'S&P500'])\
                                   + df.loc[i-1, 'delta']*df.loc[i, 'S&P500']\
                                   + df.loc[i-1, 'gamma']*df.loc[i, 'put_89']\
                                   + (1+df.loc[i-1, 'Interest_rate']*(1/Year))*df.loc[i-1, 'cash']\
                                   - df.loc[i-1, 'straddle_90']



    # profit_90 = (np.maximum(0, df[df['straddle_time'] == T_new / Year]['S&P500']-
    #                                                          df[df['straddle_time'] == 1 / Year]['strike_price'])+\
    #                                                     np.maximum(0, df[df['straddle_time'] == 1 / Year]['strike_price']-
    #                                                                (df[df['straddle_time'] == T_new / Year]['S&P500']).iloc[1:, :])+\
    #                                                     (df[df['straddle_time'] == 1 / Year]['delta']
    #                                                      *(df[df['straddle_time'] == T_new / Year]['S&P500']).iloc[1:, :])+\
    #                                                         (df[df['straddle_time'] == 1/ Year]['gamma']
    #                                                          *(df[df['straddle_time'] == T_new / Year]['put_89']).iloc[1:, :])+\
    #                                                     (1+df[df['straddle_time'] == 1 / Year]['Interest_rate']*(1/Year))\
    #                                                     * df[df['straddle_time'] == 1 / Year]['cash']\
    #                                                     -df[df['straddle_time'] == 1 / Year]['straddle_90'])
    df['return'] = df['profit'] \
                   / (df['straddle_90'].shift(1))
    # df.loc[df['straddle_time'] == T_old / Year, 'return'] = df[df['straddle_time'] == T_old / Year]['profit']/\
    #                                                     (np.maximum(0, df[df['straddle_time'] == T_new / Year]['S&P500']-
    #                                                          df[df['straddle_time'] == T_new / Year]['strike_price'].shift(1))+\
    #                                                     np.maximum(0, df[df['straddle_time'] == T_new / Year]['strike_price'].shift(1) -
    #                                                          df[df['straddle_time'] == T_new / Year]['S&P500'])+\
    #                                                     (df[df['straddle_time'] == T_new / Year]['delta'].shift(1)\
    #                                                     *df[df['straddle_time'] == T_new / Year]['S&P500'])+\
    #                                                         (df[df['straddle_time'] == T_new / Year]['gamma'].shift(1)\
    #                                                     *df[df['straddle_time'] == T_new / Year]['put_90'].shift(1))+\
    #                                                      (1+df[df['straddle_time'] == T_new / Year]['Interest_rate'].shift(1) *(1 / Year))\
    #                                                         * df[df['straddle_time'] == T_new / Year]['cash'].shift(1))
    perf = af.comput_idicators(df, Year, save_file, file_address+save_file+'.csv')
    print(perf)
