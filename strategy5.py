'''
portfolio2
'''
import pandas as pd
import numpy as np
pd.set_option('expand_frame_repr', False)
import IAQF_Functions as af
import matplotlib.pyplot as plt

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
# leverage > 1
short_leverage = 1.0
long_leverage = 0.5
save_file = 'strategy5'

if __name__ == "__main__":
    df = af.data_processing(file_address,data_list,ma_long,ma_short,short_leverage,
                    long_leverage,T_new,T_old,Year)
    # ===reprocessing data===
    cond_list = [df['signal'] > 0, df['signal'] < 0]
    choice_list1 = [1, 0]
    df['call_signal'] = np.select(cond_list, choice_list1)
    choice_list2 = [0, 1]
    df['put_signal'] = np.select(cond_list, choice_list2)
    vcallandput = np.vectorize(af.callandput)
    df['delta_call'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['S&P500']),
                                         np.array(df['VXV']), np.array(df['Interest_rate']),
                                         T_new / Year, 'c')[1], index=df.index)
    df['delta_put'] = pd.DataFrame(vcallandput(np.array(df['S&P500']), np.array(df['S&P500']),
                                         np.array(df['VXV']), np.array(df['Interest_rate']),
                                         T_new / Year, 'p')[1], index=df.index)
    df['call_position'] = 1/(df['delta_call'])
    df['put_position'] = -1 / (df['delta_put'])
    df['call_cash'] = df['S&P500']-df['call_position']*df['call_90']
    df['put_cash'] = df['S&P500']-df['put_position']*df['put_90']
    # calculating
    df['profit'] = ((df['call_signal']*df['call_89']*df['call_position'].shift(1)+df['put_signal']*df['put_89']*df['put_position'].shift(1))
                    + df['call_signal']*(1+df['Interest_rate'].shift(1)*(1/250))*df['call_cash'].shift(1)
                    + df['put_signal']*(1+df['Interest_rate'].shift(1)*(1/250))*df['put_cash'].shift(1)
                    - df['S&P500'].shift(1))
    df['return'] = df['profit'] / df['S&P500'].shift(1)
    perf = af.comput_idicators(df, Year, save_file, file_address + save_file + '.csv')
    print(perf)