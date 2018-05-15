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
# leverage
short_leverage = 1.0
long_leverage = 0.5
save_file = 'strategy2'

if __name__ == "__main__":
    df = af.data_processing(file_address,data_list,ma_long,ma_short,short_leverage,
                    long_leverage,T_new,T_old,Year)
    # ===reprocessing data===
    cond_list = [df['signal'] > 0, df['signal'] < 0]
    choice_list1 = [1, 0]
    df['call_signal'] = np.select(cond_list, choice_list1)
    choice_list2 = [0, 1]
    df['put_signal'] = np.select(cond_list, choice_list2)
    # calculating
    df['profit'] = ((df['call_signal']*df['call_89']+df['put_signal']*df['put_89'])
                      - (df['call_signal'].shift(1)*df['call_90'].shift(1)+df['put_signal'].shift(1)*
                         df['put_90'].shift(1)))
    df['return'] = df['profit'] / \
                   (df['call_signal'].shift(1) * df['call_90'].shift(1) + df['put_signal'].shift(1) *
                    df['put_90'].shift(1)) * long_leverage+(1-long_leverage)*df['Interest_rate'].shift()/Year
    perf = af.comput_idicators(df, Year, save_file, file_address + save_file + '.csv')
    save_file1 = save_file+'_longcall'
    perf_longcall = af.cut_position(df, 'call_signal',Year, save_file1, file_address + save_file1 + '.csv')
    save_file2 =save_file+'_longput'
    perf_longput = af.cut_position(df, 'put_signal',Year, save_file2, file_address + save_file2 + '.csv')
    print(perf)