'''
portfolio1
'''
import pandas as pd
pd.set_option('expand_frame_repr', False)
import IAQF_Functions as af
import numpy as np

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
long_leverage = 1.0
save_file = 'strategy6'

if __name__ == "__main__":
    df = af.data_processing(file_address,data_list,ma_long,ma_short,short_leverage,
                    long_leverage,T_new,T_old,Year)
    df['profit'] = ((df['call_89']+df['S&P500'].shift(1)*np.exp(-df['Interest_rate'].shift(1)*(T_old/Year))-df['put_89'])\
                   -(df['call_90'].shift(1)+df['S&P500'].shift(1)*np.exp(-df['Interest_rate'].shift(1)*(T_new/Year))
                     -df['put_90'].shift(1)))*df['signal'].shift(1)
    df['return'] = df['profit']/(df['call_90'].shift(1)+df['S&P500'].shift(1)*np.exp(-df['Interest_rate'].shift(1)*(T_new/Year))
                     -df['put_90'].shift(1))
    perf = af.comput_idicators(df, Year,save_file,file_address+save_file+'.csv')
    print(perf)