'''
portfolio3
'''
import pandas as pd
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
save_file = 'strategy3'

if __name__ == "__main__":
    df = af.data_processing(file_address,data_list,ma_long,ma_short,short_leverage,
                    long_leverage,T_new,T_old,Year)
    # calculating profit
    df['profit'] = (df['straddle_89'] - df['straddle_90'].shift(1))
    df['return'] = df['profit']/df['straddle_90'].shift(1)*long_leverage+\
                   (1-long_leverage)*df['Interest_rate'].shift()/Year
    perf = af.comput_idicators(df, Year,save_file,file_address+save_file+'.csv')
    print(perf)