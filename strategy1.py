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
# data_list=('Interest_rate',)
# momentum strategy parameters
ma_short = 60
ma_long = 120
# trading day
Year = 250.0
# option expiration
T_new = 90.0
T_old = 89.0
# leverage
short_leverage = 0.5
long_leverage = 0.5
save_file = 'strategy1'

if __name__ == "__main__":
    df = af.data_processing(file_address,data_list,ma_long,ma_short,short_leverage,
                    long_leverage,T_new,T_old,Year)

    df['profit'] = (df['S&P500']-df['S&P500'].shift(1))
    df['return'] = df['profit']/\
                         df['S&P500'].shift(1)*df['signal'].shift(1)+\
                   (1-long_leverage)*(np.log(df['S&P500'])-np.log(df['S&P500']).shift())
    # df['return'] = df[]
    perf = af.comput_idicators(df, Year,save_file,file_address+save_file+'.csv')
    save_file1 = save_file + '_long'
    perf_long = af.cut_position(df,'signal',Year, save_file1+'_long', file_address
                                + save_file1 + '.csv')
    save_file2 = save_file+'_short'
    perf_short = af.cut_position(df,'signal',Year, save_file2+'_short',
                                 file_address + save_file2 + '.csv',long =0)
    print(perf)