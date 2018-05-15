import numpy as np
import pandas as pd
import os

def moment(x,order):
    return 1.0/len(x)*np.nansum(np.power(x-np.nanmean(x),order))
def skewness(x):
    return moment(x,3)/np.power(moment(x,2),1.5)

def find_last_digit(x):
    def remainder(scalar):
        return int(scalar) % 10
    vremainder = np.vectorize(remainder)
    sub = vremainder(x)
    stats = np.zeros(10)
    for i in range(0,len(x)):
        stats[int(sub[i])] += 1
    return stats

if __name__ =="__main__":

    path = os.getcwd()
    data_file = os.path.join(path+'/data/S_P500.csv')
    df = pd.read_csv(data_file)
    price = np.array(df['Close'])
    returns = (price - np.roll(price,1))/np.roll(price,1)
    returns[0] = np.nan
    print('skewness = ', skewness(returns))

    a = find_last_digit(price)
    print('='*10,' Frequence of last digit','\n',a,)
    print('mean of numbers: ',np.mean(a))
    print('median :',np.median(a))
    print('standard deviation: ',np.std(a))