import pandas as pd
path = '/Users/samuelbodansky/onix/equity_vs_btc/equity_btc_data.csv'
df = pd.read_csv(path)
n = 2000
# my_column = 'NI225, TVC: Open'
my_column = 'SPX, SPCFD: Open'
corr_dic = {}
for i in range(-n,n):
    # print(i)
    # print(df['open'].corr(df['NI225, TVC: Open'].shift(i)))
    corr_dic[str(i)] = df['open'].corr(df[my_column].shift(i))

key_max = max(corr_dic.keys(), key=(lambda k: corr_dic[k]))
print('biggest_key',key_max)
print('biggest_value',corr_dic[key_max])

key_min = min(corr_dic.keys(), key=(lambda k: corr_dic[k]))
print('smallest_key',key_min)
print('smallest_value',corr_dic[key_min])

from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
series = df['open']

plot_pacf(series,lags =1000)
plt.savefig('btc_autocorrelation_mar_2020')