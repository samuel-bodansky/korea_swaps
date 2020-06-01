import pandas as pd 
path = '/Users/samuelbodansky/onix/'
btc_path = path + 'btc_daily.csv'
options_path = path + 'perp_data.csv'
btc_daily = pd.read_csv(btc_path)
options = pd.read_csv(options_path)
print(btc_daily.shape)
print(options.shape)
