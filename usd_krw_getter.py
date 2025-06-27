import pandas_datareader as pdr
from datetime import datetime

# Get USD/KRW from FRED
start = datetime(2023, 6, 27)
end = datetime(2025, 6, 27)

usdkrw = pdr.get_data_fred('DEXKOUS', start, end)
usdkrw.columns = ['USD_KRW']
print(1)