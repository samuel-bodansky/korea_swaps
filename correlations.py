import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

path = '/Users/samuelbodansky/onix/data_13042020/data_correlations_april.csv'


df = pd.read_csv(path)
data_columns = [
    'open',
    'SPX, SPCFD: Compare',
    'GOLD, TVC: Compare',
    'ETHUSD, GEMINI: Compare', 
    'USOIL, TVC: Compare'
    ]
df1 = df[data_columns]
print(df1.shape)
corr_matrix = df1.corr()
sn.heatmap(corr_matrix,annot=True)
plt.show()
