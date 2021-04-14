import pandas as pd
import numpy as np
import scipy.stats as stats

url = 'https://raw.githubusercontent.com/josjo99/dmw_21/main/train_ds.csv'
data = pd.read_csv(url, index_col='PassengerId', usecols=['PassengerId','Pclass','Survived'])
data_cross = pd.crosstab(data.Pclass,data.Survived,margins=True)

print("PASSENGER CLASS VS SURVIVAL\n",data_cross,"\n")

observed = data_cross.iloc[0:3,0:3]

print("OBSERVED\n",observed,"\n")

expected = np.outer(data_cross["All"][0:3],data_cross.loc["All"][0:3])/1010
expected = pd.DataFrame(expected)

print("EXPECTED\n",expected,"\n")

chi2,p,dof,expected2 = stats.chi2_contingency(observed=observed)

expected2 = pd.DataFrame(expected2)

print("chi value: ",chi2)
print("p value: ",p)
print("dof: ",dof)
