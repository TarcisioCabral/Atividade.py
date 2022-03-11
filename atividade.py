import pandas as pd
from apriori_python import apriori

heart = pd.read_csv('heart.csv')

agrupamento = heart.groupby('Sex')['Age'].apply(list)

print(heart.head())

freqItemSet,rules=apriori(agrupamento, minSup=0.000909, minConf=0.5)
print(len(agrupamento))
print(rules)
