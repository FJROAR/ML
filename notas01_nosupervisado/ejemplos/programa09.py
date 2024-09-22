#https://github.com/jingtt/varclushi

import pandas as pd
from varclushi import VarClusHi

#Se traen datos desde una url
demo1_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

#En este caso se quita la variable por la que se podría agrupar
demo1_df.drop('quality',axis=1,inplace=True)

demo1_vc = VarClusHi(demo1_df,
                     maxeigval2=1,
                     maxclus=None)

demo1_vc.varclus()

demo1_vc.info

demo1_vc.rsquare
