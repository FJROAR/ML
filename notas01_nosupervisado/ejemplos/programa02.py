#source: https://pbpython.com/market-basket-analysis.html

import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

path = 'data/'

df = pd.read_excel(path + 'Online_Retail.xlsx')

#Forma habitual de visualizar los datos en pandas, aunque también se puede abrir
#directamente el dataset desde la "pestaña" Variable Explorer"
df.head()

#Limpieza de datos
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

#Se prepara la tabla tal y como se quiere, según una matriz de incidencias:
#Una transacción por fila
#Las columnas son productos


#Se toma sólo el caso de Francia, pero se podría tomar el de Alemania
basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

#Visualización de la matriz de producto
basket.head()

#Para que quede como matriz de incidencias, todo lo que sea mayor que 1, 
#será 1. Se genera la matriz de incidencias a utilizar

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

#Visualización de la matriz de incidencias a usar por el algoritmo apriori
basket_sets.head()


###############################################################################
#                                                                             #
#              Aplicación del algoritmo apriori                               #
#                                                                             #
###############################################################################

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

###############################################################################
#                                                                             #
#                              Modelo MBA                                     #
#                                                                             #
###############################################################################

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()


#Análisis de los resultados del modelo MBA en Francia

#Se eligen reglas significativas

strong_rules = rules[ (rules['lift'] >= 6) &
                     (rules['confidence'] >= 0.8)]

strong_rules.columns
strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

###############################################################################



###############################################################################
#                                                                             #
#              Se aplica todo lo anterior al caso de Alemania                 #
#                                                                             #
###############################################################################

basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets2 = basket2.applymap(encode_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

strong_rules2 = rules2[ (rules2['lift'] >= 4) &
                       (rules2['confidence'] >= 0.5)]

strong_rules2.columns
strong_rules2[['antecedents', 'consequents', 'support', 'confidence', 'lift']]