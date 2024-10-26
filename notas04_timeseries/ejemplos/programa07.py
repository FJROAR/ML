import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm

series = pd.read_csv('data/paroinfla.csv', 
                     header=0, 
                     index_col=0,
                     sep = ';',
                     decimal = ',')

#Se ajusta un primero modelo lineal: "Curva de Phipllips para la Economía
#Española
y = series['INFLACION']
x = series['PARO']
x = sm.add_constant(x)

mod1 = sm.OLS(y, x).fit()
mod1.summary()

#Correlación residual
plot_acf(mod1.resid)
plot_pacf(mod1.resid)

#Transformación de Cochrane-Orcutt
e = mod1.resid
e1 = e.shift(1)

e = e[1:(len(e) + 1)]
e1 = e1[1:(len(e1) + 1)]

mode = sm.OLS(e, e1).fit()
mode.summary()

yt = y[1:(len(e1) + 1)] - mode.params[0] * y.shift(1)[1:(len(e1) + 1)] 
xt = series['PARO'][1:(len(e1) + 1)] - mode.params[0] * series['PARO'].shift(1)[1:(len(e1) + 1)]
xt = sm.add_constant(xt)

mod2 = sm.OLS(yt, xt).fit()
mod2.summary()

plot_acf(mod2.resid)
plot_pacf(mod2.resid)


#Otra posibilidad: Incluir en el modelo expectativas racionales y por tonto
#medir el incremento de la inflación en función de la tasa de paro

ye = y[1:(len(e1) + 1)] - y.shift(1)[1:(len(e1) + 1)]
xe = series['PARO'][1:(len(e1) + 1)] - series['PARO'].shift(1)[1:(len(e1) + 1)]

xe = sm.add_constant(xe)

mod3 = sm.OLS(ye, xe).fit()
mod3.summary()




