library(ISLR2)

df = Credit

mod1 = lm(Balance ~., data = df)
summary(mod1)
AIC(mod1)
BIC(mod1)

RSE1 = (mean(resid(mod1)**2))^0.5


#Cálculo AIC
AIC_form = 2 * (length(mod1$coefficients) + 1) +
  nrow(df) * (log(2*pi) + log(RSE1**2) + 1)


#Cálculo BIC
BIC_form = log(nrow(df)) * (length(mod1$coefficients) + 1) +
  nrow(df) * (log(2*pi) + log(RSE1**2) + 1)

  
mod2 = lm(Balance ~Income + Limit, data = df)
summary(mod2)

AIC(mod2)
BIC(mod2)

RSE2 = (mean(resid(mod2)**2))^0.5

AIC_form = 2 * (length(mod2$coefficients) + 1) +
  nrow(df) * (log(2*pi) + log(RSE2**2) + 1)

BIC_form = log(nrow(df)) * (length(mod2$coefficients) + 1) +
  nrow(df) * (log(2*pi) + log(RSE2**2) + 1)
