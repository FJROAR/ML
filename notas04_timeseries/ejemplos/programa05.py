import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf # lo hacemos utilizando yf como alias


#Reading data from internet


def genera_bandas(fch_ini, fch_fin, ticker, window):
    
    ticker = ticker

    # Obtenemos los datos
    df_market = yf.download(ticker, start = fch_ini, end = fch_fin)

    df_bb = pd.DataFrame()
    df_bb['Close'] = df_market['Close']

    df_market.head()


    # Moving Average
    df_bb['ma'] = df_market['Close'].rolling(window = window).mean()
    df_bb.tail()

    # Standard Desviation
    df_bb['std'] = df_market['Close'].rolling(window = window).std()
    df_bb.tail()

    #Computing the Bollinger's bands
    K = 2 # numbers of standard desviation to use
    df_bb['bb_upper'] = df_bb['ma'] + K*df_bb['std']
    df_bb['bb_lower'] = df_bb['ma'] - K*df_bb['std']
    df_bb.tail()

    #Media precovid
    media = yf.download(ticker, start = '2019-01-01', end = '2019-12-31')
    media = np.mean(media['Close'])
    
    #Mínimo amplio precovid
    
    minimo = yf.download(ticker, start = '2018-01-01', end = '2019-12-31')
    minimo = np.min(minimo['Close'])


    df_bb[['Close', 'bb_upper', 'bb_lower']].plot()
    
    print("Media precovid: ", media)
    print("Mínimo amplio precovid: ", minimo)


    return (df_bb.tail())



#ACCIONA

fch_ini = '2018-01-01'
fch_fin = '2024-10-16'
ticker = 'ANA.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#ACERINOX

fch_ini = '2018-01-01'
fch_fin = '2024-02-16'
ticker = 'ACX.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#ACS

fch_ini = '2018-01-01'
fch_fin = '2024-02-16'
ticker = 'ACS.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#AENA

fch_ini = '2018-01-01'
fch_fin = '2023-02-10'
ticker = 'AENA.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Almirall**

fch_ini = '2018-01-01'
fch_fin = '2023-02-10'
ticker = 'ALM.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Amadeus

fch_ini = '2018-01-01'
fch_fin = '2023-02-22'
ticker = 'AMS.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Acelor

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'MTS.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Sabadell

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'SAB.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Santander

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'SAN.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Bankinter

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'BKT.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#BBVA

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'BBVA'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Caixa Bank

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'CABK.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Cellnex

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'CLNX.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Cie

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'CIE.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Colonial**

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'COL.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Enagás***

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'ENG.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Endesa**

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'ELE.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Ferrovial

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'FER.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Fluidra

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'FDR.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Grifols***

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'GRF.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#IAG*

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'IAG.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Iberdrola

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'IBE.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Indra

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'IDR.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Indietx

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'ITX.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Maphre*

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'MAP.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Meliá

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'MEL.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Merlin

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'MRL.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Naturgy

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'NTGY.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Pharmamar*

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'PHM.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Red Eléctrica*

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'REE.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Repsol

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'REP.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Rovi

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'ROVI.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Siemens Gamesa

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'SGRE.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

#Solaria

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'SLR.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)


#Telefónica

fch_ini = '2018-01-01'
fch_fin = '2022-02-22'
ticker = 'TEF.MC'
window = 60

genera_bandas(fch_ini, fch_fin, ticker, window)

