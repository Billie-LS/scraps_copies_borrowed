import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import scipy as sci
import scipy.optimize as sco



# https://www.kaggle.com/code/alvarob96/tutorial-on-data-retrieval-using-investpy
# https://investpy.readthedocs.io/_info/usage.html
# https://pypi.org/project/investpy/
import sys
# !{sys.executable} -m pip install investpy -U
import random
from pprint import pprint
import investpy

df = investpy.get_stock_historical_data(stock='AAPL',
                                        country='United States',
                                        from_date='01/01/2021',
                                        to_date='01/01/2022')
print(df.head())








# https://github.com/santiment/sanpy
# https://medium.datadriveninvestor.com/using-python-to-get-crypto-market-data-ba173d93896b

"""
import san
san.ApiConfig.api_key = 'api-key-provided-by-sanbase'




ohlc_df = san.get(
“ohlc/ethereum”,
from_date=”2015–01–01",
to_date=”2020–02–10",
interval=”1d”
)
print(ohlc_df.tail())

"""