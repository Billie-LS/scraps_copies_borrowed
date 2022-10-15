"""
This code module primary write team Will, Yan, billie, Dinesh
"""

import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt



#class Data_Collection:

def get_raw_data(asset_list, market, start, end):
    stocks = market + asset_list
    stock_data = pdr.get_data_yahoo(stocks, start, end)
    stock_data = stock_data['Close']
    return stock_data

"""
def get_refined_data(asset_list, market, start, end):
    stocks = market + asset_list
    print(f"ticker list: {stocks}")
    stock_data = pdr.get_data_yahoo(stocks, start, end)
    stock_data = stock_data['Close']
    stock_data.dropna()
    print(f" resultant data frame: {stock_data}")
    returns = stock_data.pct_change()
    print(f" calculating returns for time period.... \n complete.. \n {returns}")
    mean_returns = returns.mean()
    print(f"calculating mean returns.. \n {mean_returns}")
    log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
    print(f"calculating log returns \n {log_returns}")
    stock_covariance = returns.cov()
    print(f" stock covariance: {stock_covariance}")
    return stock_data, mean_returns, log_returns, stock_covariance
"""


# Basket
# asset_list = ['AAPL', 'IBM', 'TSLA', 'GOOGL']
# market = ['^GSPC']     


# start = '2010-01-01'
# end = '2022-10-10'
# get_stock_data(asset_list, market, start, end)

# stock_data=mpt_precanned_asset_selection(stocks, start, end)
# print(stock_data)

