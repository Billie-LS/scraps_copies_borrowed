"""
This code module primary write team billie, Will
"""
import pandas as pd
import numpy as np
import scipy as sci
from scipy import stats

trade_days_year = 252  # variable to set number of trading days per year
hurdle_rate = 0.0275 # is variable for risk free rate ~ 2-3%


def get_comp_analytics_data(raw_frame_close, trade_days_year):
    raw_df = raw_frame_close
    
    # 'Close' prices dataframe daily returns for ALL portfolio assets and <market/index>
    returns = raw_df.pct_change()
    # 'Close' prices dataframe AVERAGE daily returns for ALL portfolio assets and <market/index>
    mean_returns = returns.mean()
    
    # 'Close' prices standard deviation, sorted, for ALL portfolio assets and <market/index>, sorted
    standard_deviation_returns = returns.std().sort_values()
    
    # 21 day rolling standard deviation, for all portfolio assets and <market/index>
    rolling_standard_deviation_returns = returns.rolling(window=21).std() 
    
    # annualized standard deviation for for all portfolio assets and <market/index>, sorted
    annualized_std = (standard_deviation_returns * np.sqrt(trade_days_year)).sort_values()
    
    # 'Close' prices dataframe Log returns for ALL portfolio assets and <market/index>
    log_returns = np.log(raw_frame_close / raw_frame_close.shift(1)).dropna()
    
    # annual average returns, sorted, for all portfolio assets and <market/index>, sorted
    annual_average_returns = (mean_returns*trade_days_year).sort_values()
    
    # cumulative_returns
    cumulative_returns_plus = (1 + returns).cumprod()      
    cumulative_returns = (1 + returns).cumprod() - 1       # the -1 removes initial investment and so result is the profits
    
    # asset correlation
    asset_correlation = returns.corr()
    
    # annualized Sharpe Ratios for all portfolio assets and <market/index>
    sharpe_ratios = (annual_average_returns - hurdle_rate) / (returns.std()*np.sqrt(trade_days_year))
    sharpe_ratios = sharpe_ratios.sort_values() # sorting
    
    stock_covariance = returns.cov()
    return (raw_df, 
            returns, 
            mean_returns, 
            standard_deviation_returns, 
            rolling_standard_deviation_returns, 
            annualized_std, log_returns, 
            annual_average_returns, 
            cumulative_returns_plus, 
            cumulative_returns, 
            asset_correlation, 
            sharpe_ratios, 
            stock_covariance 
    )

     