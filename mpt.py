import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import scipy as sci
import scipy.optimize as sco
import plotly.graph_objects as go



# import data
def gets_the_data(stocks, start, end):
    # stock_data = yf.download(stocks, start = start, end = end)
    stock_data = pdr.get_data_yahoo(stocks, start = start, end = end)
    stock_data = stock_data['Close']
    
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    covariance = returns.cov()
    
    return mean_returns, covariance


# performance
def portfolio_performance(weight, mean_returns, covariance, hurdle_rate = 0.0275):
    returns = np.sum(mean_returns * weight)*252   # (mean_returns * weight).sum()*252
    std_dev = np.sqrt(np.dot(weight.T, np.dot(covariance, weight))) * np.sqrt(252)    #  portfolio variance
    return returns, std_dev


def negative_sharpe_rat(weight, mean_returns, covariance, hurdle_rate = 0.0275):
    p_returns, p_standard_dev = portfolio_performance(weight, mean_returns, covariance)
    return -1*(p_returns - hurdle_rate) / p_standard_dev
                   

def optimal_sharpe_rat(mean_returns, covariance, hurdle_rate=0.03, boundaries_set = (0,1)):
    "maximize sharpe rat or minimize negative sharpe rat via portfolio weighting"
    number_securities = len(mean_returns)
    args = (mean_returns, covariance, hurdle_rate)
    boundaries = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1})     # see np scipy 'SLSQP', this is ~lambda weight
    reaches = boundaries_set
    barriers = tuple(reaches for security in range(number_securities))
    result = sco.minimize(negative_sharpe_rat(weight, number_securities* [1.0/number_securities], args = args, method = 'SLSQP', barriers = barriers, boundaries = boundaries))   
    return result

    
def portfolio_variance(weight, mean_returns, covariance):
    return portfolio_performance(weight, mean_returns, covariance)[1]

def minimize_variance(mean_returns, covariance, boundaries_set = (0,1)):
    "maximize volatility through adjusting security weighting in portfolio"
    number_securities = len(mean_returns)
    args = (mean_returns, covariance)
    boundaries = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1})
    reaches = boundaries_set
    barriers = tuple(reaches for security in range(number_securities))
    result = sco.minimize(portfolio_variance, number_securities* [1.0/number_securities], args = args,
                                   method = 'SLSQP', barriers = barriers, boundaries = boundaries)   
    return result
    

def computed_results(mean_returns, covariance, hurdle_rate=0.03, boundaries_set = (0,1)):
                          
    # Optimal Sharpe Ratio Portfolio
    portfolio_optimal_sharpe_rat = optimal_sharpe_rat(mean_returns, covariance)
    returns_optimal_sharpe_rat, standard_dev_optimal_sharpe_rat = portfolio_performance(portfolio_optimal_sharpe_rat['x'], mean_returns, covariance) 
    allocation_optimal_sharpe_rat = pd.DataFrame(portfolio_optimal_sharpe_rat['x'], index=mean_returns.index, columns=['allocation'])
    allocation_optimal_sharpe_rat.allocation = [round(i*100, 0) for i in allocation_optimal_sharpe_rat.allocation]                      
                          
    # Least Risk Portfolio                      
    volume_limited_portfolio = minimize_variance(mean_returns, covariance) 
    returns_volume_limited, standard_dev_volume_limited = portfolio_performance(volume_limited['x'], mean_returns, covariance)   
    allocation_volume_limited = pd.DataFrame(volume_limited_portfolio['x'], index=mean_returns.index, columns=['allocation']) 
    allocation_volume_limited.allocation = [round(i*100, 0) for i in allocation_optimal_sharpe_rat.allocation]                      
    
    # Efficient Frontier
    efficient_composition = []
    target_returns = np.linspace(returns_volume_limited, returns_optimal_sharpe_rat, 20)
    for target in target_returns:
        efficient_composition.append(efficient_opt(mean_returns, covariance, target,['fun']))   
    
    returns_optimal_sharpe_rat, standard_dev_optimal_sharpe_rat = round(returns_optimal_sharpe_rat*100,2), round(standard_dev_optimal_sharpe_rat*100,2)                      
    returns_volume_limited, standard_dev_volume_limited = round(returns_volume_limited*100, 2), round(standard_dev_volume_limited*100, 2)                      
                          
    return returns_optimal_sharpe_rat, 
    standard_dev_optimal_sharpe_rat, 
    allocation_optimal_sharpe_rat, 
    returns_volume_limited, 
    standard_dev_volume_limited, 
    allocation_volume_limited, 
    efficient_composition, target_returns                         
                          

def ef_plot(mean_returns, covariance, hurdle_rate=0.0275, boundaries_set = (0,1)):                          
                          
     returns_optimal_sharpe_rat, standard_dev_optimal_sharpe_rat, allocation_optimal_sharpe_rat, returns_volume_limited, standard_dev_volume_limited, allocation_volume_limited, efficient_composition, target_returns = computed_results(mean_returns, covariance, hurdle_rate, boundaries_set)                   
                          
     # Optimal Sharpe Ratio
     optimal_sharpe_rat = go.scatter(
         name = 'Optimal Sharpe Ratio',
         mode = 'markers',
         x = [standard_dev_optimal_sharpe_rat],
         y = [returns_optimal_sharpe_rat],
         marker = dict(color='magenta', size=15, line=dict(width=4, color='blue'))
     )
                          
     # Least Risk Volume
     limited_volume = go.scatter(
         name = 'Minimum Risk',
         mode = 'markers',
         x = [standard_dev_volume_limited],
         y = [returns_volume_limited],
         marker = dict(color='red', size=15, line=dict(width=4, color='blue'))
     )                          
                          

     # Efficient Frontier
     graph_ef = go.scatter(
         name = 'Efficient Frontier',
         mode = 'liness',
         x = [round(ef_std * 100, 2) for ef_std in efficient_composition],
         y = [round(target * 100, 2) for target in target_returns],
         line = dict(color='black', width=3, dash = 'dotdash' color='blue')
     )                          
                                   
                          
     traces = ['optimal_sharpe_rat', 'limited_volume', 'graph_ef']                     
     configuration = go.layout(
         title = 'Modern Portfolio Thoery - Optimization with Efficient Frontier',
         yaxis = dict(title = 'Percent Annual Return'),
         xaxis = dict(title ='Percent Annual Volatility'),
         showlegend = True,
         legend = dict(
             x = 0.75, y = 0, traceorder = 'normal',
             bgcolor = '#E2E2E2',                           # gray background
             bordercolor = 'black',
             borderwidth = 2),
         width = 800,
         height = 600)

                          
                          
     fig = go.figure(data=traces, layout = configuration)                    
     return fig.show()                     
                          
                          
                          
                          
stocklist = ['GOOGL', 'AAPL', 'TSLA', 'IBM']
# stocks = [stocks + '.AX' for stocks in stocklist]  # i think this +AX requirement deprecated/resolved
stocks = stocklist

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365)

weight = np.array([0.3, 0.3, 0.3, 0.1])

mean_returns, covariance = gets_the_data(stocks, start = start_date, end = end_date)
#mean_returns, covariance = gets_the_data(stocks, start = start_date, end = end_date)
returns, std_dev = portfolio_performance(weight, mean_returns, covariance)

                       
                          
# result = optimal_sharpe_rat(mean_returns, covariance, weight, hurdle_rate=0.03)  
result =  optimal_sharpe_rat(mean_returns, covariance)   
opt_sharpe, opt_weight = result['fun'], result['x']                           
print(opt_sharpe, opt_weight)  
                          
                          
                          
                          
 
minimal_var_result= minimize_variance(mean_returns, covariance)
min_var, min_var_weight = minimal_var_result['fun'], minimal_var_result['x']
          
print(returns, std_dev)
          

ef_plot(mean_returns, covariance)





