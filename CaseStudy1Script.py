'''
Submission to ISAZI for evaluation
By: Mason Hu
Date: 17 November 2019
'''
# Import analysis related libraries
import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm
import datetime as dt
import copy

# Import reporting related libraries
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.layouts import gridplot

# Grid search function for finding best ARIMA pdq and PDQ parameters based on AIC metric
# The ARIMA differencing order will be set to 0, and the seasonal differencing order will be set to 1.
# The differencing orders were identified through analysis with the ACF and PACF plots (not implemented here)
# The input to the function is a time series, the maximum orders for p,q,P,Q parameters, and seasonality constant for the SARIMA model. 
# Returns the fitted SARIMA model

def grid_search_arima(X, max_order=1, max_sorder=1, season=12):
    
    # Create a grid for the ARIMA pdq and seasonal PDQ parameters
    p = d = q = range(0, max_order+1)
    sp = sd = sq = range(0, max_sorder+1)
    pdq = [(x[0], 0, x[2]) for x in list(itertools.product(p, d, q))] # Always set the d term to 0
    seasonal_pdq = [(x[0], 1, x[2], season) for x in list(itertools.product(sp, sd, sq))] # Always set the D term to 1

    # Train and evaluate the SARIMA model using the parameter grid defined, runs through all permutations of pdq and PDQ parameters
    best_score, best_order, best_sorder = float('inf'), (), ()
    for param in list(set(pdq)): # set(pdq) extracts unique parameter sets from pdq
        for param_seasonal in list(set(seasonal_pdq)): # set(seasonal_pdq) extracts unique parameter sets from seasonal_pdqand
            try:
                mod = sm.tsa.statespace.SARIMAX(X,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                mod_fit = mod.fit()
                score_aic = mod_fit.aic

                # Check if the current model produces better AIC update best model configuration accordingly
                if score_aic < best_score:
                    best_score = score_aic
                    best_order = param
                    best_sorder = param_seasonal
            except:
                continue

    # Fitting the model with best parameters             
    mod = sm.tsa.statespace.SARIMAX(X,
                                    order=best_order,
                                    seasonal_order=best_sorder,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    mod_fit = mod.fit()

    print(f'Best AIC: {best_score} \nBest pdq: {best_order} \nBest Seasonal pdq: {best_sorder}')
    
    return mod_fit

# Function for using a fitted SARIMA model to forecast sales volume
# The inputs to this function includes a time series, a SARIMA model and the number of days to be forecasted
# The function will return a dataframe object with the forecast concatenated to the original dataframe.

def forecast(X, model, f_days=1):
    future_dates = [X.index[-1] + DateOffset(days=x) for x in range(1,f_days)]
    future_df = pd.DataFrame(index=future_dates, columns=X.columns)
    final_df = pd.concat([X, future_df], sort=False)
    final_df['forecast'] = model.predict(start=len(X)-1, end=len(X)+f_days, dynamic=False)
    
    return final_df

if __name__ == '__main__':
    
    # Import data
    df = pd.read_csv("DS - case study 1 - add material - sales_volumes.csv")
    
    # Convert Date column to datetime type and set as index
    df.Date = pd.to_datetime(df.Date, format="%d-%m-%y %H:%M")
    df = df.set_index('Date')
    
    """
    It was noticed during data exploration that there are a ton of returns in the middle of June, 
    which is an outlier given the data sample range. It seems like some product had printing smudges
    and got thrown away on the 14th of June. This does not seem like a frequent occurence and could skew the forecasting
    as this is the only day out of all 6 months that Volume went negative in total.
    Therefore, these product defects on the 14th will be replaced with average returns on the same day
    """ 
    # Extract June data from dataset and check when the significant return of product took place
    df_june = df[df.index >= dt.datetime(2019,6,1)]

    # Calculating the average of returns (negative volume) on the 14th of June 
    june_14_return_avg = df[(df.index>dt.datetime(2019,6,14)) & 
                            (df.index<dt.datetime(2019,6,15)) & 
                            (df.Volume<0)].Volume.mean()

    # Replacing the 3 entries of returns seen on the 14th of June with the average returns calculated
    idx = df[(df.index>dt.datetime(2019,6,14)) & (df.index<dt.datetime(2019,6,15)) & (df.Volume<-20000)].index
    df_clean = copy.deepcopy(df) 
    df_clean.Volume.loc[list(idx)] = june_14_return_avg
    
    # Removing all features except for Volume, and resampling dataset to daily sales volume
    cols = ['Unnamed: 0', 'InvoiceID', 'ProductCode', 'Description', 'UnitPrice']
    df_clean = df.drop(cols, axis=1)
    df_clean = df_clean.resample(rule='D').sum()

    # Splitting the data into train and test set (not going to validate with the test here, it was performed in the notebook)
    df_train = df_clean.iloc[:int(0.75*len(df_clean))]
    df_test = df_clean.iloc[int(0.75*len(df_clean)):]
    
    # Run grid search on SARIMA parameters. Highest order for parameters set to 2, seasonal constant set to 7 i.e. 7 days in a week
    # The data has weekly seasonality
    mod_fit = grid_search_arima(df_train.Volume, max_order=2, max_sorder=2, season=7)
    
    # Making future forecast for July (31 days)
    final_df = forecast(df_clean, mod_fit, 31)
    
    # Reporting using Bokeh
    output_file('Forecast Results.html')
    final_df = round(final_df,0) # Convert units sold to integer

    # Create a ColumnDataSource from past and forecasted sales volume
    source_p = ColumnDataSource(pd.DataFrame(df.resample(rule='D').sum().Volume.loc[:dt.datetime(2019,6,30)])) # Past sales volume
    source_f = ColumnDataSource(pd.DataFrame(final_df.forecast.loc[dt.datetime(2019,7,1):])) # Future sales volume
    source_ft = ColumnDataSource(pd.DataFrame(final_df.forecast.loc[dt.datetime(2019,7,1):])) # Future sales volume, used for linking table to plot
    
    # Setting up the time series plot
    p = figure(x_axis_type = 'datetime',plot_height=650, plot_width=1000, title = 'Past and Future Sales Volume')
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_alpha = 0.5
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Volume'
    p.xaxis[0].formatter.days = '%d-%m-%Y'
    p.xaxis.major_label_orientation = 3.14/3 # Create a 60 degrees rotation of x-axis tick labels
    p.line('Date', 'Volume', source=source_p, legend_label='Past Sales Volume')
    p.line('index', 'forecast', source=source_f, line_color='red', line_dash='dotted', legend_label='Future Sales Volume (July)')
    p.circle('index', 'forecast', source=source_ft, line_color='green', fill_alpha=0.5, legend_label='Future Sales Volume (July)')

    # Setting up the data table for July forecast
    columns = [
        TableColumn(field='index', title='Future Date', formatter=DateFormatter()),
        TableColumn(field='forecast', title='Forcasted Sales [Units]'),
    ]
    data_table = DataTable(source=source_ft, columns=columns, width=400, height=600)

    # Combine both time series plot and data table in a grid plot, displayed columnwise, tool bar location will be on the right
    plot = gridplot([[p,data_table]], toolbar_location="right")

    show(plot)

