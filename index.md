## CS 4641 - Predictive Stock Machine Learning Project Proposal
#### Fletcher Wells, Javier Arevalo, Matthew Nilsen, Cedric de Pierrefeu, Shengyuan Huang 

### Abstract: 
We want to use different technical and fundamental stock metrics to create a regression model to try and estimate the 1yr, 3yr, and/or 5yr growth of a stock. Typically, people only use technical indicators in their models. However, we feel that using fundamental indicators like RoE, RoIC, etc. can add to the model’s ability to classify growth, since these numbers are found in company’s financial statements and have a substantive meaning to the company’s performance. We expect that we will be able to estimate at least whether a company will grow or shrink in value; we aren’t sure how accurately we will be able to estimate a company’s stock price growth percentage.

### Introduction/Background: 
Since the late 1980s various world renowned researchers, institutions, hedge funds, and banks have investigated the task of how to use data mining and machine learning to increase profitability in the stock market. This is one of the most research ML areas and various newer research papers have considered novel input to the predictive models such as a company’s relevant tweets (NLP), technical indicators, environmental indicators, among many others. 

The reason why this problem is so important and crucial is because of the numbers game. Even if a model beats the market by a small percentage, given a sufficiently large amount of money this model(s) can become extremely profitable business (IB, HedgeFunds, etc.) Many of the work that has been done previously is not publicly available, because of the highly profitable nature of such models and algorithms. The research papers that are available (mostly on Sentiment Analysis on the stock market).

### Problem definition: 
Will focus on fundamental stock analysis inputs (list below) and if the case that we do not find the historical data required to train our model, we will use the quantitative indicators in the quant indicators list (already have historical data on those). 

Fundamental Stock Analysis Inputs:
* 1yr, 3yr, 5yr Return on Equity growth
* 1yr, 3yr, 5yr Return on Invested Capital growth
* Difference between Return on Equity and Return on Invested Capital
* Price to Earnings Ratio
* 1yr, 3yr, 5yr Free Cash Flow growth
* Price/ Book ratio

Quantitative Indicators:
* Relative Strength Index (RSI)
* Relative Volume (RVOL)
* Exponential Moving Average (EMA, variable days, 30d, 50d, 253d, etc)
* Short Term Moving Average (SMA, variable days, 30d, 50d, 253d, etc)
* Squeeze Theorem (TTM)
* Moving Average Convergence Divergence Indicator (MACD)

### Data Collection:
We collected all of our data from the FinancialModelingPrep API and used the following endpoints:
* api/v3/enterprise-values/
* api/v3/income-statement-growth/
* api/v3/ratios/

We selected data based on our own knowledge of ratios/ numbers used in estimating companies’ values and growths as well as data that had been used in similar studies that we had read. The ratios that we used along with descriptions are as follows:

|Name          |Formula                              |Description|
|--------------|-------------------------------------|-----------|
|Current Ratio |Current assets / Current liabilities |Measures ability to pay short term obligations|
|Quick Ratio  |(Current assets - inventory) / current liabilities  |Measures short-term liquidity |
|Gross Profit Margin  |(Revenue - Cost of Goods Sold) / Revenue  |Shows how successful a company is in producing profit above costs|
|Operating Profit Margin  |Operating Income / Revenue  |Shows how efficiently a company is able to generate profit  |
|Return on Assets  |Net Income / Total Assets  |Shows how profitable a company is relative to its assets|
|Return on Equity  |Net Income / Shareholder Equity  |Measures business profitability in relation to shareholder equity|
|Return on Capital Employed  |Earnings Before Interest and Tax / Capital Employed  |Shows how efficiently a company is able to generate profits from its capital|
|Debt Equity Ratio  |Liabilities / Shareholder Equity  |Shows how leveraged a company is  |
|Price Book Ratio  |Share Price / (Book Value / Share)  |Shows a company’s stock valuation in relation to the company’s book value  |
|Price Sales Ratio  |Market Cap / Total Sales  |Measures how much investors are paying per dollar of company sales  |
|Price Earnings Ratio  |Share Price / (Earnings / Share)  |Shows how high a company is values in relation to its actual earnings  |
|Price to Earnings Growth Ratio  |(Price / Earnings) / Expected EPS Growth  |Shows how much investors are paying for earnings in relation to expected earnings growth  |
|Price to Operating Cash Flow Ratio  |Share Price / (Operating Cash Flow / Share)  |Shows how much investors are paying for cash coming into a business before deductions  |

We also used the following values:
* Revenue Growth
* Cost of Revenue Growth
* Gross Profit Growth
* Gross Profit Ratio Growth
* Operating Expense Growth
* Earnings Before Interest, Taxes, Depreciation, and Amortization Growth
* Operating Income Growth
* Net Income Growth
* Earnings Per Share Growth
* Market Capitalization
* Enterprise Value

### Methods: 
Our first step would be to model the predictive algorithm as a regression problem. Since regression assumes independent input, we need to filter out our feature list first. 
Have compiled a complete set of indicators to use as input but suspect not all of these features are independent of each other since various fundamental metrics come from similar procedures using the same data of a company. Thus we suspect that we will have to filter our original input list as to leave only an independent input list once this process is complete.
 
If define problem as regression:
* Polynomial Regression Model with stock price

If define problem as classification:
* Neural Network Multiclass Classification with y in {bullish, bearish, neutral}
* Recurrent Neural Network for Sentiment Analysis 
* Bayesian Classifier
* Support Vector Machine

### Metrics:
We plan to use Root Mean Squared Error as our loss function.

### Results: 
We have collected data from the past 52 quarters from Apple and have run dimensionality reduction to determine which features are most independent from one another. We will use this information to cull the features we use for training our model. Next, after deciding which features to use, we will try to predict the average growth of the stock price over the next three years. To display the accuracy of our model, we will have a chart where the x-axis is time from the date that we pulled the financials to three years after. The y-axis would be the percentage difference from our estimated value.
If our three-year prediction for a stock is above what the current average growth has been, the plotted line will noticeably stray downward from the centerline of 0%. We can also use this chart to plot multiple stocks at once. Depending on if a majority of the lines end above the baseline, below the baseline, or on the baseline of 0%, we will be able to see if our model is accurately predicting growth.

### Discussion:
Our goal for this project is to determine which modelling strategy is the most appropriate for forecasting stock market changes while using financial temporal data and use it with the objective of constructing a model capable of predicting changes to stock price over the span of three years. 

### Bibliography: 
1. Milosevic, N. (2016). Equity forecast: Predicting long term stock price movement using machine learning. arXiv preprint arXiv:1603.00751.
2. Cao, Lijuan & Tay, Francis. (2001). Financial Forecasting Using Support Vector Machines. Neural Computing and Applications. 10. 184-192. 10.1007/s005210170010.
3. Nunno, L. (2014). Stock Market Price Prediction Using Linear and Polynomial Regression Models.
