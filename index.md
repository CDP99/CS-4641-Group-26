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

### Potential results: 
For a given quarter, we will take all of the features that we decide to use on our model and try to predict the average growth of the stock price over the next three years. To display the accuracy of our model, we will have a chart where the x-axis is time from the date that we pulled the financials to three years after. The y-axis would be the percentage difference from our estimated value.

E.g. If we are estimating the growth of Microsoft using data from 1/1/2010, the values on the x-axis would be from 1/1/2010 to 1/1/2013. If we estimated an average growth of 15% over 3 years and the price on 1/1/2011 was 10% less than our estimated value of 15%, the value on the y-axis for that point in time would be -5%. Then, if a month later the stock price has gone up to 16.25% greater than the price at 1/1/2010, the value on the y-axis would be 0% because that price falls perfectly in line with our estimated growth.

If our three-year prediction for a stock is above what the current average growth has been, the plotted line will noticeably stray downward from the centerline of 0%. We can also use this chart to plot multiple stocks at once. Depending on if a majority of the lines end above the baseline, below the baseline, or on the baseline of 0%, we will be able to see if our model is accurately predicting growth.

### Conclusion: 
Our goal for this project will be to determine which modelling strategy is the most appropriate for forecasting stock market changes while using financial temporal data and use it with the objective of constructing a model capable of predicting changes to stock price over the span of three years. 

### Bibliography: 
1. Milosevic, N. (2016). Equity forecast: Predicting long term stock price movement using machine learning. arXiv preprint arXiv:1603.00751.
2. Cao, Lijuan & Tay, Francis. (2001). Financial Forecasting Using Support Vector Machines. Neural Computing and Applications. 10. 184-192. 10.1007/s005210170010.
3. Nunno, L. (2014). Stock Market Price Prediction Using Linear and Polynomial Regression Models.
