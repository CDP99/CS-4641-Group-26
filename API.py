import requests
import pandas as pd
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys

import time

class API:
    def __init__(self, key, period='annual', limit=5):
        self.base_url = 'https://financialmodelingprep.com/'
        self.params = {
            'apikey': key,
            'period': period,
            'limit': limit,
        }
        self.alwaysCols = ["symbol", "date"]

        # keeps tracks of the time each api call was made (so that we know whether we go over the api call limit)
        self.apiCalls = []

        if limit == None:
            del self.params['limit']

    def overApiCallLimit(self):
        API_CALL_LIMIT = 299 # call limit per minute
        calls = self.apiCalls
        length = len(calls)
        if length < API_CALL_LIMIT:
            return False

        lastTimeIndex = length - API_CALL_LIMIT
        lastTime = calls[lastTimeIndex]

        oneMinuteAgo = datetime.now() - relativedelta(seconds = 65) # datetime.now()
        # if the last time is more recent than one minute ago
        return lastTime >= oneMinuteAgo

    def waitForMoreApiCalls(self):
        if self.overApiCallLimit():
            print("Waiting for more API calls to be available...")
            time.sleep(65)

    def getNumApiCalls(self):
        return len(self.apiCalls)

    def request(self, url):
        print("Making API call request...")
        self.waitForMoreApiCalls()
        r = requests.get(url, params=self.params)

        if r.status_code != 200:
            raise Exception('API did not return a valid response')
        else:
            self.apiCalls.append(datetime.now())

        return r

    def getDataFrameUnordered(self, endpoint): # makes api calls (helper method)
        r = self.request(endpoint)

        try:
            df = pd.json_normalize(r.json())
        except ValueError:
            raise Exception("Bad URL or API Key!")

        return df

    def getDataFrame(self, endpoint): # makes api calls (helper method)
        df = self.getDataFrameUnordered(endpoint)

        if df.empty:
            raise ValueError("Error, empty dataframe")

        # df = df.reindex(index=df.index[::-1])
        df.sort_values(by=['date'], inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def getRatios(self, ticker):
        endpoint = self.base_url + 'api/v3/ratios/' + ticker
        wantedCols = [
            "currentRatio", "quickRatio", "grossProfitMargin", "operatingProfitMargin", "returnOnAssets",
            "returnOnEquity", "returnOnCapitalEmployed", "debtEquityRatio", "priceToBookRatio", "priceToSalesRatio",
            "priceEarningsRatio", "priceEarningsToGrowthRatio", "priceToOperatingCashFlowsRatio"
        ]
        df = self.getDataFrame(endpoint)
        df = df[self.alwaysCols + wantedCols]
        return df

    def getGrowths(self, ticker): # makes api call
        endpoint = self.base_url + 'api/v3/income-statement-growth/' + ticker
        wantedCols = [
            "growthRevenue", "growthCostOfRevenue", "growthGrossProfit", "growthGrossProfitRatio",
            "growthOperatingExpenses", "growthEBITDA", "growthOperatingIncome", "growthNetIncome",
            "growthEPS"
        ]
        df = self.getDataFrame(endpoint)
        df = df[self.alwaysCols + wantedCols]
        return df

    def getValues(self, ticker): # makes API call
        endpoint = self.base_url + 'api/v3/enterprise-values/' + ticker
        wantedCols = [
            "marketCapitalization", "enterpriseValue"
        ]
        df = self.getDataFrame(endpoint)
        df = df[self.alwaysCols + wantedCols]
        return df

    def getFinStat(self, ticker, limit): # makes API call
        self.params['limit'] = limit
        self.params['period'] = 'quarter'
        endpoint = self.base_url + 'api/v3/income-statement/' + ticker
        return self.getDataFrame(endpoint)

    def getSamples(self, ticker): # makes API calls
        ratios = self.getRatios(ticker)
        growths = self.getGrowths(ticker)
        values = self.getValues(ticker)

        merge = pd.merge(ratios, growths, how='inner', left_on=["date", "symbol"], right_on=["date", "symbol"])
        merge = pd.merge(merge, values, how='inner', left_on=["date", "symbol"], right_on=["date", "symbol"])

        return merge

    def getRandTickers(self, num): # makes api calls
        endpoint = self.base_url + 'api/v3/available-traded/list'
        tickers = pd.DataFrame(self.request(endpoint).json())
        rowIdx = random.sample(range(0, len(tickers)), num)

        return list(tickers.iloc[rowIdx]['symbol'])

    def getSamplesFromTickers(self, tickers, samplesPerTicker=5, maxTickers=10): # makes API calls
        '''
        tickers: list Creates dataframe of samples from list of tickers
        samplesPerTicker: int Number of rows of data to get per ticker
        '''
        print("\nGetting X data...")

        result = pd.DataFrame()
        i = 0

        for t in tickers:
            if i >= maxTickers:
                break

            try:
                oneYearAgo = (datetime.today() - relativedelta(years=1, months=6)).strftime('%Y-%m-%d')
                df = self.getSamples(t)
                df = df[df['date'] < oneYearAgo]
                rowIdx = random.sample(range(0, len(df)), samplesPerTicker)
            except ValueError:
                # Ticker didn't have enough data
                print("Ticker didn't have enough data.")
                continue

            samples = df.iloc[rowIdx]
            result = result.append(pd.DataFrame(samples))
            i += 1

        return result

    def getYFromDF(self, X):
        print("\nGetting Y data...")

        groups = X.groupby(['symbol'])
        result = pd.DataFrame()

        for state, frame in groups:
            dates = list(frame['date'])
            try:
                priceChange = self.getPriceChange(state, dates)
                print(priceChange.shape)
            except ValueError:
                # No price data for stock
                print("Ticker {} didn't have price data".format(state))
                continue

            result = result.append(priceChange)

        return result

    def getPriceChange(self, ticker, dates): # makes api call
        # Currently returns price in dollars instead of percentage

        endpoint = self.base_url + 'api/v3/historical-price-full/' + ticker
        self.params['serietype'] = 'line'
        r = self.request(endpoint)
        del self.params['serietype']

        if not r.json():
            raise ValueError("No price data for stock")

        addOneYear = lambda d: (datetime.strptime(d, "%Y-%m-%d") + relativedelta(years=1)).strftime('%Y-%m-%d')
        subOneYear = lambda d: (datetime.strptime(d, "%Y-%m-%d") - relativedelta(years=1)).strftime('%Y-%m-%d')

        futureDates = [addOneYear(d) for d in dates]

        priceDataPresent = [o for o in r.json()['historical'] if o['date'] in dates]
        priceDataFuture = [o for o in r.json()['historical'] if o['date'] in futureDates]

        combinedPriceData = {}
        for o in (priceDataPresent + priceDataFuture):
            try:
                combinedPriceData[o['date']] = o['close']
            except Exception as e:
                print("Bad object was:")
                print(o)
                print('From:')
                print(priceDataPresent + priceDataFuture)
                print(e)
                continue

        endData = {'symbol': [], 'date': [], 'futureDate': [], 'futureClose': [], 'percentage': []}

        for i, date in enumerate(dates):

            futureDate = addOneYear(date)

            if date not in combinedPriceData or futureDate not in combinedPriceData:
                print('Date {} not found for {}'.format(date, ticker))
                continue
            else:
                print('Date found for {}'.format(ticker))

            endData['date'].append(date)
            endData['futureDate'].append(futureDate)
            endData['futureClose'].append(combinedPriceData[futureDate])
            endData['symbol'].append(ticker)

            percentageChange = (combinedPriceData[futureDate] - combinedPriceData[date]) / combinedPriceData[date]
            endData['percentage'].append(percentageChange)

        return pd.DataFrame(endData)
