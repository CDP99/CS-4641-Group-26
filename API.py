import requests
import pandas as pd
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta

class API:
    def __init__(self, key, period='annual', limit=5):
        self.base_url = 'https://financialmodelingprep.com/'
        self.params = {
            'apikey': key,
            'period': period,
            'limit': limit,
        }
        self.alwaysCols = ["symbol", "date"]

        if limit == None:
            del self.params['limit']

    def request(self, url):
        r = requests.get(url, params=self.params)

        if r.status_code != 200:
            raise Exception('API did not return a valid response')

        return r

    def getDataFrameUnordered(self, endpoint):
        r = self.request(endpoint)

        try:
            df = pd.json_normalize(r.json())
        except ValueError:
            raise Exception("Bad URL or API Key!")

        return df

    def getDataFrame(self, endpoint):
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

    def getGrowths(self, ticker):
        endpoint = self.base_url + 'api/v3/income-statement-growth/' + ticker
        wantedCols = [
            "growthRevenue", "growthCostOfRevenue", "growthGrossProfit", "growthGrossProfitRatio",
            "growthOperatingExpenses", "growthEBITDA", "growthOperatingIncome", "growthNetIncome",
            "growthEPS"
        ]
        df = self.getDataFrame(endpoint)
        df = df[self.alwaysCols + wantedCols]
        return df

    def getValues(self, ticker):
        endpoint = self.base_url + 'api/v3/enterprise-values/' + ticker
        wantedCols = [
            "marketCapitalization", "enterpriseValue"
        ]
        df = self.getDataFrame(endpoint)
        df = df[self.alwaysCols + wantedCols]
        return df

    def getFinStat(self, ticker, limit):
        self.params['limit'] = limit
        self.params['period'] = 'quarter'
        endpoint = self.base_url + 'api/v3/income-statement/' + ticker
        return self.getDataFrame(endpoint)

    def getSamples(self, ticker):
        ratios = self.getRatios(ticker)
        growths = self.getGrowths(ticker)
        values = self.getValues(ticker)

        merge = pd.merge(ratios, growths, how='inner', left_on=["date", "symbol"], right_on=["date", "symbol"])
        merge = pd.merge(merge, values, how='inner', left_on=["date", "symbol"], right_on=["date", "symbol"])

        return merge

    def getRandTickers(self, num):
        endpoint = self.base_url + 'api/v3/available-traded/list'
        tickers = pd.DataFrame(self.request(endpoint).json())
        rowIdx = random.sample(range(0, len(tickers)), num)

        return list(tickers.iloc[rowIdx]['symbol'])

    def getSamplesFromTickers(self, tickers, samplesPerTicker=5, maxTickers=10):
        '''
        tickers: list Creates dataframe of samples from list of tickers
        samplesPerTicker: int Number of rows of data to get per ticker
        '''
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
                continue

            samples = df.iloc[rowIdx]
            result = result.append(pd.DataFrame(samples))
            i += 1

        return result

    def getYFromDF(self, X):
        groups = X.groupby(['symbol'])
        result = pd.DataFrame()

        for state, frame in groups:
            dates = list(frame['date'])
            try:
                priceChange = self.getPriceChange(state, dates)
            except ValueError:
                # No price data for stock
                continue

            result = result.append(priceChange)

        return result

    def getPriceChange(self, ticker, dates):

        #Currently returns price in dollars instead of percentage

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


        endData = {'symbol': [], 'date': [], 'futureDate': [], 'futureClose': [], 'percentage': []}

        indexPricePresent = 0
        for o in priceDataFuture:

            previousYearPrice = priceDataPresent[indexPricePresent]

            endData['date'].append(subOneYear(o['date']))
            endData['futureDate'].append(o['date'])
            endData['futureClose'].append(o['close'])
            endData['symbol'].append(ticker)

            percentageChange = (o['close'] - previousYearPrice['close'])/previousYearPrice['close']
            endData['percentage'].append(percentageChange)
            print("Percentage")
            print(percentageChange)
            print("")
            indexPricePresent = indexPricePresent + indexPricePresent

        return pd.DataFrame(endData)
