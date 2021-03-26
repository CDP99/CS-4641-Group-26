import requests
import pandas as pd

class API:
    def __init__(self, key, period='annual', limit=5):
        self.base_url = 'https://financialmodelingprep.com/'
        self.params = {
            'apikey': key,
            'period': period,
            'limit': limit,
        }
        self.alwaysCols = ["symbol", "date"]

    def request(self, url):
        r = requests.get(url, params=self.params)

        if r.status_code != 200:
            raise Exception('API did not return a valid response')

        return r

    def getDataFrame(self, endpoint): 
        r = self.request(endpoint)

        try:
            df = pd.json_normalize(r.json())
        except ValueError:
            raise Exception("Bad URL or API Key!")

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