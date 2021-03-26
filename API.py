import requests
import pandas as pd

class API:
    def __init__(self, key):
        self.base_url = 'https://financialmodelingprep.com/'
        self.params = {
            'apikey': key
        }

    def request(self, url):
        r = requests.get(url, params=self.params)

        if r.status_code != 200:
            raise Exception('API did not return a valid response')

        return r

    def getDataFrame(self, endpoint): 
        r = self.request(endpoint)

        try:
            df = pd.json_normalize(r.json())
        except JSONDecodeError:
            raise Exception("Bad URL or API Key!")

        df = df.reindex(index=df.index[::-1])
        return df

    def getRatios(self, ticker):
        endpoint = self.base_url + 'api/v3/ratios/' + ticker
        return self.getDataFrame(endpoint)