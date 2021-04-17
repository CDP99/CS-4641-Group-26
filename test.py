import requests
import pandas as pd

r = requests.get('https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?serietype=line&apikey=b903b84d1ce1092d3e9fca34e25bf408')
a = r.json()
print(pd.DataFrame(a))