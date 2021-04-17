import numpy as np
import pandas as pd
import sys
import os
import subprocess

import json


def getYLabels(filename):
    training_data_path = os.getcwd() + '/' + str(filename)

    names = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']

    stock_data = pd.read_csv(training_data_path, names=names)

    print("Dimensions of file read: ")
    print("Rows")
    print(stock_data.shape[0])

    total_entries = stock_data.shape[0]


    #y-labels in following form:
    #(price_at_year_i, price_after_year_i, percentage change)

    #Initial time period = 1 year
    #On average 252 trading days so use current date + 252 entries to get approximate price one year later

    jsonYLabels = {}
    index = 1
    listDates = []
    listPriceOnDate = []
    listPriceAfterYear = []
    listPercentageChange = []
    for index2, dataPrice in stock_data.iterrows():
        if (index == 0):
            pass

        yLabelEntry = {}
        current_date = stock_data['date'].iloc[index]
        current_price = float(stock_data['close'].iloc[index])

        #Price a year later
        indexOneYearLater = index + 252
        #Check fence post to see if need to subtract 1 to avoid index out of range exception
        if (indexOneYearLater >= total_entries):
            break

        #If got here assume that indexOneYearLater not out of bounds
        price_after_year = float(stock_data['close'].iloc[indexOneYearLater])

        #print("Price after year")
        #print(price_after_year)

        #print("Current Price")
        #print(current_price)

        percentage_change = float((price_after_year - current_price)/current_price)
        #if (percentage_change < 0):
            #print("Negative percentage change")


        #Pass to json
        yLabelEntry['date'] = current_date
        yLabelEntry['price_on_date'] = current_price
        yLabelEntry['price_after_year'] = price_after_year
        yLabelEntry['percentage_change'] = percentage_change

        jsonYLabels[current_date] = yLabelEntry

        #Append to synchronous lists to create pandas dataframe
        listDates.append(current_date)
        listPriceOnDate.append(current_price)
        listPriceAfterYear.append(price_after_year)
        listPercentageChange.append(percentage_change)







        index = index + 1

        #check if new index (a year after) is below total_entries to see when to break

    #Create pandas table:
    d = {'date': listDates, 'price_on_date': listPriceOnDate, 'price_after_year': listPriceAfterYear, 'percentage_change': listPercentageChange}

    df = pd.DataFrame(data=d)

    return jsonYLabels, df



googleFilename = 'AlphabetPrice.csv'

y_labels_google_json, y_labels_google_pd= getYLabels(googleFilename)

print("Pandas DataFrame: ")
print(y_labels_google_pd)



listStocks = []
#10 to train on
appleFilename = 'ApplePrice.csv'
listStocks.append(appleFilename)

microsoftFilename = 'MicrosoftPrice.csv'
listStocks.append(microsoftFilename)

facebookFilename = 'FacebookPrice.csv'
listStocks.append(facebookFilename)

teslaFilename = 'TeslaPrice.csv'
listStocks.append(teslaFilename)

blackrockFilename = 'BlackRockPrice.csv'
listStocks.append(blackrockFilename)

czrFilename = 'CZRPrice.csv'
listStocks.append(czrFilename)

anfFilename = 'ANFPrice.csv'
listStocks.append(anfFilename)

arayFilename = 'ARAYPrice.csv'
listStocks.append(arayFilename)

avavFilename = 'AVAVPrice.csv'
listStocks.append(avavFilename)

agfyFilename = 'AGFYPrice.csv'
listStocks.append(agfyFilename)

res = []
for stock in listStocks:
    json_y_labels, pd_y_labels = getYLabels(stock)
    print("")
    print("")
    print('Stock name: ' + str(stock))
    print('Results for y labels: ')
    print(pd_y_labels)
    res.append(pd_y_labels)

combined = pd.concat(res)

print("Combined:")
print(combined)

#TODO write results to a file and upload code and files to group github


#with open("AlphabetYLabels.json", "w") as file:
    #json.dump(y_labels_google, file, indent=4, sort_keys=True)
    #print("Success dumping Alphabet Y Labels to json file")
