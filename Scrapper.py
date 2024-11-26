import tweepy
import csv
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense



consumer_key= '******'
consumer_secret= '*****'
access_token='*****'
access_token_secret='*****'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


public_tweets = api.search('company_name')



for tweet in public_tweets:    
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    

#data collection
dates = []
prices = []
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return


get_data('Twitter_data.csv')


def predict_prices(dates, prices, x):

predicted_price = predict_price(dates, prices, 29)
print(predicted_price)
