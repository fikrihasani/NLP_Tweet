# imports
from preprocessing import Preprocessing
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from classifier import Random_Forest,Naive_Bayes,Support_Vector
from classifier import Data_Processing

# main
if __name__ == "__main__":
    # initiate class Preprocessing
    prepro = Preprocessing()

    # read excel file
    tw_data = pd.read_excel("Ica_Labelled_Tweets.xlsx",sheet_name="tweets_text")

    # init kelas 'Keluhan, Respon, Netral'
    prepro.Process_Class(tw_data)

    # get column
    columns = tw_data.columns

    # get original tweet
    tweets = tw_data['Tweet']

    # preprocessing tweet data using DIY preprocessing method (by M. Fikri Hasani)
    prepro.Process_Tweet(tweets)

    # data = Data_Processing(tweets,prepro.kelas) # split data using original tweets (no preprocessing)
    data = Data_Processing(prepro.tweets_processed,prepro.kelas) # split data using DIY preprocessing method (by M. Fikri Hasani)

    # set all feature
    data.set_all()

    # inference
    testData = 'Di tribun jabar biasanya suka di post agenda kang emil. Tapi gak setiap hari sih. RT @Ry_ory: Klo saya k Bdg bs ketemu Kang @ridwankamil dmn'

    # classify (cl stand for classifier)
    cl = Naive_Bayes(data)
    cl.train()
    # cl.classify(testData)
    cl = Random_Forest(data)
    cl.train()
    # cl.classify(testData)
    cl = Support_Vector(data)
    cl.train()
    # cl.classify(testData)
