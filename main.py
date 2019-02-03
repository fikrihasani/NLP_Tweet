# imports
from preprocessing import Preprocessing
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

# main
if __name__ == "__main__":
    # initiate class Preprocessing
    prepro = Preprocessing()
    # read excel file
    tw_data = pd.read_excel("Ica_Labelled_Tweets.xlsx",sheet_name="tweets_text")
    # get column
    columns = tw_data.columns
    # get tweet
    tweets = tw_data['Tweet']
    # input class to array
    prepro.Process_Class(tw_data['Keluhan'])
    # cek
    print(prepro.kelas)
    # process tweet
    prepro.Process_Tweet(tweets)
    #cek isi data awal
    print(tweets,"\n-----------------------------------------------------\n")
    i = 0
    # cek isi array
    print("Processed array: ")
    for tweet in prepro.tweets_processed:
        print(str(i)+" - "+tweet+"\n")
        i+=1
    # cek isi array
    print("Splitted Array: ")
    i = 0
    for tweet in prepro.tweets_splitted:
        print(tweet)
        i+=1
