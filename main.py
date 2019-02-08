# imports
from preprocessing import Preprocessing
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from classifier import Random_Forest,Naive_Bayes

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
    # 
    prepro.Process_Tweet(tweets)
    
    # input class to array
    prepro.Process_Class(tw_data)
    # classify
    rf = Naive_Bayes(tweets,prepro.kelas)
    rf.train()
    rf.classify(["@diskamtam bapak/ibu mau tanya, kalo pemeliharaan taman2 yg banyak dibagun skr gimana nantinya?"])


 