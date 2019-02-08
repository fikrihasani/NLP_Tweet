# imports
from preprocessing import Preprocessing
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from classifier import Random_Forest,Naive_Bayes,Support_Vector

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
    # preprocessing tweet data
    # prepro.Process_Tweet(tweets)
    
    # input class to array
    prepro.Process_Class(tw_data)
    # print(prepro.kelas.size)
    # classify (cl stand for classifier)
    cl = Support_Vector(tweets,prepro.kelas)
    cl.train()
    # inference 
    # rf.classify("@ridwankamil @dbmpkotabdg kang teman saya tertimpa pohn dijln sangkuriang dpn polsek coblong tlg ditertibkan phn yg sdh lapuknuhun")


 