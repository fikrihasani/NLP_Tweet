# imports
from preprocessing import Preprocessing
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from classifier import Random_Forest,Naive_Bayes,Support_Vector
from classifier import Data_Processing
import os

# public var
prepro = Preprocessing()
nb = Naive_Bayes()
rf = Random_Forest()
sv = Support_Vector()

# train model
def train_classifier():
    # read excel file
    tw_data = pd.read_excel("Ica_Labelled_Tweets.xlsx",sheet_name="tweets_text")

    # init kelas 'Keluhan, Respon, Netral'
    prepro.Process_Class(tw_data)

    # get original tweet
    tweets = tw_data['Tweet']

    # preprocessing tweet data using DIY preprocessing method (by M. Fikri Hasani)
    prepro.Process_Tweet(tweets)

    # data = Data_Processing(tweets,prepro.kelas) # split data using original tweets (no preprocessing)
    data = Data_Processing(prepro.tweets_processed,prepro.kelas) # split data using DIY preprocessing method (by M. Fikri Hasani)

    # set all feature
    data.set_all()
    
    # classify (cl stand for classifier)
    nb.train(data)
    rf.train(data)
    sv.train(data)
    
# main
if __name__ == "__main__":
    # if model exist then load
    if (os.path.exists('./Model/'+nb.filenameModel) and os.path.exists('./Model/'+rf.filenameModel) and os.path.exists('./Model/'+sv.filenameModel)):
        nb.load_data()
        rf.load_data()
        sv.load_data()
    else:
        # if not then train
        train_classifier()
   
    # inference
    ch = input("\nPilihan:\n1. Input String dari Pengguna\n2. Input String dari file Text (inference.txt)\n3. Exit\nMasukkan Pilihan anda: ")
    if ch == "1":
        words = input("Masukkan kalimat untuk diklasifikasi: ") 
        print("Klasifikasi dengan Naive Bayes: ")
        nb.classify(words)
        print("Klasifikasi dengan Random Forest: ")
        rf.classify(words)
        print("Klasifikasi dengan SVM: ")
        sv.classify(words)
    elif ch == '2':
        if os.path.exists("inference.txt"):
            f = open('inference.txt','r')
            x = f.read()
            x = x.splitlines()
            # print(x)
            # print(x)
            f.close() 
            for line in x:
                print("Klasifikasi dengan Naive Bayes: ")
                nb.classify(line)
                print("Klasifikasi dengan Random Forest: ")
                rf.classify(line)
                print("Klasifikasi dengan SVM: ")
                sv.classify(line)
                print()
    elif ch == '3':
        SystemExit()

