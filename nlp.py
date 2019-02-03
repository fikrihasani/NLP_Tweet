# imports
import pandas as pd
import re
from string import punctuation
from pandas import ExcelWriter
from pandas import ExcelFile

# variables
tweets_processed = []
tweets_splitted = []
kelas = []

# methods
def normalization(sentence):
    return sentence

def Remove_Punctuation(sentence):
    return ''.join(char for char in sentence if char not in punctuation)

def Remove_Tweet_Mention(sentence):
    return ' '.join(char for char in sentence.split() if not char.startswith('@'))

def Remove_Hyperlink(text):
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
    return text

def Process_Tweet(tweets):
    rows = 0
    for tweet in tweets:
        if rows == 100:
            break
        str1 = Remove_Hyperlink(tweet)
        str2 = Remove_Tweet_Mention(str1)
        str_final = Remove_Punctuation(str2)
        tweets_splitted.append(str_final.split())
        tweets_processed.append(str_final)
        rows += 1

# main
if __name__ == "__main__":
    # read excel file
    tw_data = pd.read_excel("Ica_Labelled_Tweets.xlsx",sheet_name="tweets_text")
    # get column
    columns = tw_data.columns
    # get tweet
    tweets = tw_data['Tweet']
    # input class to array
    for keluh in tw_data['Keluhan']:
        if (keluh == 'Ya'):
            kelas.append(1)
        else:
            kelas.append(0)
    # cek
    print(kelas)
    # process tweet
    Process_Tweet(tweets)
    #cek isi data awal
    print(tweets,"\n-----------------------------------------------------\n")
    i = 0
    # cek isi array
    print("Processed array: ")
    for tweet in tweets_processed:
        print(str(i)+" - "+tweet+"\n")
        i+=1
    # cek isi array
    print("Splitted Array: ")
    for tweet in tweets_splitted:
        print(tweet)
        i+=1
