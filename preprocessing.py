# imports
import re
from string import punctuation

# class
class Preprocessing():
    # variables
    def __init__(self):
        self.tweets_processed = []
        self.tweets_splitted = []
        self.kelas = []

    # methods
    def normalization(self,sentence):
        return sentence

    def Remove_Punctuation(self,sentence):
        return ''.join(char for char in sentence if char not in punctuation)

    def Remove_Tweet_Mention(self,sentence):
        return ' '.join(char for char in sentence.split() if not char.startswith('@'))

    def Remove_Hyperlink(self,text):
        text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
        return text

    def Process_Tweet(self,tweets):
        for tweet in tweets:
            str1 = self.Remove_Hyperlink(tweet)
            str2 = self.Remove_Tweet_Mention(str1)
            str_final = self.Remove_Punctuation(str2)
            self.tweets_splitted.append(str_final.split())
            self.tweets_processed.append(str_final)
    
    def Process_Class(self,data_tweet):
        i = 0
        # print(data_tweet['Keluhan'])
        for i in data_tweet.index:
            if (data_tweet['Keluhan'][i] == 'Ya'):
                self.kelas.append('Keluhan')
            elif(data_tweet['Respon'][i] == 'Ya'):
                self.kelas.append('Respon')
            else:
                self.kelas.append('Netral')
        #     i+=1