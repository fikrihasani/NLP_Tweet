from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ekstraksi fitur tf idf
class feature_extraction:
    def __init__(self,tweet,kelas):
        self.tweet = tweet
        self.kelas = kelas
        self.X_train = self.X_test = self.y_pred = self.y_train = self.y_test = None

    def set_all(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(self.tweet)

        # ekstraksi fitur
        tf_idf = TfidfTransformer()
        X_train_tfidf = tf_idf.fit_transform(X_train_counts)

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train_tfidf, self.kelas, test_size=0.2)

# naive bayes
class Naive_Bayes:
    def __init__(self,tweet,kelas):
        self.tweet = tweet
        self.kelas = kelas
        self.acc = 0
        self.model = None
        self.classifier = None
        self.pred = None
    
    def train(self):
        # train data
        fe = feature_extraction(self.tweet,self.kelas)
        # set all feature
        fe.set_all()
        # train
        self.classifier = MultinomialNB()  
        self.classifier.fit(fe.X_train, fe.y_train)  
        self.y_pred = self.classifier.predict(fe.X_test)  
        # # check confusion matrix
        # print(confusion_matrix(fe.y_test,fe.y_pred))  
        # print(classification_report(fe.y_test,fe.y_pred))  
        self.acc = accuracy_score(fe.y_test, self.y_pred)
        print("Akurasi naive bayes: ",self.acc)  
    
    def classify(self,docs):
        print(docs)
        print(type(docs))
        vectorized_docs = CountVectorizer().fit_transform(docs)
        print(type(vectorized_docs))

# random forest
class Random_Forest:
    def __init__(self,tweet,kelas):
        self.tweet = tweet
        self.kelas = kelas
        self.acc = 0
        self.model = None
        self.classifier = None
        self.pred = None

    def train(self):
        # train data
        fe = feature_extraction(self.tweet,self.kelas)
        # set all feature
        fe.set_all()
        # train
        self.classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
        self.classifier.fit(fe.X_train, fe.y_train)  
        self.y_pred = self.classifier.predict(fe.X_test)  
        # # check confusion matrix
        # print(confusion_matrix(fe.y_test,fe.y_pred))  
        # print(classification_report(fe.y_test,fe.y_pred))  
        self.acc = accuracy_score(fe.y_test, self.y_pred)
        print("Akurasi Random Forest: ",self.acc)  

    def classify(self,docs):
        print(type(docs))
        vectorized_docs = CountVectorizer().fit_transform(docs)
        print(type(vectorized_docs))
        # self.y_pred = self.classifier.predict(vectorized_docs)
        # print("Hasil klasifikasi: ",docs,"\n",self.y_pred)