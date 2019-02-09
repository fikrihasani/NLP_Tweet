from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ekstraksi fitur TFIDF
class Data_Processing:
    def __init__(self,tweet,kelas):
        self.tweet = tweet
        self.kelas = kelas
        self.X_train = self.X_test = self.y_pred = self.y_train = self.y_test = None

    def set_all(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tweet, self.kelas, test_size=0.2)

    def cek_algorithm_result(self,y_test,y_pred):
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred))  

# Naive Bayes
class Naive_Bayes:
    def __init__(self,data):
        self.acc = 0
        self.model = None
        self.classifier = None
        self.pred = None
        self.vectorized = CountVectorizer()
        self.tf_idf = TfidfTransformer()
        self.data = data
    
    def train(self):
        X_train_counts = self.vectorized.fit_transform(self.data.X_train)

        # ekstraksi fitur tf idf
        X_train_tfidf = self.tf_idf.fit_transform(X_train_counts)
        
        # train
        self.classifier = MultinomialNB()  
        self.classifier.fit(X_train_tfidf, self.data.y_train)

        # test
        X_test = self.vectorized.transform(self.data.X_test)
        X_test_tfidf = self.tf_idf.transform(X_test)
        self.y_pred = self.classifier.predict(X_test_tfidf) 

        # get accuracy 
        self.acc = accuracy_score(self.data.y_test, self.y_pred)
        print("Akurasi Naive Bayes: ",self.acc)  
    
    def classify(self,docs):
        vectorized_docs = self.vectorized.transform([docs])
        tfidf_docs = self.tf_idf.transform(vectorized_docs)
        pred = self.classifier.predict(tfidf_docs)
        print("Hasil klasifikasi: ",docs,"\n",pred[0])

# Random Forest
class Random_Forest:
    def __init__(self,data):
        self.acc = 0
        self.model = None
        self.classifier = None
        self.pred = None
        self.vectorized = CountVectorizer()
        self.tf_idf = TfidfTransformer()
        self.data = data

    def train(self):
        X_train_counts = self.vectorized.fit_transform(self.data.X_train)

        #ekstraksi fitur        
        X_train_tfidf = self.tf_idf.fit_transform(X_train_counts)

        # train
        self.classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
        self.classifier.fit(X_train_tfidf, self.data.y_train)

        # testing
        X_test = self.vectorized.transform(self.data.X_test)
        X_test_tfidf = self.tf_idf.transform(X_test)
        self.y_pred = self.classifier.predict(X_test_tfidf) 
        self.acc = accuracy_score(self.data.y_test, self.y_pred)
        print("Akurasi Random Forest: ",self.acc)  

    def classify(self,docs):
        vectorized_docs = self.vectorized.transform([docs])
        tfidf_docs = self.tf_idf.transform(vectorized_docs)
        pred = self.classifier.predict(tfidf_docs)
        print("Hasil klasifikasi: ",docs,"\n",pred[0])

# random forest
class Support_Vector:
    def __init__(self,data):
        self.acc = 0
        self.model = None
        self.classifier = None
        self.pred = None
        self.vectorized = CountVectorizer()
        self.tf_idf = TfidfTransformer()
        self.data = data

    def train(self):
        X_train_counts = self.vectorized.fit_transform(self.data.X_train)

        #ekstraksi fitur        
        X_train_tfidf = self.tf_idf.fit_transform(X_train_counts)

        # train
        self.classifier = SVC(kernel="linear")  
        self.classifier.fit(X_train_tfidf, self.data.y_train)

        # testing
        X_test = self.vectorized.transform(self.data.X_test)
        X_test_tfidf = self.tf_idf.transform(X_test)
        self.y_pred = self.classifier.predict(X_test_tfidf) 
        self.acc = accuracy_score(self.data.y_test, self.y_pred)
        print("Akurasi SVM: ",self.acc)  

    def classify(self,docs):
        vectorized_docs = self.vectorized.transform([docs])
        tfidf_docs = self.tf_idf.transform(vectorized_docs)
        pred = self.classifier.predict(tfidf_docs)
        print("Hasil klasifikasi: ",docs,"\n",pred[0])