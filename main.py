# scripting language : Python 3.6
# modules : pandas, numpy, sklearn, bs4, jieba 

import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

import re
from bs4 import BeautifulSoup
import jieba
import jieba.analyse

# load data to dataframe
df_train = pd.read_csv("data/offsite-tagging-training-set.csv",  header=0)
df_test = pd.read_csv("data/offsite-tagging-test-set.csv",  header=0)
print("Load {} training examples and {} test cases".format(len(df_train), len(df_test)))

y_train = df_train['tags']
X_train = df_train['text']
X_test = df_test['text']

# label encode the classes
le = LabelEncoder()
y_train = le.fit_transform(y_train)

print("\nShow first training example: \n", X_train[0], "\n\nClass label:", y_train[0])
print("\nClass labels [0,1,2] refer to ",le.inverse_transform([0, 1, 2]))

def build_stopwords(filepath):
    '''
    Load customized stopwords.txt
    Return a dictionary of stopwords
    '''
    with open(filepath,'r') as file:
        return set([line.strip() for line in file])
    
def clean_text(text_array):
    '''
    Remove html tags, symbols, letters a-z A-Z, numbers
    Perform chinese word segmentation
    Remove stopwords 
    Return list of cleaned text
    '''
    start_time = time.time() 
    print("--- Text Preprocessing ---")
    
    cleaned_text_array = []
    
    for i in range(len(text_array)):
        raw_text = text_array[i]
        cleaned_text = BeautifulSoup(raw_text, 'lxml').get_text()  
        chinese_only = re.sub("[0-9a-zA-Z\xa0\r\n\t\u3000\u2000-\u206F\u2E00-\u2E7F\!#$%&()*+,\-.\/:;<=>?@\[\]^_`{|}~]", "", cleaned_text)
        words = list(jieba.cut(chinese_only, cut_all=False))
        words = [word for word in words if str(word) not in stopwords]
        cleaned_text_array.append(" ".join(words))
        
    print("--- %s seconds --- \n" % (time.time() - start_time))
    
    return cleaned_text_array

# build dictionary of stopwords from customized stopwords file
stopwords = build_stopwords(filepath='stopwords.txt')

# text preprocessing
X_train_cleaned = clean_text(X_train)
X_test_cleaned = clean_text(X_test)

print("\nShow first training example in raw text: \n", X_train[0])
print("\nShow first training example after text preprocessing: \n", X_train_cleaned[0])


# create Bag of Words using tf-idf transform
print("--- Create Bag of Words using TfidfVectorizer ---")

start_time = time.time() 
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False, min_df=100) #minumum document frequency set to 100
X_train_tfidf = tfidf.fit_transform(X_train_cleaned)

print("--- %s seconds --- \n" % (time.time() - start_time))

print("Number of words in dictionary : {}".format(len(tfidf.get_feature_names())))
print("Show some words in dictionary : \n", tfidf.get_feature_names()[::100])


# grid search for best paramter set of random forest classifier
print("--- Training Random Forest Classifier ---")
rf = RandomForestClassifier()
params = {"max_depth":[None, 10, 50],  #max depth of trees
          "n_estimators":[10, 100]  #no. of tress to ensemble
         } 

model = GridSearchCV(estimator=rf, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1, verbose=0)
model.fit(X_train_tfidf, y_train)

print("\nThe best paramenter set is : \n", model.best_params_)
print("\nScores:")
for scores in model.grid_scores_:
    print(scores)
    
    
# make prediction on test set
print("--- Making prediction on test set ---")
prediction = model.predict(tfidf.transform(X_test_cleaned))
prediction_le = le.inverse_transform(prediction)
solution = pd.DataFrame(prediction_le, df_test['id'], columns = ["tag"])
solution.to_csv("prediction.csv")

print("Prediction on all test cases saved in prediction.csv\n")

print("Show prediction on all test cases:")
for i in range(len(X_test)):
    print(prediction_le[i],":", X_test[i][0:20], "...")