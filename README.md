# Chinese Text Classification

#### Task : 
* Supervised chinese text classification (in news context, traditional Chinese) 

#### Dataset : 
* Training set : ~4000 labeled Chinese news articles (3 classes)
* Test set : ~1000 unlabeled Chinese news articles

#### Algorithm :
* Text Preprocessing and word segmentation using beautifulsoup4, jieba and customized stopwords.
* Create Bag of Words using TfidfVectorizer in sklearn.
* Train Random Forest Classifier and make prediction on test set.

#### Evaluation :
* Average of 0.99 accuracy on 10-fold CV.

---
### Requirements
* [Python 3.6](https://www.python.org/downloads/)
* Modules: pandas, numpy, scikit-learn, beautifulsoup4, jieba 

To install required modules :
$ pip install pandas numpy scikit-learn beautifulsoup4 jieba

### Instruction 
To run python script :
$ python main.py
---
### What is in this repo

*main.py*
* Python script for chinese text classifcation

*tagging-prediction.ipynb*
* Jupyter notebook with the same script

*prediction.csv*
* Prediction on test set

*stopwords.txt*
* Customized chinese stopwords


