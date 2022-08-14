# -*- coding: utf-8 -*-
"""Sentiment Analysis of Tweets.ipynb

"""

import tweepy
import numpy as np                 # used for framework
import pandas as pd                # used for data analysis
import seaborn as sns              # used for data visualization 
import re                          # used for regular expressions
import nltk                        # natural lanuage tool kit
import matplotlib.pyplot as plt    # used for graphs 
get_ipython().magic('matplotlib inline')

nltk.download('stopwords')
from nltk.corpus import stopwords

data_source_url = "https://raw.githubusercontent.com/AbhinavBahuguna2002/Sentiment-Analysis-of-Tweets/main/Tweets.csv" 
# Linking the dataset
airline_tweets = pd.read_csv(data_source_url)
# Data set passed into panda for framework

airline_tweets.head()
# Visualization of top 5 Entries

plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])
# Checking the default plot size

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 
# Changing the def plot size

airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
#Checking the number of tweets for each airline and ploting a pie chart for it

airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
#Checking the distribution of sentiments and ploting a pie chart for it

airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')
# Checking the distribution of sentiments in each of the airlines and ploting a bar graph for the same

sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=airline_tweets)
# Checking the confidence of the sentiments and ploting a bar graph for it
# Visualization OF Data is Completed

features = airline_tweets.iloc[:, 10].values 
labels = airline_tweets.iloc[:, 1].values
# Extraction of columns for Features and Lables variabes

processed_features = []
# Creating Empty Array

for sentence in range(0, len(features)):
# Loop For Data Cleansing
    
    # Removing all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # Removing all the single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Removing single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Conversion to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))

# Vectorize the most repeating words and pass them as features
processed_features = vectorizer.fit_transform(processed_features).toarray()
print(vectorizer.get_feature_names_out())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
# Spliting the Dataset into Train and Test

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logreg = LogisticRegression(random_state=0, max_iter=200)
logreg.fit(X_train, y_train)

pred_logreg = logreg.predict(X_test)

sns.heatmap(confusion_matrix(y_test,pred_logreg))
print(classification_report(y_test,pred_logreg))
print("Accuracy= {:.3f}".format(accuracy_score(y_test, pred_logreg)))

# Logistic Regression
review = "the flight was really uncomfortable"
print(logreg.predict(vectorizer.transform([review]))[0] + '\n')

review = 'the flight went great'
print(logreg.predict(vectorizer.transform([review]))[0])

from sklearn.ensemble import RandomForestClassifier
# Using the Random Forest Approach

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)
# Fitiing the model to Random Forest Classifier

predictions = text_classifier.predict(X_test)
# Predictions For the Random Forest Classifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
sns.heatmap(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

print("Accuracy= {:.3f}".format(accuracy_score(y_test, predictions)))
# Printing the Predictions for Random Forest Classifier
# Printing the Confusion Matrix to visualize the predictions

review = "the flight was really uncomfortable"
print(text_classifier.predict(vectorizer.transform([review]))[0] + '\n')

review = 'the flight went great'
print(text_classifier.predict(vectorizer.transform([review]))[0])

## Accessing Indian airline tweets via Twitter API

"""
Usernames:- 
Air India- airindiain
IndiGo- IndiGo6E
Spice Jet- flyspicejet
Air Asia- AirAsiaIndia

"""

#Twitter API Credentials 
apiKey = ""
apiSecret = ""
accessToken = ""
accessTokenSecret = ""

#creating the authentication object 
authenticate = tweepy.OAuthHandler (apiKey, apiSecret)

#setting access token and access token secret 
authenticate.set_access_token(accessToken, accessTokenSecret)

#creating a api object
api = tweepy.API(authenticate, wait_on_rate_limit= True)

#Extract n tweets from Twitter user 
post = api.user_timeline(screen_name = "airindiain", count= 10, lang = "en", tweet_mode = "extended")
i=1
for tweet in post [0:5]:
  print (str(i) + ')' + tweet.full_text) 
  print('Sentiment= ' + text_classifier.predict(vectorizer.transform([tweet.full_text]))[0] +'\n')
  i = i+1

"""#End of project"""
