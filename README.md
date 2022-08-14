# Sentiment Analysis of Tweets to understand major issues of a product or company 

Dataset: Twitter US Airline Sentiment
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment


In this particular project, we perform sentiment analysis on the Twitter US Airline dataset through machine learning (Logistic Regression & Random Forest). Sentiment analysis (or opinion mining) is simply a natural language processing technique used to determine whether data is positive, negative, or neutral. Companies can use sentiment analysis to gauge customer satisfaction and obtain insightful customer feedback. 


#
# Getting Started

First of all login from your [Twitter Developer Account](https://developer.twitter.com/en) and create a new project/app from your dashboard ([how to create Twitter App](https://medium.com/@divyeshardeshana/create-twitter-developer-account-app-4ac55e945bf4)). Now get keys and tokens w/ *elevated access* and paste your Consumer Key, Consumer Secret, Access Token and Access Token Secret in the code. 

Remember to get the username or screen name for the airline whose tweets you want to access. 

# Build With 

- Python 3.6 
- tweepy 
- numpy 
- pandas
- seaborn (for data visualization) 
- re (for regular expressions) 
- nltk (natural language toolkit) 
- matplotlib 


# Resources 

\#Bag-of-words 
- https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/
- https://en.wikipedia.org/wiki/Bag-of-words_model
- https://www.youtube.com/watch?v=IKgBLTeQQL8&ab_channel=KrishNaik

\#Stopwords 
- https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

\#TF-IDF (Term Frequency-Inverse Document Frequency)
- https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/
- https://www.youtube.com/watch?v=D2V1okCEsiE&ab_channel=KrishNaik

\#Test-Train split (sklearn.model_selection.train_test_split) 
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html 
 
![0 DKB-pJy7-G6gEkM-](https://user-images.githubusercontent.com/91340952/184548646-103a9caa-4f16-42b2-8dfe-5678dccaf559.png)

[*image source*](https://towardsdatascience.com/understanding-train-test-split-scikit-learn-python-ea676d5e3d1)


\#Logistic Regression (sklearn.linear_model.LogisticRegression)
- https://www.geeksforgeeks.org/understanding-logistic-regression/
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://www.javatpoint.com/logistic-regression-in-machine-learning


\#Random Forest
- *short video on ensemble learning* https://youtu.be/LNrBcDfUhq0
- https://www.ibm.com/cloud/learn/random-forest
- https://youtu.be/WkFtIqWmX9o 
-https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


\#Extraction of tweets
- https://www.geeksforgeeks.org/extraction-of-tweets-using-tweepy/
- There are two types of authentication-- the first one is authentication which uses the consumer key and consumer secret to identify this client and be sure that it is a valid account. The second one is called authorization which allows the resources server to identify which type of actions you have the permission to do with data, or what we call a resource, and this operation uses access token and access token secret.
