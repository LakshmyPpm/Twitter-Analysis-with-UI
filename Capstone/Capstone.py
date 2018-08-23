
# coding: utf-8

# ### Gathering and saving tweets

# In[1]:


from IPython import get_ipython
IN_JUPYTER =  'get_ipython' in globals() and get_ipython().__class__.__name__ == "ZMQInteractiveShell"

from pkg_resources import resource_filename as fpath
import sys
#sys.path.append(fpath(__name__, ""))    
import nltk
from os.path import join
from IPython import get_ipython
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('punkt', download_dir='/home/sudeep/anaconda/envs/Capstone/nltk_data')
#nltk.download('stopwords', download_dir='/home/sudeep/anaconda/envs/Capstone/nltk_data')
nltk.data.path.append( "/Capstone/nltk_data/")
import tweepy
from tweepy import OAuthHandler
import jsonpickle
import json
import preprocessor as preproc
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import os
from xgboost import XGBClassifier as XGBoostClassifier
filename    = 'finalized_model.pkl'
config_file =  os.path.join(os.getcwd(), "config/config.json")
if not IN_JUPYTER :
 filename    = fpath(__name__, "finalized_model.pkl")
 config_file = fpath(__name__, "config/config.json")
import gensim.models.keyedvectors as word2vec    
GLOVE_DIR =  os.path.join(os.getcwd(), "model/")
if not IN_JUPYTER :
 GLOVE_DIR = fpath(__name__, "model/")


# ### Data Preprocessing

# In[2]:


class TwitterData_Preproces():
    
    if not IN_JUPYTER:
        import json
        import preprocessor as preproc
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer
        import pandas as pd
       
    def __init__(self,file_name):
        self.file_name=file_name
        self.Sentiment = []
        self.data = []
        self.tweets=[]
        
 #preprocessing for text       
    def  preprocessing(self,in_txt): 
        #print "Processing file...\n"     
        #remove  Not available      
        if in_txt  == "Not Available":        
         return ''
        #strip if any double quotes        
        in_txt=str(in_txt).strip('"') 
        in_txt=str(in_txt).strip('' '')
        #use twitter preprocessing library to clean up
        process_text= preproc.clean(in_txt)         
        #print in_txt
        #remove unicoded text       
        #process_text=process_text.encode('utf-16')
        #print process_text    
        #stemming 
        ps = PorterStemmer()
        ps.stem(process_text) 
        #tokenize the word        
        word_tokens = word_tokenize(process_text)  
        #remove stop words
        stop_words = set(stopwords.words('english'))   
        filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words ]         
        return( " ".join( filtered_sentence ))   
  
    #processing for json    
    def json_process(self): 
        print "Processing Json file...\n"            
        self.data = open(self.file_name, "r")
        for line in self.data:             
             try:
            # Read in one line of the file, convert it into a json object 
                tweet = json.loads(line.strip())
            #if 'text' in tweet:
                tweet_string=tweet['text'] 
                if isinstance(tweet_string, unicode):
                     tweet_string= tweet_string.encode('utf-8')
                 #print  tweet_string  
                     self.tweets.append(self.preprocessing( tweet_string ))                      
             except:
            # read in a line is not in JSON format (sometimes error occured)
                continue
        print "Processing Json file complete...\n"         
        return self.tweets
    
    #processing for csv  
    def Csv_process(self):
        
        print "Processing Csv file...\n" 
        #doing clean up to remove null values
        #'./sanders-twitter-0.2/amazon_cells_labelled.csv'
        self.data = pd.read_csv(self.file_name ,header=None,error_bad_lines=False,dtype=object)
        # Keeping only the neccessary columns
        self.data.columns = ["Sentiment","TweetText"]
        self.data = self.data[self.data.Sentiment !="irrelevant" ]
        self.Sentiment = np.asarray(self.data["Sentiment"])
        for line in self.data['TweetText']:
            try:
        #if 'text' in tweet:
                tweet_string=line    
                #if isinstance(tweet_string, unicode):
                #tweet_string= tweet_string.encode('utf-8')
                 #print  tweet_string                    
                self.tweets.append(self.preprocessing( tweet_string )) 
            except:
            # (sometimes error occured)
                continue
        #any null value removed  
        indices = [i for i, x in enumerate(self.tweets) if ( x == '' or x.isdigit() ) ]
        for i in sorted(indices, reverse=True):
              self.tweets = np.delete(self.tweets, i, axis=0)
              self.Sentiment=np.delete(self.Sentiment, i, axis=0)
        print "Processing Csv file complete...\n"       
        return self.tweets,self.Sentiment


# ##Cleaning for bag of words

# In[3]:


class TwitterData_BagOfWords():
    
    def __init__(self,tweets_train,tweets_test):
        self.tweets_train=tweets_train
        self.tweets_test = tweets_test
        
    def Data_BagOfWords(self):
            print "Creating the bag of words...\n"
    # Initialize the "CountVectorizer" object
            vectorizer = CountVectorizer(ngram_range=(1,  2),analyzer = "word",                                            min_df=.0025, max_df=.1, max_features=250 ) 

            tweets_array=np.asarray(self.tweets_train)
            tweets_testarray=np.asarray(self.tweets_test)
            train_data_features = vectorizer.fit_transform(tweets_array)
            test_data_features = vectorizer.transform(tweets_testarray)

    # Numpy arrays are easy to work with, so convert the result to an array 
            train_data_features = train_data_features.toarray()
            test_data_features = test_data_features.toarray()
            tfidf_transformer = TfidfTransformer(use_idf=False)
            X_train_tfidf = tfidf_transformer.fit_transform(train_data_features)
            X_test_tfidf= tfidf_transformer.transform(test_data_features)
            terms = np.array(vectorizer.get_feature_names())
            #return train_data_features,test_data_features
            return X_train_tfidf,X_test_tfidf,terms


# #### Classifier
# 

# In[4]:


class Def_Classifier(object):
       
    def __init__(self,X_train, y_train, X_test, y_test, in_classifier,in_save=False):
        self.X_train=X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.in_classifier = in_classifier
        self.in_save = in_save
        
    def class_pred(self):
        print "Training and Testing the classifier...\n"
        #list_of_labels = sorted(list(set(y_train)))
        list_of_labels = (self.y_train)
        model = in_classifier.fit(self.X_train, self.y_train)
        if self.in_save: 
            f = open(filename, 'w')
            pickle.dump(model, f) 
            f.close()     
        predictions = model.predict(self.X_test)

        precision = precision_score(self.y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
        recall = recall_score(self.y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='weighted', labels=np.unique(self.y_test))
        return precision, recall, accuracy, f1


# ### Experiment -1 Naive Bayes

# In[10]:


if IN_JUPYTER :
    #Calling functions to process train data./trainingandtestdata/
    tweet_preprocess=TwitterData_Preproces('./sanders-twitter-0.2/file.csv')
    data_train ,Sentiment_train = tweet_preprocess.Csv_process()


# In[6]:


if IN_JUPYTER :
    #splitting the data
    data_train, data_test, Sentiment_train, Sentiment_test = train_test_split(data_train, Sentiment_train, test_size=0.2, random_state=42)


# In[7]:


if IN_JUPYTER :
    #Calling bag of wordTwitterData_BagOfWordss
    twitter_BagOfWords = TwitterData_BagOfWords(data_train,data_test)
    BagofWord_Train,BagofWord_Test,feature_array = twitter_BagOfWords.Data_BagOfWords()   
    #print BagofWord_Train.shape
    #print BagofWord_Test.shape
    #print len(Sentiment_train)
    #print len(Sentiment_test)


# ### Visualization

# In[9]:


if IN_JUPYTER :
    get_ipython().magic(u'matplotlib inline')
    #Visualize the values in Train data
    # library
    import matplotlib.pyplot as plt
    def visualize_data(in_Sentiment_train,in_file_name):
        pos_tweets = list(in_Sentiment_train).count('positive')
        neu_tweets = list(in_Sentiment_train).count('neutral')
        neg_tweets = list(in_Sentiment_train).count('negative')

        num=len(in_Sentiment_train)
        print num
        posative_per = (pos_tweets)*100/num
        neutal_per   = (neu_tweets)*100/num
        negative_per = (neg_tweets)*100/num

        import matplotlib.pyplot as plt
        names = 'Posative','Neutal','Negative'
        size = [posative_per,neutal_per,negative_per]
        fig = plt.figure()
        fig.patch.set_facecolor('#F8F8FF')
        plt.rcParams['text.color'] = 'Black'
        my_circle=plt.Circle( (0,0), 0.7, color='#F8F8FF')

        plt.pie(size, labels=names,colors = ['#21683d', '#784f8e','#d35e60'])
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.savefig(in_file_name, dpi=300)
        plt.show()


# In[10]:


if IN_JUPYTER :
    #visualize the data
    visualize_data(Sentiment_test,'sentiment_train_image')


# In[11]:


if IN_JUPYTER :
    visualize_data(Sentiment_train,'sentiment_test_image')


# #### classifier

# In[8]:


if IN_JUPYTER :
  df_class = pd.DataFrame()


# In[10]:


if IN_JUPYTER :
    in_classifier = MultinomialNB(alpha=0.01,fit_prior=False)
    def_Classifier= Def_Classifier(BagofWord_Train,Sentiment_train,BagofWord_Test,Sentiment_test,in_classifier)
    precision, recall, accuracy, f1 =def_Classifier.class_pred()
    df_class = df_class.append({'Classifier':'MultinomialNB','precision':precision,'recall':recall,'accuracy':accuracy,'f1':f1 }, ignore_index=True)
    print "=================== Results ==================="
    print  "classifier:"
    print   in_classifier
    print  "======================================"
    print  "Precision:"+ str(precision)
    print   
    print  "recall:"+ str(recall)
    print   
    print  "accuracy:"+ str(accuracy)
    print   
    print  "f1:"+ str(f1)
    print   


# #### XGBOOST

# In[14]:


if IN_JUPYTER :
    from xgboost import XGBClassifier as XGBoostClassifier
    in_classifier = XGBoostClassifier(n_estimators=440,max_depth=3,objective="reg:logistic",learning_rate=0.30,gamma=0)
    def_Classifier= Def_Classifier(BagofWord_Train,Sentiment_train,BagofWord_Test,Sentiment_test,in_classifier)
    precision, recall, accuracy, f1 =def_Classifier.class_pred()
    df_class = df_class.append({'Classifier':'XGBoostClassifier', 'precision':precision,'recall':recall,'accuracy':accuracy,'f1':f1}, ignore_index=True)
    #precision, recall, accuracy, f1 =def_classifier(X_train, X_test, y_train, y_test,in_classifier)

    print "=================== Results  ==================="
    print  "classifier:"
    print   in_classifier
    print  "======================================"
    print  "Precision:"+ str(precision)
    print   
    print  "recall:"+ str(recall)
    print   
    print  "accuracy:"+ str(accuracy)
    print   
    print  "f1:"+ str(f1)


# #### SVM

# In[16]:


if IN_JUPYTER :
    #from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    in_classifier =SVC(kernel='rbf', C=1E6,gamma= 0.001)
    def_Classifier= Def_Classifier(BagofWord_Train,Sentiment_train,BagofWord_Test,Sentiment_test,in_classifier)
    precision, recall, accuracy, f1 =def_Classifier.class_pred()
    df_class = df_class.append({'Classifier':'SVC', 'precision':precision,'recall':recall,'accuracy':accuracy,'f1':f1}, ignore_index=True)
    #precision, recall, accuracy, f1 =def_classifier(X_train, X_test, y_train, y_test,in_classifier)

    print "=================== Results ==================="
    print  "classifier:"
    print   in_classifier
    print  "======================================"
    print  "Precision:"+ str(precision)
    print   
    print  "recall:"+ str(recall)
    print   
    print  "accuracy:"+ str(accuracy)
    print   
    print  "f1:"+ str(f1)
    print   


# In[72]:


if IN_JUPYTER :
 #print df_class.head(15)
 df_recall=[item[0] for item in df_class["recall"]]
 print df_recall
 df_precision=[item[0] for item in df_class["precision"]]
 print df_precision


# In[76]:


#Visualize the model and results
if IN_JUPYTER :
    import plotly.offline as py
    import plotly.graph_objs as go

    trace0 = go.Bar(
        x=df_class["Classifier"],
        y=df_class["accuracy"],
        name='Accuracy',
        marker=dict(
            color='rgb(115, 105, 128 )',
        ),
    )
    data = [trace0]
    layout = go.Layout(
         title='Metrics',
         barmode='group',
         bargap=0.5,
         bargroupgap=0.1,
         width=700,
         height=700
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='accuracy_exp1.html')
    
    trace1 = go.Bar(
         x=df_class["Classifier"],
         y=df_class["f1"],
        name='f1 Score',
        marker=dict(
            color='rgb(51, 194, 204 )',
        )
    )
    data = [trace1]
    layout = go.Layout(
         title='Metrics',
         barmode='group',
         bargap=0.5,
         bargroupgap=0.1,
         width=700,
         height=700
    )
    
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='f1_exp1.html')
    
    trace2 = go.Bar(
         x=df_class["Classifier"],
         y=df_precision,
        name='Precision',
        marker=dict(
            color='rgb(0, 212, 139)',
        )
    )
    data = [trace2]
    layout = go.Layout(
         title='Metrics',
         barmode='group',
         bargap=0.5,
         bargroupgap=0.1,
         width=700,
         height=700
    )
        
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='Precision_exp1.html')
    

    trace3 = go.Bar(
        x=df_class["Classifier"],
        y=df_recall,
        name='Recall',
        marker=dict(
            color='rgb(162, 111, 219 )',
        ),
    )
    data = [trace3]
    layout = go.Layout(
         title='Metrics',
         barmode='group',
         bargap=0.5,
         bargroupgap=0.1,
         width=700,
         height=700
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='recall_exp1.html')

    



# ## Grid search

# In[120]:


if IN_JUPYTER :
    #creating pipeline
    from sklearn.cross_validation import  cross_val_score
    from sklearn.pipeline import Pipeline

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
     ])

    scores = cross_val_score(text_clf,  # steps to convert raw messages into models
                             data_train ,  # training data
                             Sentiment_train,  # training labels
                             cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                             scoring='accuracy',  # which scoring metric?
                             n_jobs=1,  # -1 = use all cores = faster
                             )
    print scores



# In[123]:


if IN_JUPYTER :
    from sklearn.model_selection import GridSearchCV
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                 }
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1,cv= 10 ,scoring='accuracy')
    gs_clf = gs_clf.fit(data_train, Sentiment_train)


# In[124]:


if IN_JUPYTER :
   print gs_clf.best_score_
   print  gs_clf.best_params_


# ### Glove

# In[45]:


if IN_JUPYTER :
    #Calling functions to process train data
    tweet_preprocess=TwitterData_Preproces('./sanders-twitter-0.2/file.csv')
    data_train ,Sentiment_train = tweet_preprocess.Csv_process()


# In[46]:


if IN_JUPYTER :
    #splitting the data
    x_train, x_test, y_train, y_test = train_test_split(data_train, Sentiment_train, test_size=0.2, random_state=42)


# In[3]:


class TwitterData(object):    

    def get_w2v_vec(self,tweet, size, vectors):
        #create the word embedding for the model
        featureVec = np.zeros(size,dtype="float32").reshape((1, size))
        count = 0
        for word in tweet.split():
            try:
                featureVec = np.add(featureVec,vectors[word])
                count += 1
            except KeyError:
                continue
        if count != 0:
                np.divide(featureVec,count)
        return featureVec


# In[4]:


#from sklearn.preprocessing import scale
class TwitterData_Vector(TwitterData):    
    
    def __init__(self,Twitter_data):
        self.Twitter_data=Twitter_data  
     
    def glove_vec(self):
        print "Reached in glove_vec "
        num_features=200
        # Preallocate a 2D numpy array, for speed
        #glove_twitter = api.load("glove-twitter-200")
        reviewFeatureVecs = np.zeros((len(self.Twitter_data),num_features),dtype="float32")
        glove_twitter = word2vec.KeyedVectors.load_word2vec_format(join(GLOVE_DIR, 'glove-twitter-200.txt'), binary=False)
        # Initialize a counter
        counter = 0
        for tweet in self.Twitter_data :   
            reviewFeatureVecs[counter] = super(TwitterData_Vector, self).get_w2v_vec(tweet, num_features, glove_twitter)           
            # Increment the counter
            counter = counter + 1

        return reviewFeatureVecs


# In[49]:


if IN_JUPYTER :
    print "Creating average feature vecs for train reviews" 
    twitterData_vector=TwitterData_Vector(x_train)
    trainDataVecs=twitterData_vector.glove_vec()


# In[50]:


if IN_JUPYTER :
    print "Creating average feature vecs for test reviews" 
    twitterData_vector=TwitterData_Vector(x_test)
    testDataVecs=twitterData_vector.glove_vec()


# #### SVM

# In[51]:


if IN_JUPYTER :
  df_class = pd.DataFrame()


# In[55]:


if IN_JUPYTER :
    #from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    in_classifier =SVC( kernel='rbf', C=1E12,gamma=.001)
    save=False
    def_Classifier= Def_Classifier(trainDataVecs,y_train,testDataVecs,y_test,in_classifier,save)
    precision, recall, accuracy, f1 =def_Classifier.class_pred()
    df_class = df_class.append({'Classifier':'SVC', 'precision':precision,'recall':recall,'accuracy':accuracy,'f1':f1}, ignore_index=True)
    #precision, recall, accuracy, f1 =def_classifier(X_train, X_test, y_train, y_test,in_classifier)

    print "=================== Results ==================="
    print  "classifier:"
    print   in_classifier
    print  "======================================"
    print  "Precision:"+ str(precision)
    print   
    print  "recall:"+ str(recall)
    print   
    print  "accuracy:"+ str(accuracy)
    print   
    print  "f1:"+ str(f1)
    print   


# #### XGBOOST

# In[56]:


if IN_JUPYTER :    
    save=True
    in_classifier = XGBoostClassifier(n_estimators=600,max_depth=3,objective="reg:logistic",learning_rate= 0.3,gamma=0.0010)
    def_Classifier= Def_Classifier(trainDataVecs,y_train,testDataVecs,y_test,in_classifier,save)
    precision, recall, accuracy, f1 =def_Classifier.class_pred()
    df_class = df_class.append({'Classifier':'xgboost', 'precision':precision,'recall':recall,'accuracy':accuracy,'f1':f1}, ignore_index=True)
    #precision, recall, accuracy, f1 =def_classifier(X_train, X_test, y_train, y_test,in_classifier)

    print "=================== Results ==================="
    print  "classifier:"
    print   in_classifier
    print  "======================================"
    print  "Precision:"+ str(precision)
    print   
    print  "recall:"+ str(recall)
    print   
    print  "accuracy:"+ str(accuracy)
    print   
    print  "f1:"+ str(f1)
    print   


# In[57]:


#Visualize the model and results
if IN_JUPYTER :
     #print df_class.head(15)
    df_recall=[item[0] for item in df_class["recall"]]
    df_precision=[item[0] for item in df_class["precision"]]

    import plotly.offline as py
    import plotly.graph_objs as go

    trace0 = go.Bar(
        x=df_class["Classifier"],
        y=df_class["accuracy"],
        name='Accuracy',
        marker=dict(
            color='rgb(115, 105, 128 )',
        ),
    )
    data = [trace0]
    layout = go.Layout(
         title='Metrics',
         barmode='group',
         bargap=0.5,
         bargroupgap=0.1,
         width=500,
         height=500
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='glove_accuracy_exp1.html')
    
    trace1 = go.Bar(
         x=df_class["Classifier"],
         y=df_class["f1"],
        name='f1 Score',
        marker=dict(
            color='rgb(51, 194, 204 )',
        )
    )
    data = [trace1]
    layout = go.Layout(
         title='Metrics',
         barmode='group',
         bargap=0.5,
         bargroupgap=0.1,
         width=500,
         height=500
    )
    
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='glove_f1_exp1.html')
    
    trace2 = go.Bar(
         x=df_class["Classifier"],
         y=df_precision,
        name='Precision',
        marker=dict(
            color='rgb(0, 212, 139)',
        )
    )
    data = [trace2]
    layout = go.Layout(
         title='Metrics',
         barmode='group',
         bargap=0.5,
         bargroupgap=0.1,
         width=500,
         height=500
    )
        
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='glove_Precision_exp1.html')
    

    trace3 = go.Bar(
        x=df_class["Classifier"],
        y=df_recall,
        name='Recall',
        marker=dict(
            color='rgb(162, 111, 219 )',
        ),
    )
    data = [trace3]
    layout = go.Layout(
         title='Metrics',
         barmode='group',
         bargap=0.5,
         bargroupgap=0.1,
         width=500,
         height=500
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='glove_recall_exp1.html')

    


# ### Twitter 

# In[15]:


import time 
class TwitterSearch(object) :
    
    if not IN_JUPYTER:
        import json
    
    def __init__(self,in_query):
        self.in_query=in_query
    def Twitter_search(self):
            with open(config_file) as f:
                #config = json.loads(f.read())
                config = json.load(f)
                consumer_key = config["consumer_key"]
                consumer_secret = config["consumer_secret"]
                access_token = config["access_token"]
                access_secret = config["access_secret"]
                
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_secret)

            api = tweepy.API(auth)

            if (not api):
                print ("Problem connecting to API")    
            #Switching to application authentication
            auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

            #Setting up new api wrapper, using authentication only
            api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

            #Error handling
            if (not api):
                print ("Problem Connecting to API")
            searchQuery = self.in_query
            maxTweets = 10000
            tweetCount = 0
            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name =fpath(__name__, searchQuery.replace("#", "") +'_' +timestr + '.json')

            #Open a text file to save the tweets to
            with open(file_name, 'w') as f:

                #Tell the Cursor method that we want to use the Search API (api.search)
                #Also tell Cursor our query, and the maximum number of tweets to return
                for tweet in tweepy.Cursor(api.search,q=searchQuery, lang="en").items(maxTweets) :         

                    #Verify the tweet has place info before writing (It should, if it got past our place filter)
                    if tweet.place is not None:

                        #Write the JSON format to the text file, and add one to the number of tweets we've collected
                        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
                        tweetCount += 1

                #Display how many tweets we have collected
                print("Downloaded {0} tweets".format(tweetCount))
            if tweetCount >0 :    
                #Preprocessing the tweets
                #Calling functions to process train data
                tweet_preprocess=TwitterData_Preproces(file_name)
                tweets_pred = tweet_preprocess.json_process() 
                #Create the vectors
                twitterData_vector=TwitterData_Vector(tweets_pred)
                PredDataVecs=twitterData_vector.glove_vec()
                #Predictions
                f = open(filename, 'r')
                clf = pickle.load(f) 
                predictions = clf.predict(PredDataVecs)
                output=pd.DataFrame( {"sentiment":predictions })
                #delete the file that ssaved 
                os.remove(file_name) 
                #pass the final output
                return  output['sentiment'].value_counts()
            else :
                output=pd.DataFrame({'sentiment':'xxx'}, index=[0])
                return   output['sentiment'].value_counts()


# In[16]:


if IN_JUPYTER :
    tweet_search=TwitterSearch('#WorldCup')
    prediction_out=tweet_search.Twitter_search()


# In[17]:


if IN_JUPYTER :
    print prediction_out

