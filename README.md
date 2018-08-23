# Project : Sentiment Analysis on Tweets

Microblogging has become a very popular communication tool in today's world. The huge number of forums and availability of data makes it a perfect way to analyze the reaction of the end users as it happens.In this project the attempt is made to analyze the sentiments expressed as Tweets.

### Installing

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [jsonpickle](https://jsonpickle.github.io/)
- [tweepy](http://www.tweepy.org/)
- [nltk](https://www.nltk.org/)
- [plotly](https://plot.ly/)
- [gensim](https://radimrehurek.com/gensim/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)
In order to execute the twitter download and prediction you will need to create a Twitter API account.Please find the insturctions here (http://docs.inboundnow.com/guide/create-twitter-application/).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

## Running the tests

The code is provided in the `Capstone.ipynb` notebook file. You will also be required to use the `file.csv` dataset file to complete your work. The flask based application can be executed by running 'run.py' file.This file will require Capstone.py file to be present.The run.py file needs to be outside the folder Capstone.The application will also require  "__init__.py" file .The templates folder is needed for the flask based application.The config folder contains the twitter connection deatils.This needs to be created and added for the application to work.


### Run

In a terminal or command window, navigate to the top-level project directory `Twitter_Analysis/Capstone` (that contains this README) and run one of the following commands:

```bash
ipython notebook Capstone.ipynb
```  
or
```bash
jupyter notebook Capstone.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

