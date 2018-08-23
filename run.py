import Capstone.Capstone
from Capstone import TwitterSearch
from flask import Flask, render_template,request,Markup
#for plotting
from plotly.offline import plot
import plotly.graph_objs as  go

app = Flask(__name__)

@app.route('/')
def index():
 return render_template('Index.html')

@app.route('/', methods=['GET', 'POST'])
def my_form_post():
   if request.method == 'POST':
     query = request.form['query']
     tweet_search=TwitterSearch(query)
     prediction_out=tweet_search.Twitter_search()
     data = [go.Scatter(x=prediction_out.index.tolist(), y=prediction_out.values.tolist())]        
     layout = go.Layout(
                         #title="<b>'Twitter Sentiment Analysis'</b>", 
                         title="<b>Twitter Sentiment Analysis for "+query +" </b>", 
                         xaxis=dict(title='Sentiment',
                                    titlefont=dict(
                                        color='#1f77b4'
                                    ),
                                    tickfont=dict(
                                        color='#1f77b4')
                                )  ,
                         yaxis=dict(title='Number of Tweets',
                                    titlefont=dict(
                                        color='#1f77b4'
                                    ),
                                    tickfont=dict(
                                        color='#1f77b4')
                                )  ,
                            paper_bgcolor='rgb(245,245,245)',
                            plot_bgcolor='rgb(245,245,245)'                    
                         )                     
                        
                        
     fig = go.Figure(data=data, layout=layout)
     my_plot_div = plot(fig, output_type='div')
     return render_template('Plot.html',div_placeholder=Markup(my_plot_div))
                                
   #return render_template('Index.html')

#def run_tweetsearch(in_query):
# tweet_search=TwitterSearch(in_query)
# prediction_out=tweet_search.Twitter_search()
# print prediction_out

if __name__ == '__main__':
    app.run(debug = False)
    #a = '#RedRising'
    #run_tweetsearch(str(a))



