import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from streamlit_apex_charts import line_chart, bar_chart, pie_chart, area_chart, radar_chart
from streamlit.elements.image import image_to_url
from streamlit_lottie import st_lottie
import json

DATA_URL = (
    "Tweets.csv"
)

st.title("关于美国航空公司的推文的情绪分析")
st.sidebar.title("推文的情感分析")
st.markdown("这是一个分析关于美国航空推文的情感的Streamlit interactive dashboard 🐦")
st.sidebar.markdown("这是一个分析关于美国航空推文的情感的Streamlit interactive dashboard🐦")
audio_file = open('C:\大三上可视化\Create-Interactive-Dashboards-with-Streamlit-and-Python-master\moumou.ogg', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')
#加载背景图
import time
# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.01)

from PIL import Image
st.sidebar.subheader("背景动画")
if not st.sidebar.checkbox("Close", True,key='81'):
#加载背景图
      #image = Image.open('C:/大三上可视化/Create-Interactive-Dashboards-with-Streamlit-and-Python-master/beach.jpg')
      #st.image(image, use_column_width=True)
      img_url = image_to_url('C:/大三上可视化/Create-Interactive-Dashboards-with-Streamlit-and-Python-master/beach.jpg',width=-3,clamp=False,channels='RGB',output_format='auto',image_id='')
      st.markdown('''
                 <style>
                 .css-fg4pbf {background-image: url(''' + img_url + ''');}
                 </style>''', unsafe_allow_html=True)

#加载动画
      with open("C:/大三上可视化/Create-Interactive-Dashboards-with-Streamlit-and-Python-master/ti1.json", "r",errors='ignore') as f:
            data1 = json.load(f)
            st_lottie(data1, key="15")

      c11, c22, c33 = st.columns(3)
      with c11:
            with open("C:/大三上可视化/Create-Interactive-Dashboards-with-Streamlit-and-Python-master/TREE.json", "r",errors='ignore') as f:
                 data2 = json.load(f)
                 st_lottie(data2, key="16")
      with c22:
            with open("C:/大三上可视化/Create-Interactive-Dashboards-with-Streamlit-and-Python-master/PEOPLE.json", "r",errors='ignore') as f:
                data3 = json.load(f)
                st_lottie(data3, key="17")
      with c33:
              with open("C:/大三上可视化/Create-Interactive-Dashboards-with-Streamlit-and-Python-master/PLANE.json", "r",errors='ignore') as f:
                  data4 = json.load(f)
                  st_lottie(data4, key="18")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.subheader("显示随机推文")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))#radio为单选按钮
st.sidebar.markdown(data[data.airline_sentiment=="positive"][["text"]].sample(n=1).iat[0, 0])

st.sidebar.markdown("按情绪划分的推文数")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("按情绪划分的推文数")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

st.sidebar.markdown("不同情绪的推文数按日期变化")
mi = st.sidebar.selectbox('ty', ['箱型图', '折线图','漏斗图'], key='28')
if not st.sidebar.checkbox("Hide", True,key='29'):
 if mi=='箱型图':
    c1=[3,838,736,751,835,1049,2266,1919,781]
    c2 = [1,297,335,329,383,278,463,676,337]
    c3=[0,273,273,296,282,230,350,433,226]
    trace1 = go.Box(y=c1,fillcolor="#ff7500",marker_color="#ff7500",name="negative")
    trace2 = go.Box(y=c2,fillcolor="#16a951",marker={'color':"#16a951"},name="neutral")
    trace2 = go.Box(y=c2,fillcolor="#16a951",marker={'color':"#46a941"},name="positive")
    data1 = [trace1,trace2]
    layout = go.Layout(plot_bgcolor='#ffffff',width=500,height=500,title='箱型图')
    fig = go.Figure(data=data1,layout=layout)
    st.plotly_chart(fig, layout=layout)

 elif mi=='折线图':
   #散点图
    c1=[3,838,736,751,835,1049,2266,1919,781]
    c2 = [1,297,335,329,383,278,463,676,337]
    c3=[0,273,273,296,282,230,350,433,226]
    trace11 = go.Scatter( x = [17,18,19,20,21,22,23,24],y=c1, mode = 'lines+markers',# 
        marker=dict(
        size=10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 1,
            color = ['blue']
        )
        
        
      ),name="negative")
    trace22 = go.Scatter( x = [17,18,19,20,21,22,23,24],y=c2, mode = 'lines+markers',# 
        marker=dict(
        size=10,
        color = 'rgba(0, 255, 255, .8)',
        line = dict(
            width = 1,
            color = ['green']
        )
        
        
      ),name="neutral")
    trace33 = go.Scatter( x = [17,18,19,20,21,22,23,24],y=c3, mode = 'lines+markers',# 
        marker=dict(
        size=10,
        color = 'rgba(127, 255, 0, .8)',
        line = dict(
            width = 1,
            color = ['green']
        )
        
        
      ),name="positive")
    data1 = [trace11,trace22,trace33]
    layout = go.Layout(width=500,height=500,title='散点图')
    fig = go.Figure(data=data1,layout=layout)
    st.plotly_chart(fig, layout=layout)
 else:
    mii = st.sidebar.selectbox('ty', ['positive', 'negative','neutral'], key='30')
    miii = st.sidebar.selectbox('ty', ['positive', 'negative','neutral'], key='31')
    if mii=='positive':
        x1=[0,273,273,296,282,230,350,433,226]
    elif mii=='negative':
        x1=[3,838,736,751,835,1049,2266,1919,781]
    else:
        x1=[1,297,335,329,383,278,463,676,337]
    if miii=='positive':
        x2=[0,273,273,296,282,230,350,433,226]
    elif miii=='negative':
        x2=[3,838,736,751,835,1049,2266,1919,781]
    else:
        x2=[1,297,335,329,383,278,463,676,337]
    #漏斗图
    trace0 = go.Funnel(
        y = ["2015/2/16","2015/2/17",
"2015/2/18",
"2015/2/19",
"2015/2/20",
"2015/2/21",
"2015/2/22",
"2015/2/23",
"2015/2/24"
],
        x=x1,
        textinfo = "value+percent initial",
        marker=dict(color=["deepskyblue"]*9),
        connector = {"line": {"color": "royalblue", "dash": "solid", "width": 3}})

    trace1 = go.Funnel(
        y = ["2015/2/16","2015/2/17",
"2015/2/18",
"2015/2/19",
"2015/2/20",
"2015/2/21",
"2015/2/22",
"2015/2/23",
"2015/2/24"
],
        x = x2,
        textinfo = "value+percent initial",
        marker=dict(color=["green"]*9),
        connector = {"line": {"color": "lightsalmon", "dash": "solid", "width": 3}})
    data = [trace0, trace1]
    layout = go.Layout(title='漏斗图')
    fig = go.Figure(data=data,layout=layout)
    st.plotly_chart(fig, layout=layout)

st.sidebar.markdown("按情绪划分的推文数其他表现")

file = st.sidebar.file_uploader("请上传csv表格", type=["csv"])

if file is not None and not st.sidebar.checkbox("Hide", True,key='11'):
    #random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))#radio为单选按钮
    st.markdown("抽选17号、19号、24号的数据集经过处理得结果")
    df1 = pd.read_csv(file, encoding="gbk")
    column = df1.columns  #获取表头
    df = pd.DataFrame(df1,columns=column)

    line_chart('Line chart',df)
    c1, c2 = st.columns(2)
    with c1:
        bar_chart('Bar chart',df)
        pie_chart('Pie chart',df)
    with c2:
        area_chart('Area chart',df)
        radar_chart('Radar chart',df)


st.sidebar.subheader("用户何时何地发推特?")
hour = st.sidebar.slider("Hour to look at", 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", True,key='7'):
    st.markdown("### Tweet locations based on time of day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)


st.sidebar.subheader("每家航空公司的推文总数")
each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Close", True,key='8'):
    if each_airline == 'Bar plot':
        st.subheader("每家航空公司的推文总数")
        fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_1)
    if each_airline == 'Pie chart':
        st.subheader("每家航空公司的推文总数")
        fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
        st.plotly_chart(fig_2)


@st.cache(persist=True)
def plot_sentiment(airline):
    df = data[data['airline']==airline]
    count = df['airline_sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
    return count


st.sidebar.subheader("按情绪细分航空公司")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'))
if len(choice) > 0:
    st.subheader("按情绪细分航空公司")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
    fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
    if breakdown_type == 'Bar plot':
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
                    row=i+1, col=j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
    else:
        fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
                    i+1, j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
st.sidebar.subheader("按情绪细分航空公司")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key=0)
if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='airline', y='airline_sentiment',
                         histfunc='count', color='airline_sentiment',
                         facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'},
                          height=600, width=800)
    st.plotly_chart(fig_0)

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close", True,key='9'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment']==word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

st.balloons()