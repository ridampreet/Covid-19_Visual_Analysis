#reference for gauge meter is https://plotly.com/python/gauge-charts/
#Reference https://www.kaggle.com/angywufeng/covid-19-s-impacts-regression-model-eda
#citation for Side bar : https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
import pathlib

import dash_bootstrap_components
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from app import app


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

df_india = pd.read_csv(str(DATA_PATH)+'/preprocessed_india_dataset.csv')
df,test=train_test_split(df_india,test_size=0.2)
cols_time=["Time spent on Online Class","Time spent on self study","Time spent on fitness","Time spent on sleep","Time spent on social media","Time spent on TV"]

#Define in which column to look for missing values
df = df.dropna(subset=['Medium for online class'])

fig_hist=px.histogram(df,x="Age_group",nbins=10,color="Age_group",title="Bifercation of Students based on the Age groups")
fig_pie=px.pie(df, names="Medium for online class",title="Medium of online classes")

reg = linear_model.LinearRegression()
pd.to_numeric(df['Numeric Rating'])
#Establish independent and dependent variables
Time= df[['Time spent on Online Class', 'Time spent on self study', 'Time spent on fitness','Time spent on sleep','Time spent on social media','Numeric Rating']]
independent_variable = Time.iloc[:, 0:5]
dependent_variable = Time['Numeric Rating']
reg.fit(independent_variable,dependent_variable)
y_pred=reg.predict(test[['Time spent on Online Class', 'Time spent on self study', 'Time spent on fitness',
                    'Time spent on sleep','Time spent on social media']])
y_test=test['Numeric Rating']

figure_decision=plt.figure(figsize=(25,20))
Accuracy_card=dbc.Card(
    [

        dbc.CardBody(

            [
                html.H4("The Mean Squared Error of this model is", className="sleep-title"),
                html.H4(
                   str(mean_squared_error(y_test,y_pred)),
                    className="card-text",
                ),

            ]
        ),
    ],
    style={"width": "18rem"},
    )

layout=dash_bootstrap_components.Container([
html.Div(
    [
        html.H2("Navigation", className="display-4"),
        html.Hr(),
        html.P(
            "Please select the tabs to navigate to specific datasets and their Visualisations", className="lead"
        ),
        dbc.Nav(
            [
                html.Hr(),
                # dbc.NavLink("Home", href='/apps/page2', active="exact"),
                html.Button(dcc.Link('Home Page', href='/apps/Home'),
                            style={"background-color": "#e7e7e7","width":"250px"}),
                html.Br(),
                html.Hr(),
                html.Button(dcc.Link('Visulisation for India\n', href='/apps/page2'),style={"background-color": "#e7e7e7","width":"250px"}),
                html.Br(),
                html.Hr(),
                html.Button(dcc.Link('Prediction of Experience', href='/apps/predictionforindia'),
                            style={"background-color": "#e7e7e7","width":"250px"}),
                html.Br(),
                html.Hr(),
 html.Button(dcc.Link('US Data visualisation', href='/apps/US-Data'),
                            style={"background-color": "#e7e7e7", "width":"250px"}),
                html.Br(),
html.Hr(),
html.Button(dcc.Link('US Data Prediction', href='/apps/US-Prediction'),
                            style={"background-color": "#e7e7e7", "width":"250px"}),
                html.Br(),
html.Hr(),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#94BBC6",
},
),

html.Div([
    html.H6('Time Spent on online Classes'),
    dcc.Slider(
        id='online-class',
        min=0,
        max=11,
        step=1,
        value=1,
        marks={i: str(i) for i in range(0, 11, 1)},
        vertical=True
    )],style={'width': '20%', 'display': 'inline-block'}
    ),
    html.Div([
        html.H6('Time Spent on self study'),
        dcc.Slider(
            id='self-study',
            min=0,
            max=11,
            step=1,
            value=1,
            marks={i: str(i) for i in range(0, 11, 1)},
            vertical=True
        )], style={'width': '20%', 'display': 'inline-block'}
    )
    ,
    html.Div([

    html.H6("Time Spent on Fitness"),
    dcc.Slider(
        id="fitness",
        min=0,
        max=5,
        step=1,
        value=1,
        marks={i: str(i) for i in range(0, 5, 1)},
        vertical=True

    )
    ],
        style={'width': '20%', 'display': 'inline-block'}
    ),
    html.Div([html.H6("Time Spent on Sleep"),
    dcc.Slider(
        id="sleep",
        min=4,
        max=15,
        step=1,
        value=1,
        marks={i: str(i) for i in range(4, 15, 1)},
        vertical=True
    )],style={'width': '20%', 'display': 'inline-block'})
    ,
    html.Div([html.H6("Time Spent on Social Media"),
    dcc.Slider(
        id="social-media",
        min=0,
        max=10,
        step=1,
        value=1,
        marks={i: str(i) for i in range(0, 10, 1)},
        vertical=True
    )],style={'width': '20%', 'display': 'inline-block'}),
dcc.Graph(id="exgraph"),
dbc.Row([dbc.Col(Accuracy_card,width=20)]),
html.Hr(),
        html.H3(id="prediction-result"),

])



@app.callback(

    Output("exgraph","figure"),  # for x axis


    [Input("online-class", "value"),Input("self-study", "value"),Input("fitness", "value"),Input("sleep", "value"),Input("social-media", "value")])
def predcition(time1,time5,time2,time3,time4):

    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reg.predict([[time1,time5,time2,time3,time4]])[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Rating of Experience"}))

    return fig3
