#citation for Side bar : https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
import pathlib


import pandas as pd
import ast
import base64
import json
from pathlib import Path
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from app import app
import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from sklearn import linear_model
reg1 = linear_model.LinearRegression()

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

df_US=pd.read_csv(str(DATA_PATH)+'/USdatasetforPrediction.csv')
df,test=train_test_split(df_US,test_size=0.2)
dependent_variable=df['engagement_index']

independent_variable=df[['pp_total_raw','is_pandemic',"pct_black/hispanic","pct_free/reduced","holiday"]]
reg1.fit(independent_variable,dependent_variable)
y_pred=reg1.predict(test[['pp_total_raw','is_pandemic',"pct_black/hispanic","pct_free/reduced","holiday"]])
y_true=test['engagement_index']
print("started")
code_dict={"Illinois":"IL","Indiana":"IN","Michigan":"MI","Missouri":"MO","New Jersey":"NJ","New York":"NY","Texas":"TX","Utah":"UT","Virginia":"VA","Washington":"WA"}

df["abbrev"]=df["state"].apply(lambda x:code_dict[x])

df_sample=df.sample(10000)
df_map=df_sample.groupby(["abbrev"],as_index=False).mean()

mod = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#80929D",
}
mod2 = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


fig_chl=px.choropleth(locations=df_map["abbrev"], locationmode="USA-states", color=df_map["engagement_index"], scope="usa")

test1 =dbc.Card(
    [

        dbc.CardBody(

            [
                html.H4("The Mean Squared Error of this model is", className="sleep-title"),
                html.H4(
                   str(mean_squared_error(y_true,y_pred)),
                    className="card-text",
                ),

            ]
        ),
    ],
    style={"width": "18rem"},
    )


layout =dbc.Container([ html.Div([
    html.P("Medals included:"),
    dcc.Checklist(
        id='medals',
        options=[{'label': x, 'value': x}
                 for x in df.columns],
        value=df.columns,
    ),
    dcc.Graph(id="graph"),
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
                            style={"background-color": "#e7e7e7", "width":"250px"}),
                html.Br(),
                html.Hr(),
                html.Button(dcc.Link('Visulisation for India\n', href='/apps/page2'),style={"background-color": "#e7e7e7", "width":"250px"}),
                html.Br(),
html.Hr(),
                html.Button(dcc.Link('Prediction of Experience', href='/apps/predictionforindia'),
                            style={"background-color": "#e7e7e7", "width":"250px"}),
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
                # html.Br(),
                # dbc.NavLink("Prediction of Online Experience", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=mod,
),
html.Div([
    html.H6('pp_total_raw'),
    dcc.Slider(
        id='slider-time11',
        min=5000,
        max=50000,
        step=2000,
        value=5000,
        marks={i: str(i) for i in range(5000, 50000, 2000)},
        vertical=True
    )],style={'width': '20%', 'display': 'inline-block'}
    ),

html.Div([

        html.H6('pct_black/hispanic'),
        dcc.Slider(
            id='slider-timebh',
            min=0.1,
            max=0.9,
            step=0.1,
            value=0.1,
            marks={i: str(i) for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]},
            vertical=True
        )], style={'width': '20%', 'display': 'inline-block'}
    ),
html.Div([
        html.H6('pct_free/reduced'),
        dcc.Slider(
            id='slider-timefr',
            min=0.1,
            max=0.9,
            step=0.1,
            value=0.1,
            marks={i: str(i) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            vertical=True
        )], style={'width': '20%', 'display': 'inline-block'}
    ),
html.Div([
        html.H6('Holiday'),
        dcc.Slider(
            id='slider-timehol',
            min=0,
            max=1,
            step=1,
            value=0,
            marks={i: str(i) for i in range(0, 1, 1)},
            vertical=True
        )], style={'width': '20%', 'display': 'inline-block'}
    ),
    html.Div([

        html.H6("is_pandemic"),
        dcc.Slider(
            id="slider-time12",
            min=0,
            max=1,
            step=1,
            value=1,
            marks={i: str(i) for i in range(0, 1, 1)},
            vertical=True

        )
    ],
        style={'width': '20%', 'display': 'inline-block'}
    ),
dcc.Graph(id="exgraph1"),
dbc.Row([dbc.Col(test1,width=20)]),
html.Hr(),
        #html.H4("Prediction results"),
        html.H3(id="prediction-result1"),
        #html.H2("Below are the recommendations which may help you in improving your Online Studying experience"),


]),
])

@app.callback(
    Output("graph", "figure"),
    [Input("medals", "value")])
def filter_heatmap(cols):
    fig = px.imshow(df[cols].corr())
    return fig


@app.callback(
    Output("exgraph1","figure")  # for x axi
,
    [Input("slider-time11", "value"),Input("slider-time12", "value"),Input("slider-timebh","value"),Input("slider-timefr","value"),Input("slider-timehol","value")])
def predcition(time1,time2,timebh,timefr,timehol):

    fig4 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reg1.predict([[time1,time2,timebh,timefr,timehol]])[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Engagement Index"}))

    return fig4





