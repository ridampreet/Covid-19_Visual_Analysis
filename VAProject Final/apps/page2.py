#citation for Side bar : https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
import ast
import base64
import json
import pathlib
from pathlib import Path
import pandas

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans

from app import app

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

df = pd.read_csv(str(DATA_PATH)+'/preprocessed_india_dataset.csv')
cols_time=["Time spent on Online Class","Time spent on self study","Time spent on fitness","Time spent on sleep","Time spent on social media","Time spent on TV"]

#Define in which column to look for missing values
df = df.dropna(subset=['Medium for online class'])

fig_hist=px.histogram(df,x="Age_group",nbins=10,color="Age_group",title="Bifercation of Students based on the Age groups")

fig_pie=px.pie(df, names="Medium for online class",title="Medium of online classes")

# reg = linear_model.LinearRegression()
pd.to_numeric(df['Numeric Rating'])
#Establish independent and dependent variables
df_TimeMgmt= df[['Time spent on Online Class', 'Time spent on self study', 'Time spent on fitness',
                    'Time spent on sleep','Time spent on social media','Numeric Rating']]
independent_variable = df_TimeMgmt.iloc[:, 0:5]
dependent_variable = df_TimeMgmt['Numeric Rating']
dependent_variable_decision=df['Rating of Online Class experience']
df_for_clustering=df[["Age of Subject",'Time spent on Online Class','Time spent on self study', 'Time spent on fitness','Time spent on sleep','Time spent on social media','Numeric Rating']]

#Convert string to integer
pd.to_numeric(df['Numeric Change in Weight'])

df_pi_num = df[['Number of meals per day', 'Numeric Change in Weight', 'Health Issue (1 or 0)','Time spent on Online Class', 'Time spent on self study', 'Time spent on fitness',
                    'Time spent on sleep','Time spent on social media','Numeric Rating']]

'''citation for sidebar:https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/'''

df_pi_num['Numeric Rating'] = df_pi_num['Numeric Rating'].astype(int)
mod2 = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

layout = dbc.Container([
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
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"position": "fixed","top": 0,"left": 0,"bottom": 0,"width": "25rem","padding": "2rem 1rem","background-color": "#80929D",},
),
    html.H1(children='Covid-19 learning impact in INDIA'),

    html.Hr(),


    html.H2(children='X-Axis'),
    dcc.Dropdown(id="xaxisB",
                 options=[{'label': i, 'value': i} for i in df.columns]),
    html.H2(children='Y-Axis'),
    dcc.Dropdown(id="yaxisB",value=3,
                 options=[{'label': i, 'value': i} for i in df.columns]),


    dbc.Row([
        dbc.Col(dcc.Graph(id='example-graph',style={'margin-left':'2%','width':'67em'},figure={"layout":{"height":600}})),
        # Not including fig here because it will be generated with the callback
    ]),
    html.Hr(),

    dbc.Row([
        dbc.Col(dcc.Graph(id='example-graph2', figure=fig_pie)),
        # Not including fig here because it will be generated with the callback
    ]),
    html.H2(children='X-Axis'),
    dcc.Dropdown(id="xaxis",
                 options=[{'label': i, 'value': i} for i in cols_time]),
    html.H2(children='Y-Axis'),
    dcc.Dropdown(id="yaxis",
                 options=[{'label': i, 'value': i} for i in cols_time]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='example-graph3')),
        dbc.Col(dcc.Graph(id="lasso1")),
        dbc.Col(dcc.Graph(id="lasso2"))
        # Not including fig here because it will be generated with the callback
    ])
    ,
    html.Hr(),
    html.H4(id="lasso")
    ,
    html.Hr()
    ,
    dcc.Checklist(
        id='features',
        options=[{'label': x, 'value': x}
                 for x in df_pi_num.columns],
        value=df_pi_num.columns.tolist(),
    ),
    dcc.Graph(id="heatmap"),
    html.H5('Number of clusters'),
    dcc.Slider(
        id='slider-for-cluster',
        min=2,
        max=7,
        step=1,
        value=2,
        marks={i: str(i) for i in range(2, 7, 1)},

    ),
    dcc.Graph(id='cluster'),
    html.H2(children='X axis'),
        dcc.Dropdown(id="x_axis",
                     options=[{'label': i, 'value': i} for i in df_for_clustering.columns],value='Time spent on Online Class'),
    html.H2(children='Y axis'),
        dcc.Dropdown(id="y_axis",
                     options=[{'label': i, 'value': i} for i in df_for_clustering.columns],value='Age of Subject')

])

@app.callback(
    Output('cluster', 'figure'),
    [Input('x_axis', 'value'), Input('y_axis', 'value'),Input('slider-for-cluster', 'value')])
def cluster(valx, valy,num_cluster):
    model = KMeans(n_clusters=int(num_cluster))

    df_for_clustering['Kmeans_cluster']=model.fit_predict(df_for_clustering)
    return px.scatter(df_for_clustering, x=valx, y=valy, color='Kmeans_cluster')



@app.callback(
    Output('example-graph3', 'figure'),
    Input('xaxis', 'value'), Input('yaxis', 'value'))
def pie_figure(valx, valy):
    df = pd.read_csv(str(DATA_PATH)+'/COVID-19 Survey Student Responses.csv')
    df = df.dropna(subset=["Rating of Online Class experience"])

    if valx == None or valy == None:
        valx = "Time spent on sleep"
        valy = "Time spent on fitness"

    return px.scatter(df, x=valx, y=valy, color="Rating of Online Class experience")

@app.callback(
    Output("heatmap", 'figure'),
    [Input("features", "value")])
def filter_heatmap(cols):
    fig = px.imshow(df_pi_num[cols].corr())
    return fig

@app.callback(
    [Output('lasso1',"figure"),
    Output('lasso2',"figure")],
    Input('example-graph3', 'selectedData'))
def filter_heatmap(selecData):
    if selecData==None:
        df_dummy=df.copy();
        df_dummy=df_dummy.head(5)
        df_dummy["num_Numeric Rating"]=df_dummy["Numeric Rating"].astype(int)
        fig=px.line(df_dummy.sort_values(by="Time spent on fitness"),x="Time spent on fitness",y="num_Numeric Rating",title="Cross Filtering Graphs")
        return fig,fig
    else:

        res=str(selecData)

        se=dict(selecData)


        l1=se.get("points")
        st=""
        x_collab=[]
        y_collab=[]

        for i in range(len(l1)):
            x_collab.append(l1[i]["x"])
            y_collab.append(l1[i]["y"])

        df["num"]=df["Numeric Rating"].astype(int)

        x_uniq=pd.unique(x_collab)
        y_uniq=pd.unique(y_collab)

        mn_rating_arr_x=[]
        mn_rating_arr_y = []
        for i in x_uniq:
            mn_rating_arr_x.append(df[(df["Time spent on sleep"] == i)]["num"].mean())

        for j in y_uniq:
            mn_rating_arr_y.append(df[(df["Time spent on fitness"] == j)]["num"].mean())

        dict_for_dataframe={"TimeX":x_uniq,"Rating":mn_rating_arr_x}
        df_lasso1=pd.DataFrame(data=dict_for_dataframe)
        dict_for_dataframe = {"TimeX": y_uniq, "Rating": mn_rating_arr_y}
        df_lasso2 = pd.DataFrame(data=dict_for_dataframe)
        #


        fig1=px.line(df_lasso1.sort_values(by="TimeX"),x="TimeX",y="Rating",title="Cross Filtering with Time spent on Sleep VS Rating of Expeirence")
        fig2 = px.line(df_lasso2.sort_values(by="TimeX"), x="TimeX", y="Rating",title="Cross Filtering with Time spent on fitness VS Rating of Expeirence")

        return fig1,fig2

@app.callback(
    Output('xaxisB', 'value'),  # for x axis
    Input('xaxisB', 'options'))
def set_values(available_options):
    return available_options[2]['value']

@app.callback(
    Output('yaxisB', 'value'),  # for x axis
    Input('yaxisB', 'options'))
def set_values(available_options):
    return available_options[3]['value']

@app.callback(
    Output('example-graph', 'figure'),
    Input('xaxisB', 'value'),Input('yaxisB', 'value'))
def bar_figure(valx,valy):
    df2 = df[:]

    return px.bar(df2, x=valx, y=valy)

@app.callback(
    Output('example-graph1', 'figure'),
    Input('xaxis', 'value'), Input('yaxis', 'value'))
def bar_figure(valx, valy):

    df["Age_group"]=df["Age of Subject"]

    df["Age_group"] = df["Age_group"].apply(lambda x: "less than 60" if type(x)==int and x <= 60 and x>50 else x)
    df["Age_group"] = df["Age_group"].apply(lambda x: "less than 50" if type(x)==int and x <= 50 and x > 40 else x)
    df["Age_group"] = df["Age_group"].apply(lambda x: "less than 40" if type(x)==int and x <= 40 and x > 30 else x)
    df["Age_group"] = df["Age_group"].apply(lambda x: "less than 30" if type(x)==int and x <= 30 and x > 20 else x)
    df["Age_group"] = df["Age_group"].apply(lambda x: "less than 20" if type(x)==int and x <= 20 and x > 10 else x)
    df["Age_group"] = df["Age_group"].apply(lambda x: "less than 10" if type(x)==int and x <= 10 else x)



    return px.pie(df,values="Age of Subject",names="Age_group")