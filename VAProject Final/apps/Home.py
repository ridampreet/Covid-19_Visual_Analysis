
#World Map Reference. The code of the world map is based on https://github.com/Coding-with-Adam/Dash-by-Plotly/blob/master/Other/Dash_Introduction/intro.py
#citation for Side bar : https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
#Reference for the line chart :https://plotly.com/python/line-charts/

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import pathlib
from app import app

# from apps import *
import dash_bootstrap_components as dbc
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

df_life_expectancy_updated=pd.read_csv(str(DATA_PATH)+'/Preprocessed_world_dataset.csv')

# ------------------------------------------------------------------------------
# App layout
'''Sidebar'''
year_dict={i: str(i) for i in range(2004, 2021, 1)}
layout = html.Div([
 html.Div([html.H2("Navigation", className="display-4"),html.Hr(),html.P("Please select the tabs to navigate to specific datasets and their Visualisations", className="lead"),dbc.Nav([html.Hr(),html.Button(dcc.Link('Home Page', href='/apps/Home'),style={"background-color": "#e7e7e7","width":"250px"}),
                html.Br(),
                html.Hr(),

                html.Button(dcc.Link('Visulisation for India\n', href='/apps/page2'),style={"background-color": "#e7e7e7","width":"250px"}),
                html.Br(),
                html.Hr(),


                html.Button(dcc.Link('Prediction of Experience', href='/apps/predictionforindia'),
                            style={"background-color": "#e7e7e7","width":"250px"}),
                html.Br(),
                html.Hr(),


                html.Button(dcc.Link('Prediction of US Data', href='/apps/US-Prediction'),
                            style={"background-color": "#e7e7e7", "width": "250px"}),
                html.Br(),
                html.Hr(),


                html.Button(dcc.Link('Visualisation for US Data ', href='/apps/US-Data'),
                            style={"background-color": "#e7e7e7","width":"250px"}),
                html.Br(),
                html.Hr(),


            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"position": "fixed","top": 0,"left": 0, "bottom": 0,"width": "25rem","padding": "2rem 1rem","background-color": "skyblue",},
),


    html.Div(
    html.H1("Online learning during COVID-19 and its impact on different countries", style={'text-align': 'center', "margin-top":'0%',"margin-bottom":"2%"})),
    html.Hr(),
html.Div([
    html.H6('Year to be selected'),
    dcc.Slider(id='year-slider',min=2004,max=2021,step=1,value=2005,marks=year_dict,)],style={'margin-left':'16em',"width":'80%',"margin-right":"20em"}),
    html.Br(),
    dcc.Graph(id='world_map', figure={},style={"width": "80%",'margin-left':'10em',"margin-right":"10em","margin-top":"0%","height":"50%"})

,
    html.Hr()
    ,html.Div([
dcc.Checklist(
        id="country-checklist",
        options=[{"label": x, "value": x}
                 for x in pd.unique(df_life_expectancy_updated['country'])],
        value=pd.unique(df_life_expectancy_updated['country']),
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="countries-value",style={"width": "100%",'margin-left':'0em',"margin-right":"25%"})
    ],style={'margin-left':'16em',"width":'80%',"margin-right":"20em"})
],style={'backgroundColor':'#FFFFFF',"width":"100%","margin-right":"100%","height":"100%","margin-left":"8%"},


)


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    Output("countries-value", "figure"),
    [Input("country-checklist", "value")])
def countries_line_graph(continents):
    mask = df_life_expectancy_updated.country.isin(continents)
    fig = px.line(df_life_expectancy_updated[mask], x="year", y="value", color='country')
    return fig

@app.callback(
     Output(component_id='world_map', component_property='figure'),
    [Input('year-slider', 'value')]
)
def Map(year_selected=2004):
    dataframe=df_life_expectancy_updated.copy()
    dataframe['value'] = dataframe['value'].apply(lambda x: float(x))
    fig = px.choropleth(
        data_frame=dataframe[(dataframe['year']==year_selected)],
        locationmode='country names',
        locations='country',
        scope="world",
        color='value',
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={"Online studies":"value"},
        template='ggplot2'
    )
    return fig


