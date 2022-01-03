#citation for Side bar : https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
#reference----https://www.kaggle.com/mauromauro/learning-in-cyberspace-a-story-of-pandemic-times
import pathlib
import pandas as pd
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from app import app

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
df=pd.read_csv(str(DATA_PATH)+'/USdatasetforPrediction.csv')


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



code_dict={"Illinois":"IL","Indiana":"IN","Michigan":"MI","Missouri":"MO","New Jersey":"NJ","New York":"NY","Texas":"TX","Utah":"UT","Virginia":"VA","Washington":"WA"}


df["abbrev"]=df["state"].apply(lambda x:code_dict[x])

df_sample=df.sample(10000)
df_map=df_sample.groupby(["abbrev"],as_index=False).mean()




fig_chl=px.choropleth(locations=df_map["abbrev"], locationmode="USA-states", color=df_map["engagement_index"], scope="usa")





df_groupbydate=df.groupby(["state","time"],as_index=False).mean()
df=df.sort_values(by="engagement_index")
df_top_products=df.tail(10)
print(df_top_products["Product Name"],df_top_products["engagement_index"])
layout = dbc.Container([html.Div([
    html.H4("Prediction results"),
    html.H3(id="prediction-result1"),
    #html.H2("Below are the recommendations which may help you in improving your Online Studying experience"),

]),
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
dcc.Graph(id="usMap", figure=fig_chl),
# html.H6(id="graph1")
dcc.Graph(id="graph1"),
dcc.Graph(id="top-pie-chart",figure=fig_chl),
dcc.Graph(id="chart_by_time",figure=fig_chl),
dcc.Graph(id="chart_by_area",figure=fig_chl),
dcc.Graph(id="pct_vs_engagement_index",figure=fig_chl),
html.Br(),
dcc.Graph(id="engagement_vs_primary_essential",figure=fig_chl)
    ])
@app.callback(
    Output("engagement_vs_primary_essential", "figure"),
    [Input("usMap", "selectedData")])
def pict_of_connectivity(selected):
    state_name = ""
    df_sample_dated = df_sample.copy()
    if selected is None:
        pass
    else:

        data_dict = dict(selected)
        # temp_dict=dict(data_dict.get("points"))
        state_code = str(data_dict.get("points")[0]["location"])
        for i, j in code_dict.items():
            if state_code == j:
                state_name = (i)
        df_sample_dated = df_sample_dated[df_sample_dated['state'] == state_name]
    df_sam2=df_sample_dated[['Primary Essential Function','Provider/Company Name','Product Name','engagement_index']]
    df_sam2.dropna(inplace=True)
    symbol = []

    for funct in df_sam2['Primary Essential Function']:
        f = funct.split(' - ')
        symbol.append(f[0])
    df_sam2['Symbol']=symbol
    grouped = df_sam2.groupby(['Primary Essential Function', 'Provider/Company Name', 'Product Name', 'Symbol'], as_index=False).mean()

    figure1 = px.treemap(grouped, path=['Symbol', 'Primary Essential Function', 'Provider/Company Name'],values='engagement_index', color='Symbol',color_discrete_map={'LC': 'blue', 'SDO': 'cyan', 'CM': 'lightcyan', 'LC/CM/SDO': 'white'})

    figure1.update_layout(
        title={
            'text': "TreeMap of products by their essential function",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        template='plotly_dark'
    )

    grouped_sorted = grouped.sort_values('engagement_index')[-15:]
    figure2 = px.bar(grouped_sorted, x='Product Name', y='engagement_index', color='engagement_index')
    new_figure = make_subplots(rows=1, cols=2, specs=[[{"type": "sunburst"}, {"type": "bar"}]])

    new_figure.add_trace(figure1.data[0], row=1, col=1)
    new_figure.add_trace(figure2.data[0], row=1, col=2)

    new_figure.update_layout(
        title={'text': "Engagement by Primary Essential Function | Engagement by Product - 2020",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        template="presentation",
        xaxis_tickangle=45
    )

    new_figure.layout.coloraxis.colorscale = [(0.0, 'cyan'), (0.5, 'blue'), (1.0, 'darkblue')]
    return new_figure

@app.callback(
    Output("pct_vs_engagement_index", "figure"),
    [Input("usMap", "selectedData")])
def pict_of_connectivity(selected):
    state_name = ""
    df_sample_dated = df_sample.copy()
    if selected is None:
        pass
    else:

        data_dict = dict(selected)
        # temp_dict=dict(data_dict.get("points"))
        state_code = str(data_dict.get("points")[0]["location"])
        for i, j in code_dict.items():
            if state_code == j:
                state_name = (i)
        df_sample_dated = df_sample_dated[df_sample_dated['state']==state_name]
    df_sam=df_sample_dated[['time','pct_access','engagement_index']].groupby('time', as_index=False).mean()
    from sklearn.preprocessing import StandardScaler
    std_scaler = StandardScaler()
    df_sam[['engagement_index','pct_access']]=std_scaler.fit_transform(df_sam[['engagement_index','pct_access']])
    figure = px.line(df_sam, x='time', y=['pct_access', 'engagement_index'], height=500, title='Engagement Index and pct_access', color_discrete_map={ "pct_access": "Red", "engagement_index": "cyan"},template="presentation")

    figure.layout.legend.x = 0.75
    figure.layout.legend.y = 1.15
    figure.layout.legend.title.text = ''

    return figure

@app.callback(
    Output("graph1", "figure"),
    [Input("usMap", "selectedData")])
def lassso_graph1(selected):


    state_name = ""
    df_sample_dated = df_sample.copy()
    df_sample_dated["month"] = pd.DatetimeIndex(df_sample["time"]).month
    month_dct = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
                 9: "September", 10: "October", 11: "November", 12: "December"}

    df_sample_top_ten1 = df_sample_dated.groupby(["month"], as_index=False).mean()
    df_sample_top_ten1["month_name"] = df_sample_top_ten1["month"].apply(lambda y: month_dct[y])
    df_sample_top_ten = df_sample_dated.groupby(["state", "month"], as_index=False).mean()

    if selected is None:
        df_sample_temp = df_sample.head(10)
        return px.treemap(df_sample, path=['locale', 'Product Name'],
                   values='engagement_index', color='locale')
    else:
        data_dict = dict(selected)
        # temp_dict=dict(data_dict.get("points"))
        state_code = str(data_dict.get("points")[0]["location"])
        for i, j in code_dict.items():
            if state_code == j:
                state_name = (i)
        y=df_sample[df_sample['state'] == state_name]
        #district_dataset.loc[:, 'district'] = district_dataset.district.apply(lambda x: str(x))
        f=px.treemap(y, path=['locale', 'Product Name'],
                   values='engagement_index', color='locale')
        df_sample_pct = df_sample_dated[df_sample_dated["state"] == state_name].sort_values(by="month")
        x = df_sample_top_ten[df_sample_top_ten["state"] == state_name]
        month_dct = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
                     9: "September", 10: "October", 11: "November", 12: "December"}
        x["month_name"] = x["month"].apply(lambda y: month_dct[y])

        fig4 = px.area(x.sort_values(by="month"), x="month_name", y="engagement_index")
        # fig3.add_bar(x=x["month_name"],y=x["pct_access"])
        return f

@app.callback(
    Output("top-pie-chart", "figure"),
    [Input("usMap", "selectedData")])
def lassso_pie_for_top_products(selected):
    state_name =""
    df_sample_top_ten=df.groupby(["state","Product Name"],as_index=False).size()

    if selected is None:
        df_sample_temp=df_sample.head(10)
        return px.pie(df_sample_temp,names="Product Name",values=[2,2,3,4,4,5,6,7,8,9])
    else:
        data_dict=dict(selected)
        # temp_dict=dict(data_dict.get("points"))
        state_code= str(data_dict.get("points")[0]["location"])
        for i,j in code_dict.items():
            if state_code==j:
                state_name= (i)

        df_sample_top_ten=df_sample_top_ten[df_sample_top_ten["state"]==state_name].sort_values(by="size")

        df_for_pie=df_sample_top_ten.tail(10)
        return px.pie(df_for_pie,names="Product Name",values="size")

@app.callback(
    Output("chart_by_time", "figure"),
    [Input("usMap", "selectedData")])
def chart_by_time(selected):
    state_name = ""
    df_sample_dated=df_sample.copy()
    df_sample_dated["month"]=pd.DatetimeIndex(df_sample["time"]).month
    month_dct = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
                 9: "September", 10: "October", 11: "November", 12: "December"}

    df_sample_top_ten1 = df_sample_dated.groupby(["month"],as_index=False).mean()
    df_sample_top_ten1["month_name"] = df_sample_top_ten1["month"].apply(lambda y: month_dct[y])
    df_sample_top_ten = df_sample_dated.groupby(["state", "month"], as_index=False).mean()

    if selected is None:

        fig_t=px.line(df_sample_top_ten1,x="month_name",y="pct_access")


        fig_t.add_bar(x=df_sample_top_ten1["month_name"], y=df_sample_top_ten1["pct_access"])
        return fig_t
    else:
        data_dict = dict(selected)
        state_code = str(data_dict.get("points")[0]["location"])
        for i, j in code_dict.items():
            if state_code == j:
                state_name = (i)

        df_sample_pct = df_sample_dated[df_sample_dated["state"] == state_name].sort_values(by="month")
        x=df_sample_top_ten[df_sample_top_ten["state"]==state_name]
        month_dct={1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        x["month_name"]=x["month"].apply(lambda y:month_dct[y])

        fig3=px.line(x.sort_values(by="month"),x="month_name",y="pct_access")
        fig3.add_bar(x=x["month_name"],y=x["pct_access"])
        return fig3



@app.callback(
    Output("chart_by_area", "figure"),
    [Input("usMap", "selectedData")])
def chart_by_time(selected):
    state_name = ""
    df_sample_dated = df_sample.copy()
    df_sample_dated["month"] = pd.DatetimeIndex(df_sample["time"]).month
    month_dct = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
                 9: "September", 10: "October", 11: "November", 12: "December"}

    df_sample_top_ten1 = df_sample_dated.groupby(["month"], as_index=False).mean()
    df_sample_top_ten1["month_name"] = df_sample_top_ten1["month"].apply(lambda y: month_dct[y])
    df_sample_top_ten = df_sample_dated.groupby(["state", "month"], as_index=False).mean()



    if selected is None:

        return px.area(df_sample_top_ten1.sort_values(by="month"),x="month_name",y="engagement_index")
    else:
        data_dict = dict(selected)

        state_code = str(data_dict.get("points")[0]["location"])
        for i, j in code_dict.items():
            if state_code == j:
                state_name = (i)

        df_sample_pct = df_sample_dated[df_sample_dated["state"] == state_name].sort_values(by="month")
        x=df_sample_top_ten[df_sample_top_ten["state"]==state_name]
        month_dct={1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        x["month_name"]=x["month"].apply(lambda y:month_dct[y])

        fig4=px.area(x.sort_values(by="month"),x="month_name",y="engagement_index")

        return fig4








