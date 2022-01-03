# Reference of the app. The structure of this app is based on code from this git repo:-
# https://github.com/Coding-with-Adam/Dash-by-Plotly/tree/master/Deploy_App_to_Web/Multipage_App
#citation for Side bar : https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from apps import page2, page3, predictionforindia, Home, US_data_viz

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    html.Div(id='page-content', children=[]),
],style={'backgroundColor':'#FFFFFF',"width":"100%","margin-right":"100%","height":"100%"})


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/page2':
        return page2.layout
    if pathname == '/apps/predictionforindia':
        return predictionforindia.layout
    if pathname ==  '/apps/page3':
        return page3.layout
    if pathname=="/apps/Home":
        return Home.layout
    if pathname=="/apps/US-Data":
        return US_data_viz.layout
    if pathname=="/apps/US-Prediction":
        return page3.layout
    else:
        return Home.layout


if __name__ == '__main__':
    app.run_server(debug=True)
