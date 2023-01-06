from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd

from simulation import run_simulation

dropdown_selections_datasets = ['iris']

app = Dash(__name__)

app.layout = html.Div([

    html.H1(children='Active Learning Dashboard'),

    html.Div([
        dcc.Dropdown(dropdown_selections_datasets, id="dropdown_dataset")
    ]),

    html.Div([
        dcc.Graph(id='output_graph')
    ])
])

@app.callback(
    Output('output_graph', 'figure'),
    [Input('dropdown_dataset', 'value')])
def update_output(dropdown_dataset):
    data = run_simulation(dropdown_dataset)

    fig = px.line(data, x='queries', y='accuracy', color='type')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)