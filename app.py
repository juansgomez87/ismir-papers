#!/usr/bin/env python3
"""
WEIRD ISMIR Explorer visualization app on Dash

Copyright 2024, J.S. Gómez-Cañón, Erick Siavichay
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import numpy as np
import pandas as pd
from collections import Counter
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.offline as pyo
import ast


class Plotter:
    def __init__(self):
        self.data = pd.read_csv("data/ismir_all_papers.csv")

    def make_paper_info(self, idx):
        df_subset = self.data.iloc[idx]
        url = self.data.loc[idx, "Link"]
        authors = self.data.loc[idx, "Authors"]
        title = self.data.loc[idx, "Title"]
        abstract = self.data.loc[idx, "Abstract"]
        year = self.data.loc[idx, "Year"]
        countries = self.data.loc[idx, "first_country"].split(", ")

        fig_map = go.Figure()
        fig_map.add_trace(
            go.Choropleth(
                locations=list(Counter(countries).keys()),
                z=list(Counter(countries).values()),
                colorscale="Reds",
                reversescale=True,
                autocolorscale=False,
                colorbar_title="Counts",
            )
        )

        fig_map.update_geos(showcountries=True)
        fig_map.update_layout(
            height=300,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            geo=dict(
                showframe=False, showcoastlines=False, projection_type="equirectangular"
            ),
        )

        layout = html.Div(
            [
                html.H6("Article information:"),
                html.P(f"Title: {title}"),
                html.P(f"Authors: {authors}"),
                html.P(f"Year: {year}"),
                html.P(f"Abstract: {abstract}"),
                html.A(
                    html.Img(
                        src="assets/zenodo.png",
                        alt="zenodo_logo",
                        style={"width": "12%"},
                    ),
                    href=url,
                    target="_blank",
                ),
                html.Div(
                    children=[dcc.Graph(figure=fig_map)],
                    className="twelve columns",
                ),
            ],
            className="twelve columns",
        )

        return layout

    def create_layout(self, app):
        layout = html.Div(
            style={"background-color": "#ffffff"},
            children=[
                html.Div(
                    className="row header",
                    style={"background-color": "#f9f9f9", "margin": "5px 5px 5px 5px"},
                    children=[
                        html.H3(
                            "Not so WEIRD: A bibliometric analysis of ISMIR authorship",
                            style={"text-align": "right"},
                            className="nine columns",
                        ),
                    ],
                ),
                html.Section(
                    className="row",
                    style={"padding": "0px"},
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            children=[
                                                dcc.Graph(
                                                    id="graph-papers",
                                                    style={"height": "68vh"},
                                                ),
                                            ],
                                            className="nine columns",
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    children=[
                                                        "Select dimensionality reduction method:",
                                                        dcc.Dropdown(
                                                            style={
                                                                "margin": "0px 5px 5px 0px"
                                                            },
                                                            id="dropdown-method",
                                                            searchable=False,
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": "t-SNE",
                                                                    "value": "tsne",
                                                                },
                                                                {
                                                                    "label": "UMAP",
                                                                    "value": "umap",
                                                                },
                                                            ],
                                                            value="tsne",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        "Select plot dimensionality:",
                                                        dcc.Dropdown(
                                                            style={
                                                                "margin": "0px 5px 5px 0px"
                                                            },
                                                            id="dropdown-dim",
                                                            searchable=False,
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": "2D",
                                                                    "value": "2d",
                                                                },
                                                                {
                                                                    "label": "3D",
                                                                    "value": "3d",
                                                                },
                                                            ],
                                                            value="2d",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        "Select source of embeddings:",
                                                        dcc.Dropdown(
                                                            style={
                                                                "margin": "0px 5px 5px 0px"
                                                            },
                                                            id="dropdown-embeddings",
                                                            searchable=False,
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": "Paper titles",
                                                                    "value": "title",
                                                                },
                                                                {
                                                                    "label": "Paper abstracts",
                                                                    "value": "abstract",
                                                                },
                                                            ],
                                                            value="abstract",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        "Colorize using properties:",
                                                        dcc.Dropdown(
                                                            style={
                                                                "margin": "0px 5px 5px 0px"
                                                            },
                                                            id="dropdown-color",
                                                            searchable=False,
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": "Year",
                                                                    "value": "Year",
                                                                },
                                                                {
                                                                    "label": "Country",
                                                                    "value": "first_country",
                                                                },
                                                                {
                                                                    "label": "Affiliation Type",
                                                                    "value": "first_aff_cat",
                                                                },
                                                                {
                                                                    "label": "UN categories",
                                                                    "value": "first_aff_cat_UN",
                                                                },
                                                            ],
                                                            value="Year",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                            className="three columns",
                                        ),
                                    ],
                                ),
                            ],
                            className="six columns",
                        ),
                        html.Div(
                            [
                                html.Table(
                                    id="table-element", className="table__container"
                                )
                            ],
                            id="click-information",
                            className="six columns",
                        ),
                    ],
                ),
            ],
        )
        return layout

    def run_callbacks(self, app):
        @app.callback(
            [Output("graph-papers", "figure")],
            [
                Input("dropdown-dim", "value"),
                Input("dropdown-embeddings", "value"),
                Input("dropdown-color", "value"),
                Input("dropdown-method", "value"),
            ],
        )
        def display_plot(dim, emb_type, color, method):
            # Read the main dataframe if needed
            self.data = pd.read_csv("data/ismir_all_papers.csv")

            # Get column names for the coordinates
            col_prefix = f"{emb_type}_{method}_{dim}"
            self.data[col_prefix] = self.data[col_prefix].apply(ast.literal_eval)

            if dim == "2d":
                axes = dict(showgrid=True, zeroline=True, showticklabels=False)
                layout = go.Layout(
                    margin=dict(l=0, r=0, b=0, t=40),
                    legend=dict(x=0, y=0, orientation="h"),
                )

                data = [
                    go.Scattergl(
                        x=[coord[0] for coord in group[col_prefix]],
                        y=[coord[1] for coord in group[col_prefix]],
                        mode="markers",
                        name=sel,
                        marker=dict(size=6, symbol="circle", opacity=0.6, line_width=1),
                        text=group["Title"],
                    )
                    for sel, group in self.data.groupby(color)
                ]
                figure = go.Figure(data=data, layout=layout)

            else:  # 3d
                axes = dict(showgrid=True, zeroline=True, showticklabels=False)
                layout = go.Layout(
                    margin=dict(l=0, r=0, b=0, t=40),
                    legend=dict(x=0, y=0, orientation="h"),
                )

                data = [
                    go.Scatter3d(
                        x=[coord[0] for coord in group[col_prefix]],
                        y=[coord[1] for coord in group[col_prefix]],
                        z=[coord[2] for coord in group[col_prefix]],
                        mode="markers",
                        name=sel,
                        marker=dict(size=3, symbol="circle", opacity=0.6, line_width=1),
                        text=group["Title"],
                    )
                    for sel, group in self.data.groupby(color)
                ]
                figure = go.Figure(data=data, layout=layout)

            figure.update_layout(
                title=f"{method.upper()} embeddings of {emb_type}s",
                scene=dict(
                    xaxis_title="Dim 1",
                    yaxis_title="Dim 2",
                    zaxis_title="Dim 3" if dim == "3d" else None,
                ),
            )

            return [figure]

        @app.callback(
            Output("click-information", "children"),
            [Input("graph-papers", "clickData"), Input("dropdown-dim", "value")],
        )
        def display_info(click_data, dim):
            if not click_data:
                return "Each point in the plot is a paper, select one to view more information."

            coords = [
                click_data["points"][0][i]
                for i in ["x", "y"] + (["z"] if dim == "3d" else [])
            ]

            # Find the paper with matching coordinates
            col_name = f"{emb_type}_{method}_{dim}"
            coords_str = ",".join(map(str, coords))
            matching_papers = self.data[self.data[col_name] == coords_str]

            if not matching_papers.empty:
                return self.make_paper_info(matching_papers.index[0])
            return "Paper not found"


plotter = Plotter()

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

dash_app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=external_stylesheets,
    serve_locally=True,
)

app = dash_app.server
dash_app.layout = plotter.create_layout(dash_app)
plotter.run_callbacks(dash_app)

if __name__ == "__main__":
    dash_app.run_server(host="0.0.0.0", debug=True)
