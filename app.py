"""
WEIRD ISMIR Explorer visualization app on Dash

Copyright 2024, J.S. Gómez-Cañón, Erick Siavichay
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import logging
import pandas as pd
from collections import Counter
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.offline as pyo
import ast
import re
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Plotter:
    """Handles the visualization of ISMIR paper data using Dash."""

    LAYOUT_COMMON = dict(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=0, orientation="h"),
    )

    def __init__(self):
        """Initialize plotter with data and clean it."""
        self.data = pd.read_csv("data/ismir_all_papers.csv")
        self.data['point_index'] = list(range(0, self.data.shape[0], 1))
        self.data['Affiliation country'] = self.data['Authors with Affiliations'].apply(lambda x: ', '.join([entry.split('>')[1] for entry in x.split(';')]))
        self.data['Affiliation type'] = self.data['Authors with Affiliations'].apply(lambda x: ', '.join([entry.split('>')[-1] for entry in x.split(';')]))
        self.data['UN Categories'] = self.data['Affiliation country'].apply(self.map_countries_to_categories)

        self.data = self.data[self.data["Abstract"] != '""']

        coord_cols = [
            f"{emb}_{method}_{dim}"
            for emb in ["title", "abstract"]
            for method in ["tsne", "umap"]
            for dim in ["2d", "3d"]
        ]

        for col in coord_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].str.strip().str.strip('"')
                self.data[col] = self.data[col].apply(ast.literal_eval)

    def map_countries_to_categories(self, iso_list):
        """Load UN categorization"""
        un_cat = pd.read_csv("data/UN_categorization.csv")[['ISO Code', 'Economic Category']]
        iso_to_category = dict(zip(un_cat['ISO Code'], un_cat['Economic Category']))
        return ', '.join([iso_to_category.get(iso.strip(), 'Unknown') for iso in iso_list.split(',')])

    def make_paper_info(self, idx: int) -> html.Div:
        """Creates a layout div containing information about a specific paper."""
        url = self.data.loc[idx, "Link"]
        authors = self.data.loc[idx, "Authors"]
        title = self.data.loc[idx, "Title"]
        abstract = self.data.loc[idx, "Abstract"]
        year = self.data.loc[idx, "Year"]
        countries = self.data.loc[idx, "Affiliation country"].split(", ")
        country_counts = Counter(countries)
        locations = list(country_counts.keys())
        counts = list(country_counts.values())
        fig_map = go.Figure()
        fig_map.add_trace(
            go.Choropleth(
                locations=locations,
                z=counts,
                colorscale="plasma",
                reversescale=False,
                autocolorscale=False,
                colorbar_title="Counts",
                showscale=False, 
            )
        )
        fig_map.add_trace(
            go.Scattergeo(
                locations=locations,
                locationmode="ISO-3",
                text=[f"{loc}: {cnt}" for loc, cnt in zip(locations, counts)],  # Country: Count
                mode="text",  # Display only text
                textfont=dict(color="black", size=15, family="Arial"),
            )
        )
        fig_map.update_traces(
            hovertext=[f"{loc}: {cnt}" for loc, cnt in zip(locations, counts)],
            hoverinfo="text",
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
                html.P(f"Title: {title} ({year})"),
                html.P(f"Authors: {authors}"),
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
        """Creates the main layout for the Dash application."""
        layout = html.Div(
            style={"background-color": "#ffffff"},
            children=[
                html.Div(
                    className="row header",
                    style={"background-color": "#f9f9f9", "margin": "5px 5px 5px 5px"},
                    children=[
                        html.H3(
                            "Beyond a western center of MIR: A bibliometric analysis of 2000-2024 ISMIR authorship",
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
                                                            value="umap",
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
        """Sets up the callbacks for the Dash application."""

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
            try:
                col_prefix = f"{emb_type}_{method}_{dim}"

                if dim == "2d":
                    data = [
                        go.Scattergl(
                            x=[coord[0] for coord in group[col_prefix]],
                            y=[coord[1] for coord in group[col_prefix]],
                            mode="markers",
                            name=sel,
                            marker=dict(
                                size=6, symbol="circle", opacity=0.6, line_width=1
                            ),
                            text=group.apply(lambda row: f"Title: {row['Title']}<br>Year: {row['Year']}<br>Index: {row['point_index']}", axis=1),
                            hoverinfo="text"
                        )
                        for sel, group in self.data.groupby(color)
                    ]
                    figure = go.Figure(data=data, layout=self.LAYOUT_COMMON)

                else:  # 3d
                    data = [
                        go.Scatter3d(
                            x=[coord[0] for coord in group[col_prefix]],
                            y=[coord[1] for coord in group[col_prefix]],
                            z=[coord[2] for coord in group[col_prefix]],
                            mode="markers",
                            name=sel,
                            marker=dict(
                                size=3, symbol="circle", opacity=0.6, line_width=1
                            ),
                            text=group.apply(lambda row: f"Title: {row['Title']}<br>Year: {row['Year']}<br>Index: {row['point_index']}", axis=1),
                            hoverinfo="text"
                        )
                        for sel, group in self.data.groupby(color)
                    ]
                    figure = go.Figure(data=data, layout=self.LAYOUT_COMMON)

                figure.update_layout(
                    title=f"{method.upper()} embeddings of {emb_type}s",
                    scene=dict(
                        xaxis_title="Dim 1",
                        yaxis_title="Dim 2",
                        zaxis_title="Dim 3" if dim == "3d" else None,
                    ),
                )
                return [figure]
            except Exception as e:
                logger.error(f"Error creating plot: {e}")
                return [go.Figure()]

        @app.callback(
            Output("click-information", "children"),
            [Input("graph-papers", "clickData")],
        )
        def display_info(click_data):
            if not click_data:
                return "Each point in the plot is a paper, select one to view more information."
            try:
                match = re.search(r"Index: (\d+)", click_data["points"][0]['text'])
                if match:
                    point_index = int(match.group(1)) 

                if point_index >= len(self.data):
                    return "Invalid paper index"
                return self.make_paper_info(point_index)
            except Exception as e:
                logger.error(f"Error processing click data: {e}")
                return "Error processing paper information"


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
