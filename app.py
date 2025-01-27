#!/usr/bin/env python3
"""
WEIRD ISMIR Explorer visualization app on Dash


Copyright 2024, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import numpy as np
import pandas as pd
import os
import subprocess
from collections import Counter


import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go


class Plotter():
    def __init__(self):
        self.data = pd.read_csv('./data/output_Title_tsne_2d_emb.csv', sep=';')

    def make_paper_info(self, idx):
        df_subset = self.data.iloc[idx]
        url = self.data.loc[idx, 'Link']
        authors = self.data.loc[idx, 'Authors']
        title = self.data.loc[idx, 'Title']
        abstract = self.data.loc[idx, 'Abstract']
        year = self.data.loc[idx, 'Year']
        countries = self.data.loc[idx, 'Affiliation country'].split(', ')
        # print(countries)
        # print(list(Counter(countries).values()))
        # print(list(Counter(countries).keys()))

        fig_map = go.Figure()
        # fig_map.add_trace(go.Scattergeo(
        #     locations=countries,
        #     # colorscale='Reds'
        #     ))
        fig_map.add_trace(go.Choropleth(
            locations=list(Counter(countries).keys()),
            z=list(Counter(countries).values()),
            colorscale = 'Reds',
            reversescale=True,
            autocolorscale=False,
            colorbar_title = 'Counts',
            ))

        fig_map.update_geos(showcountries=True)
        fig_map.update_layout(height=300, 
            margin={"r":0,"t":0,"l":0,"b":0},
            geo=dict(
                    showframe=False,
                    showcoastlines=False,
                    projection_type='equirectangular'
                ),
            )
        # fig_map.update_layout(title='Affiliations')

        # spoti_link_emb = 'https://open.spotify.com/embed/track/{}'.format(df.loc[idx, 'track_id'])
        # spoti_link = 'https://open.spotify.com/track/{}'.format(df.loc[idx, 'track_id'])
        # muziek_link = 'https://www.muziekweb.nl/Embed/{}?theme=static&color=dark'.format(df.loc[idx, 'cdr_track_num'])


        # # figure quads
        # fig_quads = go.Figure()
        # fig_quads.add_trace(go.Bar(x=df_subset[list(self.quads.values())], y=list(self.quads.keys()), orientation='h'))
        # fig_quads.update_layout(title='Quadrant frequency')
        #
        # # figure moods
        # fig_moods = go.Figure()
        # fig_moods.add_trace(go.Barpolar(r=df_subset[self.tags], theta=self.tags))
        #
        # fig_moods.update_layout(title='Mood frequency',
        #     polar=dict(angularaxis=dict(showline=False), radialaxis=dict(visible = False)),
        #     font_size=11)
        #
        # # figure wordcloud
        # try:
        #     txt_cld = df.iloc[idx].txt_free + ' ' + df.iloc[idx].txt_quad + ' ' + df.iloc[idx].txt_mood
        #     cloud = self.get_word_cloud(txt_cld)
        # except:
        #     cloud = np.zeros((500,500))
        # fig_cld = go.Figure(go.Image(z=cloud))
        # fig_cld.update_layout(title='Wordcloud')
        # fig_cld.update_xaxes(visible=False)
        # fig_cld.update_yaxes(visible=False)

        layout = html.Div([
            # html.Iframe(src=url+"", style={'marginRight': 50, 'marginLeft': 50, 'height': '68vh', 'width': '60vh'}),
            html.H6('Article information:'),

            # html.Iframe(src='https://zenodo.org/records/1416794/preview/OliverK06.pdf?include_deleted=0', style={'marginRight': 50, 'marginLeft': 50, 'height': '68vh', 'width': '60vh'}),
            html.P('Title: {}'.format(title)),
            html.P('Authors: {}'.format(authors)),
            html.P('Year: {}'.format(year)),
            html.P('Abstract: {}'.format(abstract)),
            # html.P('Link to Zenodo:'),
            html.A(
                html.Img(src='assets/zenodo.png', alt='zenodo_logo',style={"width": "12%"}),
                href=url,
                target='_blank',
                
            ),
            # html.Embed(src=spoti_link_emb, height=80, width=300, style={'marginRight': 50, 'marginLeft': 50}),
            # html.H6('Summary:'),
            # html.P('Summary: {0:.0f} annotators - preference ({1:.0f}%) - familiarity ({2:.0f}%) - tempo ({3:.1f} BPM)'.format(df.iloc[idx].num_users,
            #                                                                                                           float(df.iloc[idx].pref) * 100,
            #                                                                                                           float(df.iloc[idx].fam) * 100,
            #                                                                                                           df.iloc[idx].tempo),
            #        style={'text-align': 'center'}),
            # html.Div(children=[
            #
            # html.Div(children=[
            #     dcc.Graph(figure=fig_quads,  style={'height': '35vh'}),
            # ], className='four columns'),
            #
            # html.Div(children=[
            #     dcc.Graph(figure=fig_moods,  style={'height': '35vh'}),
            # ], className='eight columns'),
            # ], className='twelve columns'),
            #
            html.Div(children=[
                dcc.Graph(figure=fig_map),
            ], className='twelve columns'),
            
        ], className='twelve columns')

        return layout
    def create_layout(self, app):
        with open('assets/intro_text.md', 'r') as file:
            intro_txt = file.read()
        layout = html.Div(style={'background-color': '#ffffff'}, children=[
            # header
            html.Div(className="row header",
                     style={"background-color": "#f9f9f9",
                            "margin": "5px 5px 5px 5px"},
                     children=[
                         # html.A(
                         #     html.Img(src=app.get_asset_url('mtg.png'), alt='mtg_logo', height=70),
                         #     href='https://www.upf.edu/web/mtg/',
                         #     target='_blank',
                         #     className='three columns'),
                         # html.Div(className='three columns'),
                         html.H3('Not so WEIRD: A bibliometric analysis of ISMIR authorship',
                                 style={'text-align': 'right'},
                                 className='nine columns'),
                     ]),

            # text and graph
            html.Section(className='row', style={'padding': '0px'}, children=[
                html.Div(children=[
                    # dcc.Markdown(intro_txt),

                    html.Div(className='row', children=[

                        html.Div(children=[
                            dcc.Graph(id='graph-papers', style={'height': '68vh'}),
                        ], className='nine columns'),

                        html.Div(children=[

                            html.Div(children=['Select plot dimensionality:',
                                               dcc.Dropdown(
                                                   style={"margin": "0px 5px 5px 0px"},
                                                   id='dropdown-dim',
                                                   # placeholder='Select AV representation:',
                                                   searchable=False,
                                                   clearable=False,
                                                   options=[
                                                       {'label': '2D', 'value': '2d'},
                                                       {'label': '3D', 'value': '3d'},
                                                   ],
                                                   value='2d',
                                               ),
                                               ],
                                     ),

                            html.Div(children=['Select source of embeddings:',
                                               dcc.Dropdown(
                                                   style={"margin": "0px 5px 5px 0px"},
                                                   id='dropdown-embeddings',
                                                   placeholder='Select source of embeddings:',
                                                   searchable=False,
                                                   clearable=False,
                                                   options=[
                                                       {'label': 'Paper titles', 'value': 'Title'},
                                                       {'label': 'Paper abstracts', 'value': 'Abstract'},
                                                   ],
                                                   value='Abstract',
                                               ),
                                               ],
                                     ),

                            html.Div(children=['Colorize using properties:',
                                               dcc.Dropdown(
                                                   style={"margin": "0px 5px 5px 0px"},
                                                   id='dropdown-color',
                                                   placeholder='Colorize using properties:',
                                                   searchable=False,
                                                   clearable=False,
                                                   options=[
                                                       {'label': 'Year', 'value': 'Year'},
                                                       {'label': 'Country', 'value': 'Country Split 0'},
                                                       {'label': 'Affiliation Type', 'value': 'Entity Type'},
                                                       {'label': 'UN categories', 'value': 'First Author Affiliation UN Category'},
                                                   ],
                                                   value='Year',
                                               ),
                                               ],
                                     ),



                            # html.Div(
                            #     style={"margin": "0px 5px 5px 0px"},
                            #     children=["Mode",
                            #               dcc.Slider(
                            #                   id="slider-mode",
                            #                   min=-1,
                            #                   max=1,
                            #                   step=None,
                            #                   value=-1,
                            #                   marks={-1: 'All',
                            #                          0: 'Minor',
                            #                          1: 'Major'},
                            #                   vertical=False,
                            #               ),
                            #               ],
                            # ),
                            # html.Div(
                            #     style={"margin": "0px 5px 5px 0px"},
                            #     children=["Tempo",
                            #               dcc.RangeSlider(
                            #                   id="slider-tempo",
                            #                   min=0,
                            #                   max=220,
                            #                   step=1,
                            #                   value=[0, 220],
                            #                   tooltip={"placement": "bottom", "always_visible": True},
                            #                   allowCross=False,
                            #               ),
                            #               ],
                            # ),
                            #
                            # html.Div(
                            #     style={"margin": "0px 5px 5px 0px"},
                            #     children=["Key",
                            #               dcc.Slider(
                            #                   id="slider-key",
                            #                   min=0,
                            #                   max=15,
                            #                   step=None,
                            #                   value=15,
                            #                   marks={0: 'C',
                            #                          1: 'C#',
                            #                          2: 'D',
                            #                          3: 'D#',
                            #                          4: 'E',
                            #                          5: 'F',
                            #                          6: 'F#',
                            #                          7: 'G',
                            #                          8: 'G#',
                            #                          9: 'A',
                            #                          10: 'A#',
                            #                          11: 'B',
                            #                          15: 'All'},
                            #
                            #                   vertical=False,
                            #               ),
                            #               ],
                            # ),
                            #
                            # html.Div(
                            #     style={"margin": "0px 5px 5px 0px"},
                            #     children=["Num. Annotators",
                            #               dcc.RangeSlider(
                            #                   id="slider-users",
                            #                   min=0,
                            #                   max=np.max(self.data.num_users),
                            #                   step=1,
                            #                   value=[0, np.max(self.data.num_users)],
                            #                   tooltip={"placement": "bottom", "always_visible": True},
                            #                   vertical=False,
                            #               ),
                            #               ],
                            # ),
                            #
                            # html.Div(id='agreement-text',
                            #          children=[]),

                        ], className='three columns'),
                    ]),

                ], className='six columns',
                ),

                # annotation info
                html.Div(
                    [
                        html.Table(
                            id="table-element",
                            className="table__container",
                        )
                    ],
                    id="click-information",
                    className='six columns',
                ),
            ]),

            # footer
            # html.Div(className="row footer",
            #          style={"background-color": "#f9f9f9"},
            #          children=[
            #
            #              html.P('Created by Juan Sebastián Gómez-Cañón',
            #                     style={'text-align': 'center',
            #                            'color': 'grey',
            #                            'font-size': '10px'}, ),
            #              html.Div(
            #                  children=[
            #                      html.A(
            #                          html.Img(src=app.get_asset_url('juan_gomez.png'), alt='juan_logo', height=25),
            #                          href='https://juansgomez87.github.io/',
            #                          target='_blank',
            #                          style={"margin": "0px 15px 0px 15px"}),
            #                      html.A(
            #                          html.Img(src=app.get_asset_url('twitter.png'), alt='twitter', height=25),
            #                          href='https://twitter.com/juan_s_gomez',
            #                          target='_blank',
            #                          style={"margin": "0px 15px 0px 15px"}),
            #                      html.A(
            #                          html.Img(src=app.get_asset_url('github.png'), alt='github', height=25),
            #                          href='https://github.com/juansgomez87',
            #                          target='_blank',
            #                          style={"margin": "0px 15px 0px 15px"}),
            #                      html.A(
            #                          html.Img(src=app.get_asset_url('scholar.png'), alt='scholar', height=25),
            #                          href='https://scholar.google.com/citations?user=IvIQqUwAAAAJ&hl=en',
            #                          target='_blank',
            #                          style={"margin": "0px 15px 0px 15px"}),
            #                  ],
            #                  style={'text-align': 'center'}, ),
            #          ]),

        ])
        return layout

    def run_callbacks(self, app):
        @app.callback(
            [Output("graph-papers", "figure"),
             # Output('agreement-text', 'children')
             ],
            [
                Input("dropdown-dim", "value"),
                Input("dropdown-embeddings", "value"),
                Input("dropdown-color", "value"),
                # Input("slider-key", "value"),
                # Input("slider-tempo", "value"),
                # Input("slider-users", "value"),
                # Input('dropdown-filters', 'value')
            ],
        )
        def display_plot(dim, emb_type, color):
            filename = 'data/output_{}_tsne_{}_emb.csv'.format(emb_type, dim)
            self.data = pd.read_csv(filename, sep=';')


            # dimensionality
            if dim == '2d':
                axes = dict(showgrid=True, zeroline=True, showticklabels=False)

                layout = go.Layout(
                    margin=dict(l=0, r=0, b=0, t=40),
                    scene=dict(xaxis=axes, yaxis=axes),
                    legend=dict(x=0, y=0, orientation="h"),
                )

                data = []
                for sel, group in self.data.groupby(color):
                    scatter = go.Scattergl(
                        x=group['dim_1'],
                        y=group['dim_2'],
                        mode='markers',
                        name=sel,
                        marker=dict(size=6, symbol='circle', opacity=0.6,
                                    line_width=1),
                        text=group['Title'],
                    )
                    data.append(scatter)

                figure = go.Figure(data=data, layout=layout)
                figure.update_layout(title='Embeddings {}'.format(emb_type),
                                     scene=dict(
                                         xaxis_title='Dim 1',
                                         yaxis_title='Dim 2',
                                     ))
            elif dim == '3d':
                axes = dict(showgrid=True, zeroline=True, showticklabels=False)

                layout = go.Layout(
                    margin=dict(l=0, r=0, b=0, t=40),
                    scene=dict(xaxis=axes, yaxis=axes),
                    legend=dict(x=0, y=0, orientation="h"),
                )

                data = []
                for sel, group in self.data.groupby(color):
                    scatter = go.Scatter3d(
                        x=group['dim_1'],
                        y=group['dim_2'],
                        z=group['dim_3'],
                        mode='markers',
                        name=sel,
                        marker=dict(size=3, symbol='circle', opacity=0.6,
                                    line_width=1),
                        text=group['Title'],
                    )
                    data.append(scatter)

                figure = go.Figure(data=data, layout=layout)
                figure.update_layout(title='Embeddings {}'.format(emb_type),
                                     scene=dict(
                                         xaxis_title='Dim 1',
                                         yaxis_title='Dim 2',
                                         zaxis_title='Dim 3'
                                     ))

            return [figure]


        @app.callback(
            Output('click-information', 'children'),
            [
                Input("graph-papers", 'clickData'),
                Input("dropdown-dim", "value"),
                # Input("dropdown-arousalvalence", "value"),
                # Input("dropdown-color", "value"),
                # Input("slider-mode", "value"),
                # Input("slider-key", "value"),
                # Input("slider-tempo", "value"),
            ],
        )
        def display_info(click_data, dim):
            # this_df = self.data

            # # filter with musical properties
            # if sl_mode != -1:
            #     this_df = this_df[this_df['mode'] == sl_mode].reset_index()
            # if sl_key != 15:
            #     this_df = this_df[this_df['key'] == sl_key].reset_index()
            # if sl_tempo != [0, 220]:
            #     this_df = this_df[(this_df['tempo'] >= sl_tempo[0]) & (this_df['tempo'] <= sl_tempo[1])].reset_index()
            #
            # # show arousal valence representations
            # if av_rep == 'spoti_api':
            #     aro_col = 'energy'
            #     val_col = 'valence'
            # elif av_rep == 'norm':
            #     aro_col = 'norm_energy'
            #     val_col = 'norm_valence'
            #
            # # colorize according to features
            # if spoti_filt == 'none':
            #     this_color = None
            #     show_scale = False
            # else:
            #     this_color = this_df[spoti_filt]
            #     show_scale = True

            text = 'Each point in the plot is a song, select one to view more information.'
            if click_data and dim == '2d':
                click_point_np = np.array([click_data['points'][0][i] for i in ['x', 'y']]).astype(np.float64)
                bool_mask = (self.data.loc[:, ['dim_1', 'dim_2']].eq(click_point_np).all(axis=1))

                if bool_mask.any():
                    idx = self.data[bool_mask == True].index[0]

                    return self.make_paper_info(idx)
            elif click_data and dim == '3d':
                click_point_np = np.array([click_data['points'][0][i] for i in ['x', 'y', 'z']]).astype(np.float64)
                bool_mask = (self.data.loc[:, ['dim_1', 'dim_2', 'dim_3']].eq(click_point_np).all(axis=1))

                if bool_mask.any():
                    idx = self.data[bool_mask == True].index[0]

                    return self.make_paper_info(idx)

            else:
                return 'Each point in the plot is a paper, select one to view more information.'

# instanciate Plotter
plotter = Plotter()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width"}],
    external_stylesheets=external_stylesheets,
    # comment two following lines for local tests
    # routes_pathname_prefix='/',
    # requests_pathname_prefix='/vis-mtg-mer/',
    serve_locally=True
)

app = dash_app.server

dash_app.layout = plotter.create_layout(dash_app)

plotter.run_callbacks(dash_app)


if __name__ == "__main__":
    dash_app.run_server(host='0.0.0.0', debug=True)