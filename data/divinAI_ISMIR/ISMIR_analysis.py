#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
import country_converter as coco
import matplotlib

from matplotlib import pyplot as plt
from itertools import cycle, islice

EDITIONS = ["2000", "2001", "2002", "2003", "2004",
            "2005", "2006", "2007", "2008", "2009", 
            "2010", "2011", "2012", "2013", "2014",
            "2015", "2016", "2017", "2018", "2019",
            "2020", "2021", "2022", "2023"]

COLOR_LIST = ['darkblue', 'mediumblue', 'cornflowerblue', 
              'darkred', 'red', 'tomato', 'lightsalmon']


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

def plot_aff_type():
    """
    """
    aff_df = pd.DataFrame.from_dict(typeDict, orient='index')
    my_colors = list(islice(cycle(COLOR_LIST), None, len(aff_df)))
    aff_df.plot(kind='bar', stacked=True, color=my_colors, width=.8)


    plt.ylabel('% of papers')
    plt.legend(['education','facility','company',
                'academia & facility','academia & company',
                'facility & company','academia & facility & company'])

    plt.title('')
    plt.ylim([0,100])
    plt.xticks(np.arange(24), EDITIONS, rotation=90, ha='center')
    plt.show()


def plot_aff_country():
    """
    """

    aff_df = pd.DataFrame.from_dict(CountryCountDictSorted, orient='index')
    aff_df.plot(kind='barh', stacked=True)


    # plt.ylabel('number of papers')
    plt.legend(['Single country', 'Cross-country'])
    plt.title('')
    plt.ylim([-.5,19.5])
    plt.yticks(np.arange(20), 
                         [key for key, val in take(20,CountryCountDictSorted.items())], 
                         rotation=0)
    plt.show()


def plot_worldmap():
    """
    """
    # Setting the path to the shapefile
    SHAPEFILE = 'worldmap/ne_10m_admin_0_countries.shp'
    # Read shapefile using Geopandas
    geo_df = gpd.read_file(SHAPEFILE)[['ADMIN', 'ADM0_A3', 'geometry']]
    # Rename columns.
    geo_df.columns = ['country', 'country_code', 'geometry']
    geo_df = geo_df.drop(geo_df.loc[geo_df['country'] == 'Antarctica'].index)


    iso3_codes = geo_df['country'].to_list()
    # Convert to iso3_codes
    iso2_codes_list = coco.convert(names=iso3_codes, to='ISO2', not_found='NULL')
    # Add the list with iso2 codes to the dataframe
    geo_df['iso2_code'] = iso2_codes_list
    # There are some countries for which the converter could not find a country code. 
    # We will drop these countries.
    geo_df = geo_df.drop(geo_df.loc[geo_df['iso2_code'] == 'NULL'].index)


    CountryCountDictPlot = {key: val for key, val in temp2 if key!="unknown"}
    country_df = pd.DataFrame(list(CountryCountDictPlot.items()), columns=['Country', 'Value'])

    merged_df = pd.merge(left=geo_df, right=country_df, how='left', left_on='country_code', right_on='Country')

    merged_df["Value"].fillna(1, inplace=True)

    merged_df["Value"] = np.log(merged_df["Value"])
    print(merged_df)

    col = 'Value'
    vmin = merged_df[col].min()
    vmax = merged_df[col].max()
    cmap = 'viridis'
    # Create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(20, 8))
    # Remove the axis
    ax.axis('off')
    merged_df.plot(column=col, ax=ax, edgecolor='0.8', linewidth=1, cmap=cmap)

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    # Empty array for the data range
    sm._A = []
    # Add the colorbar to the figure
    cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
    cbar = fig.colorbar(sm, cax=cbaxes)

    plt.show()



if __name__ == "__main__":

    
    dfISMIR = pd.read_csv("data/ISMIR2024/ISMIR-full-20240328.tsv", delimiter="\t")
    print(dfISMIR)


    type_e = 0
    type_f = 0
    type_c = 0
    type_ef = 0
    type_ec = 0
    type_fc = 0
    type_efc = 0

    type_lists = []

    typeDict = {}
    countryDict = {}

    country_names = set()

    for ed in EDITIONS:
        print("###",ed)

        dfISMIRyear = dfISMIR[dfISMIR["Year"] == int(ed)]
        countryDict[ed] = {}

        for index, row in dfISMIRyear.iterrows():

            aff_types = [x.strip() for x in row["Affiliation type"].split(",")]
            aff_countries = [x.strip() for x in row["Affiliation country"].split(",")]

            # Check list lenghts
            it = iter([aff_types,aff_countries])
            the_len = len(next(it))
            if not all(len(l) == the_len for l in it):
                 raise ValueError('not all lists have same length!')

            # Aff types
            if len(set(aff_types)) == 1:
                if "education" in aff_types:
                    type_e += 1
                elif "facility" in aff_types:
                    type_f += 1
                elif "company" in aff_types:
                    type_c += 1
            elif len(set(aff_types)) == 2:
                if "education" in aff_types and "facility" in aff_types:
                    type_ef += 1
                elif "education" in aff_types and "company" in aff_types:
                    type_ec += 1
                elif "facility" in aff_types and "company" in aff_types:
                    type_fc += 1
            elif len(set(aff_types)) == 3:
                type_efc += 1


            # Aff countries
            if len(set(aff_countries)) == 1:
                co = next(iter(set(aff_countries)))
                country_names.add(co)
                if co not in countryDict[ed]:
                    countryDict[ed][co] = {"sc":1, "mc":0}
                else:
                    countryDict[ed][co]["sc"] += 1
            else:
                for co in set(aff_countries):
                    country_names.add(co)
                    if co not in countryDict[ed]:
                        countryDict[ed][co] = {"sc":0, "mc":1}
                    else:
                        countryDict[ed][co]["mc"] += 1



        type_list = [type_e, type_f, type_c, type_ef, type_ec, type_fc, type_efc]
        type_list_perc = [round(x*100/np.sum(type_list), 2) for x in type_list]

        typeDict[ed] = type_list_perc
        
        print(type_list)
        print(type_list_perc)
        print(len(dfISMIRyear), len(countryDict[ed]))




    plot_aff_type()

    CountryCountDict = {}
    for co in country_names:
        if co == 'unknown':
            continue
        CountryCountDict[co] = [0,0]
        for ed in EDITIONS:
            if co in countryDict[ed]:
                CountryCountDict[co][0] += countryDict[ed][co]['sc']
                CountryCountDict[co][1] += countryDict[ed][co]['mc']

    # Sort Dict Countries
    temp1 = {val: sum(int(idx) for idx in key) 
               for val, key in CountryCountDict.items()}
    # using sorted to perform sorting as required
    temp2 = sorted(temp1.items(), key = lambda ele : temp1[ele[0]], reverse=True)
    # rearrange into dictionary
    CountryCountDictSorted = {key: CountryCountDict[key] for key, val in temp2 if key!="unknown"}


    plot_aff_country()

    plot_worldmap()

