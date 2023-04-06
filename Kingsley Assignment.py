# -*- coding: utf-8 -*-
"""
Created on Tue Apr 4 17:47:06 2023

@author: Kingsley Ejeza
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data from the csv file


def read_data(a, b):
    """
    Reads and imports files from comma seperated values, to a python DataFrame

    Arguments:
    a: string, The name of the csv file which is to be read
    b: integer, indicates the number of rows on the csv file to be
    skipped

    Returns:
    data: A pandas dataframe with all values from the excel file
    data_t: The transposed pandas dataframe
    """
    data = pd.read_csv(a, skiprows=b)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    transposed_data = data.set_index(
        data['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    transposed_data = transposed_data.set_index('Year').dropna(axis=1)
    transposed_data = transposed_data.drop(['Country Name'])
    return data, transposed_data


a = 'API_19_DS2_en_csv_v2_5336371.csv'
b = 4

data, transposed_data = read_data(a, b)

# Slicing the dataframe to get data for the indicators of interest


def indicator_set(c, d, e, f, g, h):
    """
    Reads and selects precise indicators from world bank dataframe,
    to a python DataFrame

    Arguments:
    c - h: 6 selected indicators from the dataset

    Returns:
    ind: A pandas dataframe with specific indicators selected
    """
    ind = data[data['Indicator Name'].isin([c, d, e, f, g, h])]

    return ind


c = 'CO2 emissions (kt)'
d = 'Total greenhouse gas emissions (kt of CO2 equivalent)'
e = 'CO2 emissions from liquid fuel consumption (kt)'
f = 'CO2 emissions from gaseous fuel consumption (kt)'
g = 'CO2 intensity (kg per kg of oil equivalent energy use)'
h = 'Energy use (kg of oil equivalent per capita)'

ind = indicator_set(c, d, e, f, g, h)

# Slicing the dataframe to get data for the countries of interest


def country_set(countries):
    """
    Reads and selects country of interest from world bank dataframe,
    to a python DataFrame

    Arguments:
    countries: A list of countries selected from the dataframe
    Returns:
    specific_count: A pandas dataframe with specific countries selected
    """
    specific_count = ind[ind['Country Name'].isin(countries)]
    specific_count = specific_count.dropna(axis=1)
    specific_count = specific_count.reset_index(drop=True)
    return specific_count


# Selecting the countries specifically
countries = ['Australia', 'China', 'Canada', 'Belgium',
             'Japan', 'France', 'United States', 'United Kingdom']

specific_count = country_set(countries)

# STATISTICS OF THE DATA
stats_desc = specific_count.groupby(["Country Name", "Indicator Name"])
print(stats_desc.describe())


def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / len(dist-1)

    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-1) - 3.0

    return value


def grp_countries_ind(indicator):
    """
    Selects and groups countries based on the specific indicators,
    to a python DataFrame

    Arguments:
    indicator: Choosing the indicator

    Returns:
    grp_ind_con: A pandas dataframe with specific countries selected
    """
    grp_ind_con = specific_count[specific_count["Indicator Name"] == indicator]
    grp_ind_con = grp_ind_con.set_index('Country Name', drop=True)
    grp_ind_con = grp_ind_con.transpose().drop('Indicator Name')
    grp_ind_con[countries] = grp_ind_con[countries].apply(
        pd.to_numeric, errors='coerce', axis=1)
    return grp_ind_con


# Giving each indicator a dataframe
CO2_emission = grp_countries_ind("CO2 emissions (kt)")
Green_emission = grp_countries_ind(
    "Total greenhouse gas emissions (kt of CO2 equivalent)")
CO2_gfc = grp_countries_ind("CO2 emissions from gaseous fuel consumption (kt)")
CO2_lfc = grp_countries_ind("CO2 emissions from liquid fuel consumption (kt)")
CO2_ins = grp_countries_ind(
    "CO2 intensity (kg per kg of oil equivalent energy use)")
ene_use = grp_countries_ind("Energy use (kg of oil equivalent per capita)")

print(skew(CO2_emission))
print(kurtosis(CO2_emission))

# HEAT MAP AND PLOTS

china_df = data[data['Country Name'].isin(['China'])]
china_df = china_df.drop(['Country Name'], axis=1)
china_df = china_df[china_df[
    'Indicator Name'].isin([
        'CO2 emissions (kt)',
        'Total greenhouse gas emissions (kt of CO2 equivalent)',
        'CO2 emissions from liquid fuel consumption (kt)',
        'CO2 emissions from gaseous fuel consumption (kt)',
        'Arable land (% of land area)',
        'Energy use (kg of oil equivalent per capita)'])]
china_df = china_df.set_index('Indicator Name')
china_df = china_df.drop(china_df.loc[:, '1990':'2014'], axis=1)
china_df = china_df.transpose()
china_cor = china_df.corr().round(2)

# plotting the heatmap and specifying the plot parameters
plt.imshow(china_cor, cmap='Accent_r', interpolation='none')
plt.colorbar()
plt.xticks(range(len(china_cor)), china_cor.columns, rotation=90)
plt.yticks(range(len(china_cor)), china_cor.columns)
plt.gcf().set_size_inches(8, 5)
plt.rcParams["figure.dpi"] = 300
# labelling of the little boxes and creation of a legend
labels = china_cor.values
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        plt.text(x, y, '{:.2f}'.format(labels[y, x]), ha='center', va='center',
                 color='black')
plt.title('Correlation Map of Indicators for China')

plt.style.use('seaborn-whitegrid')  # matplotlib plot style

# PLOT 1 (Line Plot)
plt.figure(figsize=(12, 8))
ene_use.plot()
plt.title('Energy Use per Capital')
plt.xlabel('Years')
plt.ylabel('kg of oil eqv. per. capital')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.rcParams["figure.dpi"] = 300
plt.show()

# PLOT 2 (Bar Plot for Emission)
CO2_emission.iloc[15::3].plot(kind='bar', figsize=[10, 4])
plt.title('C02 Emission Over the years', fontsize=12, c='k')
plt.xlabel('Years', c='k')
plt.ylabel('CO2 emissions (kt)', c='k')
plt.rcParams["figure.dpi"] = 300
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.show()

# PLOT 3 (Pie Plot for Arable Lands in pct.)
labels = ['Australia', 'Belgium', 'Canada', 'China',
          'France', 'United Kingdom', 'Japan', 'United States']
countries_2010 = Green_emission.iloc[20]
plt.figure(figsize=(12, 6))
plt.pie(countries_2010, labels=labels, autopct='%.2f%%')
plt.rcParams["figure.dpi"] = 300
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Green Emission in 2010')
plt.show()
