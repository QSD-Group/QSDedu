#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
QSDsan: Quantitative Sustainable Design for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <mailto.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/QSDsan/blob/main/LICENSE.txt
for license details.
'''

import os, numpy as np, pandas as pd, country_converter as coco
from matplotlib import pyplot as plt
from systems import create_mcda
from models import create_model, country_params, country_params_units


# %%

# =============================================================================
# Update values and calculate results
# =============================================================================

# Import data
dir_path = os.path.dirname(__file__)
path = os.path.join(dir_path, 'data/contextual_parameters.xlsx')
file = pd.ExcelFile(path)

read_excel = lambda name: pd.read_excel(file, name) # name is sheet name

countries = read_excel('Countries')

def get_country_val(sheet, country, index_col='Code', val_col='Value'):
    df = read_excel(sheet) if isinstance(sheet, str) else sheet
    idx = df[df.loc[:, index_col]==country].index
    val = df.loc[idx, val_col]

    # When no country-specific data or no this country, use continental data
    if (val.isna().any() or val.size==0):
        region = countries[countries.Code==country].Region.item()
        idx = df[df.loc[:, index_col]==region].index
        val = df.loc[idx, val_col]

        # If not even continental data or no this country, use world data
        if (val.isna().any() or val.size==0):
            idx = df[df.loc[:, index_col]=='World'].index
            val = df.loc[idx, val_col]
    return val.values.item()

# Country-specific input values
val_dct_cached = {} # for cached results
def lookup_val(country):
    val_dct = val_dct_cached.get(country)
    if val_dct is not None: return val_dct

    country = coco.convert(country)
    if country == 'not found': return

    val_dct = val_dct_cached[country] = {
        'Caloric intake': get_country_val('Caloric Intake', country),
        'Vegetable protein intake': get_country_val('Vegetal Protein', country),
        'Animal protein intake': get_country_val('Animal Protein', country),
        'N fertilizer price': get_country_val('N Fertilizer Price', country),
        'P fertilizer price': get_country_val('P Fertilizer Price', country),
        'K fertilizer price': get_country_val('K Fertilizer Price', country),
        'Food waste ratio': get_country_val('Food Waste', country)/100,
        'Price level ratio': get_country_val('Price Level Ratio', country),
        'Income tax': get_country_val('Tax Rate', country)/100,
        }
    return val_dct

def get_results(country, models=()):
    if not models:
        modelA = create_model('A', country_specific=True)
        modelB = create_model('B', country_specific=True)
        models = modelA, modelB
    else: modelA, modelB = models

    # Update the baseline values of the models based on the country
    paramA_dct = {param.name: param for param in modelA.parameters}
    paramB_dct = {param.name: param for param in modelB.parameters}

    if isinstance(country, str):
        val_dct = val_dct_cached.get(country)
        if val_dct is None: val_dct = val_dct_cached[country] = lookup_val(country)
    else:
        val_dct = country

    global result_dct
    result_dct = {}
    for model in models:
        param_dct = paramA_dct if model.system.ID[-1]=='A' else paramB_dct
        for reg_name, param_name in country_params.items():
            param = param_dct[param_name]
            param.baseline = val_dct[reg_name]
        result_dct[model.system.ID] = model.metrics_at_baseline()
    return result_dct

# Cache Uganda results for comparison
result_dct_uganda = get_results('Uganda')


# %%

# =============================================================================
# Prettify things for displaying
# =============================================================================

val_df_cached = {} # for cached results
def get_val_df(data):
    if isinstance(data, str) and data != 'customized':
        country = data # assume country name is provided
        val_df = val_df_cached.get(data)
        if val_df is not None: return val_df

        val_dct = lookup_val(data)
        if not val_dct:
            return f'No available information for country "{data}."'
    else:
        val_dct = data # assume that the `val_dct` is provided instead of country name
        country = 'customized'

    val_df = val_df_cached[country] = pd.DataFrame({
        'Parameter': val_dct.keys(),
        'Value': val_dct.values(),
        'Unit': country_params_units.values(),
        })
    return val_df


metric_names = [
    'N recovery [fraction]',
    'P recovery [fraction]',
    'K recovery [fraction]',
    'Net cost [¢/cap/yr]',
    'Net emission [g CO2-e/cap/d]',
    ]
def extract_vals(df):
    vals = [df.recovery.loc[m] for m in metric_names[:3]]
    vals.append(df.TEA.loc[metric_names[-2]])
    vals.append(df.LCA.loc[metric_names[-1]])
    return vals


def plot(data, mcda=None, econ_weight=0.5):
    if isinstance(data, str): # assume to be the country name
        result_dct = get_results(data)
    elif isinstance(data, dict): # assume to be compiled results dict
        result_dct = data
    else: # assume to be model objects
        modelA, modelB = data
        result_dct = {model.system.ID: model.metrics_at_baseline() for model in data}

    dfA, dfB = result_dct.values()
    valsA = extract_vals(dfA)
    valsB = extract_vals(dfB)

    fig, axs = plt.subplots(1, 4, figsize=(9, 4.5),
                            gridspec_kw={'width_ratios': [2.5, 1, 1, 1]})
    ax1, ax2, ax3, ax4 = axs
    ylabel_size = 12
    xticklabel_size = 10

    # Recoveries
    bar_width = 0.3
    recovery_labels = ('N', 'P', 'K')
    x = np.arange(len(recovery_labels))
    ax1.bar(x-bar_width/2, valsA[:3], label='sysA', width=bar_width)
    ax1.bar(x+bar_width/2, valsB[:3], label='sysB', width=bar_width)
    ax1.set_ylabel('Recovery', fontsize=ylabel_size)
    ax1.set_xticks(x, recovery_labels, fontsize=xticklabel_size)

    # Cost
    bar_width /= 4
    x = np.array([0])
    ax2.bar(x-bar_width, valsA[-2], label='sysA', width=bar_width)
    ax2.bar(x+bar_width, valsB[-2], label='sysB', width=bar_width)
    ax2.set_ylabel(metric_names[-2], fontsize=ylabel_size)
    ax2.set_xticks(x, ['Cost'], fontsize=xticklabel_size)

    # Emission
    ax3.bar(x-bar_width, valsA[-1], label='sysA', width=bar_width)
    ax3.bar(x+bar_width, valsB[-1], label='sysB', width=bar_width)
    ax3.set_ylabel(metric_names[-1], fontsize=ylabel_size)
    ax3.set_xticks(x, ['Emission'], fontsize=xticklabel_size)

    # Score
    mcda = mcda or create_mcda()
    mcda.indicator_type.Econ[0] = mcda.indicator_type.Env[0] = 0 # net cost/emission
    ind_score_df = mcda.indicator_scores.copy()
    for num, vals in enumerate((valsA, valsB)):
        ind_score_df.loc[num, 'Econ'] = vals[-2]
        ind_score_df.loc[num, 'Env'] = vals[-1]

    mcda.run_MCDA(
        criterion_weights=(econ_weight, 1-econ_weight),
        indicator_scores=ind_score_df)
    perfm_scores = mcda.performance_scores

    ax4.bar(x-bar_width, perfm_scores.sysA, label='sysA', width=bar_width)
    ax4.bar(x+bar_width, perfm_scores.sysB, label='sysB', width=bar_width)
    ax4.set_ylabel('Score', fontsize=ylabel_size)
    ax4.set_xticks(x, ['Score'], fontsize=xticklabel_size)

    for ax in axs: ax.legend()
    fig.tight_layout()
    plt.close()
    return fig