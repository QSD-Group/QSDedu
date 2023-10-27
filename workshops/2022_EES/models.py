#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
QSDsan: Quantitative Sustainable Design for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <mailto.yalin.li@gmail.com>

Part of this module is based on the EXPOsan repository:
https://github.com/QSD-Group/EXPOsan

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/QSDsan/blob/main/LICENSE.txt
for license details.

Ref:
    [1] Trimmer et al., Navigating Multidimensional Social–Ecological System
        Trade-Offs across Sanitation Alternatives in an Urban Informal Settlement.
        Environ. Sci. Technol. 2020, 54 (19), 12641–12653.
        https://doi.org/10.1021/acs.est.0c03296.
'''


# %%

import os, numpy as np, pandas as pd, qsdsan as qs
from chaospy import distributions as shape
from thermosteam.functional import V_to_rho, rho_to_V
from qsdsan import Model, Metric
from qsdsan.utils import (
    ospath, load_data, data_path,
    DictAttrSetter, copy_samples
    )
from exposan import bwaise as bw
from exposan.utils import batch_setting_unit_params
from exposan.bwaise.models import add_LCA_CF_parameters, add_pit_latrine_parameters

dir_path = os.path.dirname(__file__)


# %%

# =============================================================================
# Functions for batch-making metrics and -setting parameters
# =============================================================================

import systems as s
from systems import (
    create_system,
    get_daily_cap_cost as get_cost,
    get_daily_cap_ghg as get_ghg,
    get_decay_k,
    get_recovery,
    )


def add_LCA_metrics(system, metrics):
    unit = 'g CO2-e/cap/d'
    cat = 'LCA'
    metrics.extend([
        Metric('Net emission', lambda: get_ghg(system, 'net', False), unit, cat),
        Metric('Construction', lambda: get_ghg(system, 'construction', False), unit, cat),
        Metric('Transportation', lambda: get_ghg(system, 'transportation', False), unit, cat),
        Metric('Direct', lambda: get_ghg(system, 'direct', False), unit, cat),
        Metric('Offset', lambda: get_ghg(system, 'offset', False), unit, cat),
        ])
    return metrics

def add_metrics(system):
    metrics = []
    cat = 'recovery'
    metrics.extend([
        Metric('N recovery', lambda: get_recovery(system, 'N', False), 'fraction', cat),
        Metric('P recovery', lambda: get_recovery(system, 'P', False), 'fraction', cat),
        Metric('K recovery', lambda: get_recovery(system, 'K', False), 'fraction', cat),
        ])

    unit = '¢/cap/yr'
    cat = 'TEA'
    metrics.extend([
        Metric('Net cost', lambda: get_cost(system, 'net', False), unit, cat),
        Metric('CAPEX', lambda: get_cost(system, 'CAPEX', False), unit, cat),
        Metric('OPEX', lambda: get_cost(system, 'OPEX', False), unit, cat),
        Metric('sales', lambda: get_cost(system, 'sales', False), unit, cat),
        ])

    metrics = add_LCA_metrics(system, metrics)

    return metrics


# %%

# =============================================================================
# Shared by all three systems
# =============================================================================

su_data_path = ospath.join(data_path, 'sanunit_data/')

def add_shared_parameters(model, main_crop_application_unit):
    ########## Related to multiple units ##########
    sys = model.system
    sys_stream = sys.flowsheet.stream
    param = model.parameter
    tea = sys.TEA

    # UGX-to-USD
    Excretion = sys.path[0]
    b = bw.exchange_rate
    D = shape.Triangle(lower=3600, midpoint=b, upper=3900)
    @param(name='Exchange rate', element=Excretion, kind='cost', units='UGX/USD',
           baseline=b, distribution=D)
    def set_exchange_rate(i):
        bw.exchange_rate = i

    ########## Related to human input ##########
    # Diet and excretion
    path = ospath.join(data_path, 'sanunit_data/_excretion.tsv')
    excretion_data = load_data(path)
    batch_setting_unit_params(excretion_data, model, Excretion)

    # Household size
    Toilet = sys.path[1]
    b = bw.household_size
    D = shape.Trunc(shape.Normal(mu=b, sigma=1.8), lower=1)
    @param(name='Household size', element=Toilet, kind='coupled', units='cap/household',
           baseline=b, distribution=D)
    def set_household_size(i):
        bw.household_size = i

    # Toilet
    b = bw.household_per_toilet
    D = shape.Uniform(lower=3, upper=5)
    @param(name='Toilet density', element=Toilet, kind='coupled', units='household/toilet',
           baseline=b, distribution=D)
    def set_toilet_density(i):
        bw.household_per_toilet = i

    path = ospath.join(data_path, 'sanunit_data/_toilet.tsv')
    toilet_data = load_data(path)
    batch_setting_unit_params(toilet_data, model, Toilet,
                              exclude=('desiccant_rho',)) # set separately

    toilet_type = type(Toilet).__name__

    cmps = qs.get_components()
    WoodAsh = cmps.WoodAsh
    b = V_to_rho(WoodAsh.V(298.15), WoodAsh.MW)
    D = shape.Triangle(lower=663, midpoint=b, upper=977)
    @param(name=f'{toilet_type} desiccant density', element=Toilet, kind='coupled',
           units='kg/m3', baseline=b, distribution=D)
    def set_desiccant_density(i):
        WoodAsh.V.local_methods['USER_METHOD'].value = rho_to_V(i, WoodAsh.MW)
        setattr(Toilet, 'desiccant_rho', i)

    b = WoodAsh.i_Mg
    D = shape.Triangle(lower=0.008, midpoint=b, upper=0.0562)
    @param(name=f'{toilet_type} desiccant Mg content', element=Toilet, kind='coupled',
           units='fraction', baseline=b, distribution=D)
    def set_desiccant_Mg(i):
        WoodAsh.i_Mg = i

    b = WoodAsh.i_Ca
    D = shape.Triangle(lower=0.0742, midpoint=b, upper=0.3716)
    @param(name=f'{toilet_type} desiccant Ca content', element=Toilet, kind='coupled',
           units='fraction', baseline=b, distribution=D)
    def set_desiccant_Ca(i):
        WoodAsh.i_Ca = i

    ##### Universal degradation parameters #####
    # Max methane emission
    unit = sys.path[1] # the first unit that involves degradation
    b = s.max_CH4_emission
    D = shape.Triangle(lower=0.175, midpoint=b, upper=0.325)
    @param(name='Max CH4 emission', element=unit, kind='coupled', units='g CH4/g COD',
           baseline=b, distribution=D)
    def set_max_CH4_emission(i):
        s.max_CH4_emission = i
        for unit in sys.units:
            if hasattr(unit, 'max_CH4_emission'):
                setattr(unit, 'max_CH4_emission', i)

    # Time to full degradation
    b = bw.tau_deg
    D = shape.Uniform(lower=1, upper=3)
    @param(name='Full degradation time', element=unit, kind='coupled', units='yr',
           baseline=b, distribution=D)
    def set_tau_deg(i):
        bw.tau_deg = i
        k = get_decay_k(i, bw.log_deg)
        for unit in sys.units:
            if hasattr(unit, 'decay_k_COD'):
                setattr(unit, 'decay_k_COD', k)
            if hasattr(unit, 'decay_k_N'):
                setattr(unit, 'decay_k_N', k)

    # Reduction at full degradation
    b = bw.log_deg
    D = shape.Uniform(lower=2, upper=4)
    @param(name='Log degradation', element=unit, kind='coupled', units='-',
           baseline=b, distribution=D)
    def set_log_deg(i):
        bw.log_deg = i
        k = get_decay_k(bw.tau_deg, i)
        for unit in sys.units:
            if hasattr(unit, 'decay_k_COD'):
                setattr(unit, 'decay_k_COD', k)
            if hasattr(unit, 'decay_k_N'):
                setattr(unit, 'decay_k_N', k)

    ##### Toilet material properties #####
    density = unit.density_dct
    b = density['Plastic']
    D = shape.Uniform(lower=0.31, upper=1.24)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Plastic'),
          name='Plastic density', element=unit, kind='isolated', units='kg/m2',
          baseline=b, distribution=D)

    b = density['Brick']
    D = shape.Uniform(lower=1500, upper=2000)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Brick'),
          name='Brick density', element=unit, kind='isolated', units='kg/m3',
          baseline=b, distribution=D)

    b = density['StainlessSteelSheet']
    D = shape.Uniform(lower=2.26, upper=3.58)
    param(setter=DictAttrSetter(unit, 'density_dct', 'StainlessSteelSheet'),
          name='SS sheet density', element=unit, kind='isolated', units='kg/m2',
          baseline=b, distribution=D)

    b = density['Gravel']
    D = shape.Uniform(lower=1520, upper=1680)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Gravel'),
          name='Gravel density', element=unit, kind='isolated', units='kg/m3',
          baseline=b, distribution=D)

    b = density['Sand']
    D = shape.Uniform(lower=1281, upper=1602)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Sand'),
          name='Sand density', element=unit, kind='isolated', units='kg/m3',
          baseline=b, distribution=D)

    b = density['Steel']
    D = shape.Uniform(lower=7750, upper=8050)
    param(setter=DictAttrSetter(unit, 'density_dct', 'Steel'),
          name='Steel density', element=unit, kind='isolated', units='kg/m3',
          baseline=b, distribution=D)

    ########## Crop application ##########
    unit = main_crop_application_unit
    D = shape.Uniform(lower=0, upper=0.1)
    param(setter=DictAttrSetter(unit, 'loss_ratio', 'NH3'),
          name='NH3 application loss', element=unit, kind='coupled',
          units='fraction of applied', baseline=0.05, distribution=D)

    # Mg, Ca, C actually not affecting results
    D = shape.Uniform(lower=0, upper=0.05)
    param(setter=DictAttrSetter(unit, 'loss_ratio', ('NonNH3', 'P', 'K', 'Mg', 'Ca')),
          name='Other application losses', element=unit, kind='coupled',
          units='fraction of applied', baseline=0.02, distribution=D)

    ######## General TEA settings ########
    # Discount factor for the excreta-derived fertilizers
    b = bw.price_factor
    D = shape.Uniform(lower=0.1, upper=0.4)
    @param(name='Price factor', element='TEA', kind='isolated', units='-',
           baseline=b, distribution=D)
    def set_price_factor(i):
        bw.price_factor = i

    D = shape.Uniform(lower=1.164, upper=2.296)
    @param(name='N fertilizer price', element='TEA', kind='isolated', units='USD/kg N',
           baseline=1.507, distribution=D)
    def set_N_price(i):
        price = i * bw.price_factor
        mixed_N = sys_stream.search('mixed_N')
        if mixed_N: mixed_N.price = price
        else: sys_stream.liq_N.price = sys_stream.sol_N.price = price

    D = shape.Uniform(lower=2.619, upper=6.692)
    @param(name='P fertilizer price', element='TEA', kind='isolated', units='USD/kg P',
           baseline=3.983, distribution=D)
    def set_P_price(i):
        price = i * bw.price_factor
        mixed_P = sys_stream.search('mixed_P')
        if mixed_P: mixed_P.price = price
        else: sys_stream.liq_P.price = sys_stream.sol_P.price = price

    D = shape.Uniform(lower=1.214, upper=1.474)
    @param(name='K fertilizer price', element='TEA', kind='isolated', units='USD/kg K',
           baseline=1.333, distribution=D)
    def set_K_price(i):
        price = i * bw.price_factor
        mixed_K = sys_stream.search('mixed_K')
        if mixed_K: mixed_K.price = price
        else: sys_stream.liq_K.price = sys_stream.sol_K.price = price

    # Money discount rate
    b = tea.discount_rate
    D = shape.Uniform(lower=0.03, upper=0.06)
    @param(name='Discount rate', element='TEA', kind='isolated', units='fraction',
           baseline=b, distribution=D)
    def set_discount_rate(i):
        tea.discount_rate = i


# %%

# =============================================================================
# Pit latrine
# =============================================================================

def create_modelA(**model_kwargs):
    system = model_kwargs.pop('system', None)
    sysA = create_system('A') if not system else system
    modelA = Model(sysA, add_metrics(sysA), **model_kwargs)
    add_shared_parameters(modelA, modelA.system.flowsheet.unit.A4)
    add_LCA_CF_parameters(modelA, lca_kind='original')
    add_pit_latrine_parameters(modelA)
    return modelA


# %%

# =============================================================================
# UDDT
# =============================================================================

def create_modelB(**model_kwargs):
    system = model_kwargs.pop('system', None)
    sysB = create_system('B') if not system else system
    modelB = Model(sysB, add_metrics(sysB), **model_kwargs)
    unitB = modelB.system.flowsheet.unit
    paramB = modelB.parameter
    add_shared_parameters(modelB, unitB.B5)
    add_LCA_CF_parameters(modelB, lca_kind='original')

    # UDDT
    B2 = unitB.B2
    path = ospath.join(su_data_path, '_uddt.tsv')
    uddt_data = load_data(path)
    batch_setting_unit_params(uddt_data, modelB, B2)

    b = B2.CAPEX
    D = shape.Uniform(lower=476, upper=630)
    @paramB(name='UDDT capital cost', element=B2, kind='cost',
           units='USD/toilet', baseline=b, distribution=D)
    def set_UDDT_CAPEX(i):
        B2.CAPEX = i

    b = B2.OPEX_over_CAPEX
    D = shape.Uniform(lower=0.05, upper=0.1)
    @paramB(name='UDDT annual operating cost', element=B2, kind='cost',
           units='fraction of capital cost', baseline=b, distribution=D)
    def set_UDDT_OPEX(i):
        B2.OPEX_over_CAPEX = i

    # Conveyance
    B3 = unitB.B3
    B4 = unitB.B4
    b = B3.loss_ratio
    D = shape.Uniform(lower=0.02, upper=0.05)
    @paramB(name='Transportation loss', element=B3, kind='coupled', units='fraction',
           baseline=b, distribution=D)
    def set_trans_loss(i):
        B3.loss_ratio = B4.loss_ratio = i

    b = B3.single_truck.distance
    D = shape.Uniform(lower=2, upper=10)
    @paramB(name='Transportation distance', element=B3, kind='coupled', units='km',
           baseline=b, distribution=D)
    def set_trans_distance(i):
        B3.single_truck.distance = B4.single_truck.distance = i

    b = bw.handcart_fee
    D = shape.Uniform(lower=0.004, upper=0.015)
    @paramB(name='Handcart fee', element=B3, kind='cost', units='USD/cap/d',
           baseline=b, distribution=D)
    def set_handcart_fee(i):
        bw.handcart_fee = i

    b = bw.truck_fee
    D = shape.Uniform(lower=17e3, upper=30e3)
    @paramB(name='Truck fee', element=B3, kind='cost', units='UGX/m3',
           baseline=b, distribution=D)
    def set_truck_fee(i):
        bw.truck_fee = i

    return modelB

country_params = {
    'Caloric intake': 'Excretion e cal',
    'Vegetable protein intake': 'Excretion p veg',
    'Animal protein intake': 'Excretion p anim',
    'N fertilizer price': 'N fertilizer price',
    'P fertilizer price': 'P fertilizer price',
    'K fertilizer price': 'K fertilizer price',
    'Food waste ratio': 'Food waste ratio', # not in the original model
    'Price level ratio': 'Price level ratio', # not in the original model
    'Income tax': 'Income tax', # not in the original model
    }
country_params_units = {
    'Caloric intake': 'kcal/cap/d',
    'Vegetable protein intake': 'g/cap/d',
    'Animal protein intake': 'g/cap/d',
    'N fertilizer price': 'USD/kg N',
    'P fertilizer price': 'USD/kg P',
    'K fertilizer price': 'USD/kg K',
    'Food waste ratio': 'fraction',
    'Price level ratio': '',
    'Income tax': 'fraction',
    }

def create_model(model_ID='A', country_specific=False, **model_kwargs):
    model_ID = model_ID.lstrip('model').lstrip('sys') # so that it'll work for "modelA"/"sysA"/"A"
    if model_ID == 'A': model = create_modelA(**model_kwargs)
    elif model_ID == 'B': model = create_modelB(**model_kwargs)
    else: raise ValueError(f'`model_ID` can only be "A" or "B", not "{model_ID}".')
    if country_specific: # add the remaining three more country-specific parameters
        # model.parameters = [p for p in model.parameters if p.name not in country_params.values()]
        param = model.parameter
        system = model.system

        unit = system.path[0]
        b = unit.waste_ratio
        D = shape.Uniform(lower=b*0.9, upper=b*1.1)
        @param(name='Food waste ratio', element=unit, kind='cost', units='fraction',
               baseline=b, distribution=D)
        def set_food_waste_ratio(i):
            unit.waste_ratio = i

        b = s.price_ratio
        D = shape.Uniform(lower=b*0.9, upper=b*1.1)
        @param(name='Price level ratio', element='TEA', kind='cost', units='',
               baseline=b, distribution=D)
        def set_price_ratio(i):
            s.price_ratio = i

        tea = system.TEA
        b = tea.income_tax
        D = shape.Uniform(lower=b*0.9, upper=b*1.1)
        @param(name='Income tax', element='TEA', kind='cost', units='fraction',
               baseline=b, distribution=D)
        def set_income_tax(i):
            tea.income_tax = i

    return model


# %%

# =============================================================================
# Functions to run simulation and generate plots
# =============================================================================

def run_uncertainty(model, N=1000, seed=None,rule='L',
                    percentiles=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1),
                    file_path='',
                    only_load_samples=False,
                    only_organize_results=False):
    if not only_organize_results:
        if seed: np.random.seed(seed)
        samples = model.sample(N, rule)
        model.load_samples(samples)
        if only_load_samples: return

        model.evaluate()

    # Spearman's rank correlation,
    spearman_results = model.spearman_r()[0]
    spearman_results.columns = pd.Index([i.name_with_units for i in model.metrics])

    index_p = len(model.parameters)
    parameters_df = model.table.iloc[:, :index_p].copy()
    results_df = model.table.iloc[:, index_p:].copy()
    percentiles_df = results_df.quantile(q=percentiles)
    spearman_df = spearman_results

    if file_path is not None:
        if not file_path:
            results_path = os.path.join(dir_path, 'results')
            if not os.path.isdir(results_path): os.mkdir(results_path)
            file_path = os.path.join(results_path, f'{model.system.ID}_uncertainties.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            parameters_df.to_excel(writer, sheet_name='Parameters')
            results_df.to_excel(writer, sheet_name='Uncertainty results')
            percentiles_df.to_excel(writer, sheet_name='Percentiles')
            spearman_df.to_excel(writer, sheet_name='Spearman')
            model.table.to_excel(writer, sheet_name='Raw data')
    return model


def run_uncertainties(models=(), N=100, seed=None, rule='L',
                      percentiles=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1),):
    if not models:
        modelA = create_model('A')
        modelB = create_model('B')
        models = modelA, modelB

    for model in models:
        run_uncertainty(model, N, seed, rule, percentiles,
                        only_load_samples=True, only_organize_results=False)
    copy_samples(modelA, modelB)

    for model in models:
        run_uncertainty(model, N, seed, rule, percentiles,
                        only_load_samples=False, only_organize_results=False)

    return modelA, modelB


def get_param_metric(name, model, kind='parameter'):
    kind = 'parameters' if kind.lower() in ('p', 'param', 'parameter', 'parameters') \
        else 'metrics'
    for obj in getattr(model, kind):
        if obj.name == name: break
    return obj


# %%

if __name__ == '__main__':
    global modelA, modelB
    modelA = create_model('A')
    modelB = create_model('B')
