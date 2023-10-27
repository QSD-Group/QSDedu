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
'''


# %%

# Filter out warnings related to solid content
import warnings
warnings.filterwarnings('ignore', message='Solid content')

import numpy as np, pandas as pd, qsdsan as qs
from matplotlib import pyplot as plt
from qsdsan import (
    ImpactItem,
    LCA,
    main_flowsheet,
    sanunits as su,
    SimpleTEA,
    System,
    WasteStream,
    )
from qsdsan.utils import clear_lca_registries
from exposan.bwaise import (
    app_loss,
    create_components,
    discount_rate,
    get_decay_k,
    get_handcart_and_truck_fee,
    get_tanker_truck_fee,
    get_toilet_user,
    max_CH4_emission,
    price_dct,
    )
from exposan.bwaise.systems import (
    _load_lca_data,
    adjust_NH3_loss,
    batch_create_streams,
    update_toilet_param,
    )
from dmsan import MCDA

# Universal unit parameters
ppl = 20000 # number of people served by the toilets
price_ratio = 1 # to account for different in deployment locations
lifetime = 8 # year


# %%

# =============================================================================
# sysA: pit latrine system
# =============================================================================

def create_systemA(flowsheet=None):
    # Set flowsheet to avoid stream replacement warnings
    flowsheet = flowsheet or main_flowsheet
    streamA = flowsheet.stream
    batch_create_streams('A', phases=('mixed',))

    ##### Human Inputs #####
    A1 = su.Excretion('A1', outs=('urine', 'feces'), waste_ratio=0.02) # Uganda

    ##### User Interface #####
    CH4_item = ImpactItem.get_item('CH4_item')
    N2O_item = ImpactItem.get_item('N2O_item')
    WasteStream('pit_CH4', stream_impact_item=CH4_item.copy(set_as_source=True))
    WasteStream('pit_N2O', stream_impact_item=N2O_item.copy(set_as_source=True))
    A2 = su.PitLatrine('A2', ins=(A1-0, A1-1,
                                  'toilet_paper', 'flushing_water',
                                  'cleansing_water', 'desiccant'),
                       outs=('mixed_waste', 'leachate', 'A2_CH4', 'A2_N2O'),
                       N_user=get_toilet_user(), N_toilet=ppl/get_toilet_user(),
                       OPEX_over_CAPEX=0.05,
                       decay_k_COD=get_decay_k(),
                       decay_k_N=get_decay_k(),
                       max_CH4_emission=max_CH4_emission,
                       price_ratio=price_ratio)
    A2.specification = lambda: update_toilet_param(A2, ppl)

    ##### Conveyance #####
    A3 = su.Trucking('A3', ins=A2-0, outs=('transported', 'conveyance_loss'),
                     load_type='mass', distance=5, distance_unit='km',
                     interval=A2.emptying_period, interval_unit='yr',
                     loss_ratio=0.02, price_ratio=price_ratio)
    def update_A3_param():
        A3._run()
        truck = A3.single_truck
        truck.interval = A2.emptying_period*365*24
        truck.load = A3.F_mass_in*truck.interval/A2.N_toilet
        rho = A3.F_mass_in/A3.F_vol_in
        vol = truck.load/rho
        A3.fee = get_tanker_truck_fee(vol)
        A3.price_ratio = price_ratio
        A3._design()
    A3.specification = update_A3_param

    ##### Reuse or Disposal #####
    A4 = su.CropApplication('A4', ins=A3-0, outs=('liquid_fertilizer', 'reuse_loss'),
                            loss_ratio=app_loss)
    A4.specification = lambda: adjust_NH3_loss(A4)

    A5 = su.Mixer('A5', ins=(A2-2,), outs=streamA.CH4)
    A5.line = 'fugitive CH4 mixer'

    A6 = su.Mixer('A6', ins=(A2-3,), outs=streamA.N2O)
    A6.line = 'fugitive N2O mixer'

    A7 = su.ComponentSplitter('A7', ins=A4-0,
                               outs=(streamA.mixed_N, streamA.mixed_P, streamA.mixed_K,
                                     'A_liq_non_fertilizers'),
                               split_keys=(('NH3', 'NonNH3'), 'P', 'K'))

    ##### Simulation, TEA, and LCA #####
    sysA = System('sysA', path=flowsheet.unit)
    teaA = SimpleTEA(system=sysA, discount_rate=discount_rate, income_tax=0.3, # Uganda
                     start_year=2018, lifetime=lifetime, uptime_ratio=1,
                     lang_factor=None, annual_maintenance=0, annual_labor=0)

    lcaA = LCA(system=sysA, lifetime=lifetime, lifetime_unit='yr', uptime_ratio=1,
               annualize_construction=True)

    def update_sysA_lifetime():
        A7._run()
        A7.outs[0].price = price_dct['N']
        A7.outs[1].price = price_dct['P']
        A7.outs[2].price = price_dct['K']
        teaA.lifetime = lcaA.lifetime = lifetime
    A7.specification = update_sysA_lifetime

    return sysA


# %%

# =============================================================================
# sysB: urine-diverting dry toilet (UDDT) system
# =============================================================================

def create_systemB(flowsheet=None):
    flowsheet = flowsheet or main_flowsheet
    streamB = flowsheet.stream
    batch_create_streams('B')

    biogas_item = ImpactItem.get_item('Biogas_item').copy('biogas_item', set_as_source=True)
    WasteStream('biogas', phase='g', price=price_dct['Biogas'], stream_impact_item=biogas_item)

    ##### Human Inputs #####
    B1 = su.Excretion('B1', outs=('urine', 'feces'), waste_ratio=0.02) # Uganda

    ##### User Interface #####
    B2 = su.UDDT('B2', ins=(B1-0, B1-1,
                            'toilet_paper', 'flushing_water',
                            'cleaning_water', 'desiccant'),
                 outs=('liq_waste', 'sol_waste',
                       'struvite', 'HAP', 'B2_CH4', 'B2_N2O'),
                 N_user=get_toilet_user(), N_toilet=ppl/get_toilet_user(),
                 OPEX_over_CAPEX=0.1,
                 decay_k_COD=get_decay_k(),
                 decay_k_N=get_decay_k(),
                 max_CH4_emission=max_CH4_emission,
                 price_ratio=price_ratio)
    B2.specification = lambda: update_toilet_param(B2, ppl)

    ##### Conveyance #####
    # Liquid waste
    B3 = su.Trucking('B3', ins=B2-0, outs=('transported_l', 'loss_l'),
                     load_type='mass', distance=5, distance_unit='km',
                     loss_ratio=0.02)

    # Solid waste
    B4 = su.Trucking('B4', ins=B2-1, outs=('transported_s', 'loss_s'),
                     load_type='mass', load=1, load_unit='tonne',
                     distance=5, distance_unit='km',
                     loss_ratio=0.02)
    def update_B3_B4_param():
        B4._run()
        truck3, truck4 = B3.single_truck, B4.single_truck
        hr = truck3.interval = truck4.interval = B2.collection_period*24
        N_toilet = B2.N_toilet
        ppl_per_toilet = ppl / N_toilet
        truck3.load = B3.F_mass_in * hr / N_toilet
        truck4.load = B4.F_mass_in * hr / N_toilet
        rho3 = B3.F_mass_in/B3.F_vol_in
        rho4 = B4.F_mass_in/B4.F_vol_in
        B3.fee = get_handcart_and_truck_fee(truck3.load/rho3, ppl_per_toilet, True, B2)
        B4.fee = get_handcart_and_truck_fee(truck4.load/rho4, ppl_per_toilet, False, B2)
        B3.price_ratio = B4.price_ratio = price_ratio
        B3._design()
        B4._design()
    B4.specification = update_B3_B4_param

    ##### Reuse or Disposal #####
    B5 = su.CropApplication('B5', ins=B3-0, outs=('liquid_fertilizer', 'liquid_reuse_loss'),
                            loss_ratio=app_loss)
    B5.specification = lambda: adjust_NH3_loss(B5)

    B6 = su.CropApplication('B6', ins=B4-0, outs=('solid_fertilizer', 'solid_reuse_loss'),
                            loss_ratio=app_loss)
    def adjust_B6_NH3_loss():
        B6.loss_ratio.update(B5.loss_ratio)
        adjust_NH3_loss(B6)
    B6.specification = adjust_B6_NH3_loss

    B7 = su.Mixer('B7', ins=(B2-4,), outs=streamB.CH4)
    B7.line = 'fugitive CH4 mixer'

    B8 = su.Mixer('B8', ins=(B2-5,), outs=streamB.N2O)
    B8.line = 'fugitive N2O mixer'

    B9 = su.ComponentSplitter('B9', ins=B5-0,
                               outs=(streamB.liq_N, streamB.liq_P, streamB.liq_K,
                                     'B_liq_non_fertilizers'),
                               split_keys=(('NH3', 'NonNH3'), 'P', 'K'))

    B10 = su.ComponentSplitter('B10', ins=B6-0,
                               outs=(streamB.sol_N, streamB.sol_P, streamB.sol_K,
                                     'B_sol_non_fertilizers'),
                               split_keys=(('NH3', 'NonNH3'), 'P', 'K'))


    ##### Simulation, TEA, and LCA #####
    sysB = System('sysB', path=flowsheet.unit)

    teaB = SimpleTEA(system=sysB, discount_rate=discount_rate, income_tax=0.3, # Uganda
                     start_year=2018, lifetime=lifetime, uptime_ratio=1,
                     lang_factor=None, annual_maintenance=0, annual_labor=0)

    lcaB = LCA(system=sysB, lifetime=lifetime, lifetime_unit='yr', uptime_ratio=1,
               annualize_construction=True)

    def update_sysB_params():
        B10._run()
        B9.outs[0].price = B10.outs[0].price = price_dct['N']
        B9.outs[1].price = B10.outs[1].price = price_dct['P']
        B9.outs[2].price = B10.outs[2].price = price_dct['K']
        teaB.lifetime = lcaB.lifetime = lifetime
    B10.specification = update_sysB_params

    return sysB


# %%

# =============================================================================
# Wrapper function
# =============================================================================

def create_system(system_ID='A', flowsheet=None, lca_kind='original'):
    ID = system_ID.lower().lstrip('sys').upper() # so that it'll work for "sysA"/"A"
    reload_lca = False
    global components
    try: components = qs.get_components()
    except: components = create_components()

    # Set flowsheet to avoid stream replacement warnings
    if flowsheet is None:
        flowsheet_ID = f'bw{ID}'
        if hasattr(main_flowsheet.flowsheet, flowsheet_ID): # clear flowsheet
            getattr(main_flowsheet.flowsheet, flowsheet_ID).clear()
            clear_lca_registries()
            reload_lca = True
        flowsheet = qs.Flowsheet(flowsheet_ID)
        main_flowsheet.set_flowsheet(flowsheet)

    loaded_status = _load_lca_data(lca_kind, reload_lca)

    if system_ID == 'A': system = create_systemA(flowsheet)
    elif system_ID == 'B': system = create_systemB(flowsheet)
    else: raise ValueError(f'`system_ID` can only be "A", or "B", not "{ID}".')

    if loaded_status != lca_kind: # to refresh the impact items
        lca = system.LCA
        for i in lca.lca_streams:
            source_ID = i.stream_impact_item.source.ID
            i.stream_impact_item.source = qs.ImpactItem.get_item(source_ID)

    return system



# %%

# =============================================================================
# Analyses
# =============================================================================

def get_recovery(system, nutrient='N', print_msg=True):
    if nutrient not in ('N', 'P', 'K'):
        raise ValueError('`nutrient` can only be "N", "P", or "K", '
                         f'not "{nutrient}".')
    sum_up = lambda streams: sum(getattr(s, f'T{nutrient}')*s.F_vol for s in streams) # g/hr
    tot_in = sum_up(system.path[0].outs)
    u = system.flowsheet.unit
    if system.ID[-1] == 'A': tot_out = sum_up(u.A7.ins)
    else: tot_out = sum_up((u.B9.ins[0], u.B10.ins[0]))
    recovery = tot_out / ppl / tot_in
    if print_msg: print(f'{nutrient} recovery for {system.ID} is {recovery:.1%}.')
    return recovery


def get_daily_cap_cost(system, kind='net', print_msg=True):
    tea = system.TEA
    kind_lower = kind.lower()
    if kind_lower == 'net':
        cost = (tea.annualized_equipment_cost-tea.net_earnings)
    elif kind_lower in ('capex', 'capital', 'construction'):
        cost = tea.annualized_equipment_cost
    elif kind_lower in ('opex', 'operating', 'operation'):
        cost = tea.AOC
    elif kind_lower in ('sales', 'sale', 'revenue'):
        cost = tea.sales
    else:
        raise ValueError(f'Invalid `kind` input "{kind}", '
                         'try "net", "CAPEX", "OPEX", or "sales".')
    cost = cost / ppl / 365 * 100 # from $/cap/yr to ¢/cap/d
    if print_msg: print(f'Daily {kind} cost for {system.ID} is ¢{cost:.2f}/cap/d.')
    return cost


def get_daily_cap_ghg(system, kind='net', print_msg=True):
    lca = system.LCA
    ind_ID = 'GlobalWarming'
    kind_lower = kind.lower()
    if kind_lower == 'net':
        ghg = lca.total_impacts[ind_ID]
    elif kind_lower in ('capex', 'capital', 'construction'):
        ghg = lca.total_construction_impacts[ind_ID]
    elif kind_lower in ('transportation', 'transporting'):
        ghg = lca.total_transportation_impacts[ind_ID]
    elif kind_lower == 'direct':
        ghg = lca.get_stream_impacts(kind='direct_emission')[ind_ID]
    elif kind_lower == 'offset':
        ghg = -lca.get_stream_impacts(kind='offset')[ind_ID] # make it positive for credits
    elif kind_lower in ('opex', 'operating', 'operation'):
        ghg = lca.total_transportation_impacts[ind_ID] + lca.get_stream_impacts()[ind_ID]
    else:
        raise ValueError(f'Invalid `kind` input "{kind}", '
                         'try "net", "construction", "transportation", '
                         '"operating", "direct", or "offset".')
    ghg = ghg / lca.lifetime / ppl / 365 * 1000 # from kg CO2-e/lifetime to g CO2-e/cap/d
    if print_msg: print(f'Daily {kind} emission for {system.ID} is {ghg:.1f} g CO2-e/cap/d.')
    return ghg

def plot_tea_lca(systems, tea_metrics=('net',), lca_metrics=('net',)):
    sysA, sysB = systems
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1, ax2 = axs
    ylabel_size = 12
    xticklabel_size = 10

    # Cost
    bar_width = 0.3
    x = np.array(range(len(tea_metrics)))

    ax1.bar(x-bar_width,
            [get_daily_cap_cost(sysA, m, False) for m in tea_metrics],
            label='sysA', width=bar_width)
    ax1.bar(x+bar_width,
            [get_daily_cap_cost(sysB, m, False) for m in tea_metrics],
            label='sysB', width=bar_width)
    ax1.set_ylabel('Cost [¢/cap/d]', fontsize=ylabel_size)
    ax1.set_xticks(x, tea_metrics, fontsize=xticklabel_size)

    # Emission
    x = np.array(range(len(lca_metrics)))
    ax2.bar(x-bar_width,
            [get_daily_cap_ghg(sysA, m, False) for m in lca_metrics],
            label='sysA', width=bar_width)
    ax2.bar(x+bar_width,
            [get_daily_cap_ghg(sysB, m, False) for m in lca_metrics],
            label='sysB', width=bar_width)
    ax2.set_ylabel('Emission [g CO2-e/cap/d]', fontsize=ylabel_size)
    ax2.set_xticks(x, lca_metrics, fontsize=xticklabel_size)

    for ax in axs: ax.legend()
    fig.tight_layout()
    plt.close()
    return fig


score_df = pd.DataFrame({
    'Econ': (0, 0),
    'Env': (0, 0),
    })
def get_indicator_scores(systems, tea_metric='net', lca_metric='net'):
    for num, sys in enumerate(systems):
        score_df.loc[num, 'Econ'] = get_daily_cap_cost(sys, tea_metric, print_msg=False)
        score_df.loc[num, 'Env'] = get_daily_cap_ghg(sys, lca_metric, print_msg=False)
    return score_df

supported_criteria = ('Econ', 'Env')
single_cr_df = pd.DataFrame({k: 1 for k in supported_criteria}, dtype='float', index=[0])
single_cr_df['Ratio'] = [':'.join(single_cr_df.values.astype('int').astype('str')[i])
                         for i in range(len(single_cr_df))]
def create_mcda(systems=()):
    systems = systems or [create_system('A'), create_system('B')]
    alt_names = [sys.ID for sys in systems]
    indicator_type = pd.DataFrame({
            'Econ': 0,
            'Env': 0,
            }, index=[0])
    indicator_weights = indicator_type.copy()
    indicator_weights.iloc[0] = 1
    mcda = MCDA(
        alt_names=alt_names,
        systems=systems,
        indicator_type=indicator_type,
        indicator_weights=indicator_weights,
        indicator_scores=get_indicator_scores(systems),
        criteria=supported_criteria,
        criterion_weights=single_cr_df,
        )
    return mcda

def run_mcda(mcda=None, tea_metric='net', lca_metric='net',
             econ_weight=0.5, print_msg=True):
    mcda = mcda or create_mcda()
    mcda.indicator_type.Econ[0] = 0 if tea_metric.lower() not in ('sales', 'sale', 'revenue') else 1
    mcda.indicator_type.Env[0] = 0 if lca_metric.lower() !='offset' else 1
    indicator_scores = get_indicator_scores(mcda.systems, tea_metric, lca_metric)
    mcda.run_MCDA(criterion_weights=(econ_weight, 1-econ_weight),
                  indicator_scores=indicator_scores)
    if print_msg:
        alt_names = mcda.alt_names
        scoreA = mcda.performance_scores[alt_names[0]].item()
        scoreB = mcda.performance_scores[alt_names[1]].item()
        winner = mcda.winners.Winner.item()
        sysA, sysB = mcda.systems
        print(f'The score for {sysA.ID} is {scoreA:.3f}, for {sysB.ID} is {scoreB:.3f}, '
              f'{winner} is selected.')
    return mcda.performance_scores


def plot_mcda(mcda=None, tea_metric='net', lca_metric='net',
             econ_weights=np.arange(0, 1.1, 0.1)):
    mcda = mcda or create_mcda()
    dfs = [run_mcda(mcda, tea_metric, lca_metric, wt, False) for wt in econ_weights]
    scoresA = [df.sysA.item() for df in dfs]
    scoresB = [df.sysB.item() for df in dfs]
    fig, ax = plt.subplots()
    ax.plot(econ_weights, scoresA, '-', label='sysA')
    ax.plot(econ_weights, scoresB, '--', label='sysB')
    ax.set_xlabel('Economic Weight', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.legend()
    plt.close()
    return fig


if __name__ == '__main__':
    global sysA, sysB
    sysA = create_system('A')
    sysB = create_system('B')
    for sys in (sysA, sysB):
        get_daily_cap_cost(sys)
        get_daily_cap_ghg(sys)
    global mcda
    mcda = create_mcda(systems=(sysA, sysB))
    run_mcda(mcda)
