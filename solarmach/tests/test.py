import os

import astropy
import astropy.units as u
import datetime as dt
import matplotlib
import numpy as np
import pandas
import sunkit_magex.pfss as pfsspy
import pytest
import sunpy
from pathlib import Path
from solarmach import SolarMACH, print_body_list, get_gong_map, calculate_pfss_solution, sc_distance, get_sw_speed


def test_print_body_list():
    df = print_body_list()
    assert isinstance(df, pandas.core.frame.DataFrame)
    assert df['Body'].loc['PSP'] == 'Parker Solar Probe'


def test_solarmach_initialize():
    body_list = ['STEREO-A', 'JUICE']
    vsw_list = [400]
    date = '2021-10-28 15:15:00'
    reference_long = 273
    reference_lat = 9

    sm = SolarMACH(date=date, body_list=body_list, vsw_list=vsw_list, reference_long=reference_long, reference_lat=reference_lat)

    assert sm.date == date
    assert sm.reference_lat == reference_lat
    assert sm.reference_long == reference_long
    assert np.round(sm.max_dist, 3) == 0.958

    assert isinstance(sm.pos_E, astropy.coordinates.sky_coordinate.SkyCoord)
    assert sm.pos_E.lat.unit == u.deg
    assert sm.pos_E.lon.unit == u.deg
    assert sm.pos_E.radius.unit == u.AU

    assert sm.coord_table.shape == (1, 11)

    assert np.round(sm.coord_table['Longitudinal separation between body and reference_long'][0], 1) == -39.9
    assert np.round(sm.coord_table["Longitudinal separation between body's magnetic footpoint and reference_long"][0], 2) == 19.33

    # verify backwards compatibility: undefined coord_sys is interpreted as 'Carrington'
    assert sm.coord_sys == 'Carrington'


def test_solarmach_get_sw_speed():
    body_list = ['STEREO-A', 'SOHO', 'Parker Solar Probe', 'Solar Orbiter', 'Wind', 'ACE', 'Earth', 'BepiColombo']
    date = '2023-02-28 15:00:00'
    sm = SolarMACH(date=date, body_list=body_list, coord_sys='Stonyhurst')
    try:
        import speasy as spz
        vsw_stereoa = 636.0
        vsw_soho = 655.0
        vsw_psp = 476.0
        vsw_solo = 496.0
        vsw_wind = 597.0
        vsw_ace = 616.0
    except ModuleNotFoundError:
        vsw_stereoa = 400.0
        vsw_soho = 400.0
        vsw_psp = 400.0
        vsw_solo = 400.0
        vsw_wind = 400.0
        vsw_ace = 400.0
    assert np.round(sm.coord_table[sm.coord_table['Spacecraft/Body']=='STEREO-A']['Vsw'].values[0]) == vsw_stereoa
    assert np.round(sm.coord_table[sm.coord_table['Spacecraft/Body']=='SOHO']['Vsw'].values[0]) == vsw_soho
    assert np.round(sm.coord_table[sm.coord_table['Spacecraft/Body']=='Parker Solar Probe']['Vsw'].values[0]) == vsw_psp
    assert np.round(sm.coord_table[sm.coord_table['Spacecraft/Body']=='Solar Orbiter']['Vsw'].values[0]) == vsw_solo
    assert np.round(sm.coord_table[sm.coord_table['Spacecraft/Body']=='Wind']['Vsw'].values[0]) == vsw_wind
    assert np.round(sm.coord_table[sm.coord_table['Spacecraft/Body']=='ACE']['Vsw'].values[0]) == vsw_ace
    assert sm.coord_table[sm.coord_table['Spacecraft/Body']=='Earth']['Vsw'].values[0] == sm.coord_table[sm.coord_table['Spacecraft/Body']=='ACE']['Vsw'].values[0]
    assert sm.coord_table[sm.coord_table['Spacecraft/Body']=='BepiColombo']['Vsw'].values[0] == 400.0


def test_solarmach_wrong_datetime_format():
    body_list = ['Earth', 'STEREO-A', 'BepiColombo']
    date = '202110-28 15:15:00'
    default_vsw = 999.9

    # check only get_sw_speed
    vsw_earth = get_sw_speed(body_list[0], date, trange=1, default_vsw=default_vsw)
    assert vsw_earth == default_vsw

    # check only sc_distance
    distance = sc_distance(body_list[0], body_list[1], date)
    assert np.isnan(distance.value)
    assert distance.unit == u.AU

    # check SolarMACH
    with pytest.raises(ValueError):
        sm = SolarMACH(date=date, body_list=body_list, coord_sys='Stonyhurst')


"""
Create/update hash library for the following matplotlib tests by running for example the following command from the base package dir:
tox -e py310-test -- --mpl-generate-hash-library=solarmach/tests/figure_hashes_mpl_390.json --mpl-deterministic
"""


@pytest.mark.mpl_image_compare(hash_library=Path(__file__).parent / 'figure_hashes_mpl_390.json', deterministic=True)
def test_solarmach_plot():
    body_list = ['STEREO-A']
    vsw_list = [400]
    date = '2021-10-28 15:15:00'
    reference_long = 273
    reference_lat = 9
    reference_vsw = 400
    filename = 'test.png'
    long_sector=[290, 328]
    long_sector_vsw=[400, 600]
    long_sector_color='red'
    background_spirals=[6, 600]

    sm = SolarMACH(date=date, body_list=body_list, vsw_list=vsw_list, reference_long=reference_long, reference_lat=reference_lat)
    fig, ax = sm.plot(plot_spirals=True, plot_sun_body_line=True,
                      reference_vsw=reference_vsw, transparent=False,
                      show_earth_centered_coord=False, markers='numbers',
                      long_sector=long_sector, long_sector_vsw=long_sector_vsw, long_sector_color=long_sector_color,
                      background_spirals=background_spirals, outfile=filename, return_plot_object=True)
    assert os.path.exists(os.getcwd()+os.sep+filename)
    return fig


@pytest.mark.mpl_image_compare(hash_library=Path(__file__).parent / 'figure_hashes_mpl_390.json', deterministic=True)
def test_solarmach_pfss():
    date = '2021-4-1 1:00:00'
    body_list = ['Earth', 'STEREO-A']
    vsw_list = [400, 400]   # position-sensitive solar wind speed per body in body_list
    sm = SolarMACH(date, body_list, vsw_list, reference_long=100, reference_lat=10, coord_sys='Carrington')
    gong_map = get_gong_map(time=date, filepath=None)
    assert isinstance(gong_map, sunpy.map.sources.gong.GONGSynopticMap)
    pfss_solution = calculate_pfss_solution(gong_map=gong_map, rss=2.5, coord_sys='Carrington')
    assert isinstance(pfss_solution, pfsspy.output.Output)
    fig, ax = sm.plot_pfss(rss=2.5, pfss_solution=pfss_solution, vary=True, return_plot_object=True,
                           markers='numbers', long_sector=[290, 328], long_sector_vsw=[400, 600],
                           long_sector_color='red', reference_vsw=400.0)
    assert isinstance(fig, matplotlib.figure.Figure)
    return fig


def test_sc_distance():
    distance = sc_distance('SolO', 'PSP', "2020/12/12")
    assert np.round(distance.value, 8) == 1.45237361
    assert distance.unit == u.AU
    #
    distance = sc_distance('SolO', 'PSP', dt.date(2020, 12, 12))
    assert np.round(distance.value, 8) == 1.45237361
    assert distance.unit == u.AU
    #
    distance = sc_distance('SolO', 'PSP', "2000/12/12")
    assert np.isnan(distance.value)
    assert distance.unit == u.AU
