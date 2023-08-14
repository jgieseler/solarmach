import os

import astropy
import astropy.units as u
import matplotlib
import numpy as np
import pandas
import pfsspy
from solarmach import SolarMACH, print_body_list, get_gong_map, calculate_pfss_solution


def test_print_body_list():
    df = print_body_list()
    assert isinstance(df, pandas.core.frame.DataFrame)
    assert df['Body'].loc['PSP'] == 'Parker Solar Probe'


def test_solarmach_initialize():
    body_list = ['STEREO-A']
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
    assert np.round(sm.coord_table["Longitudinal separation between body's mangetic footpoint and reference_long"][0], 2) == 19.33

    # verify backwards compatibility: undefined coord_sys is interpreted as 'Carrington'
    assert sm.coord_sys == 'Carrington'


def test_solarmach_get_sw_speed():
    body_list = ['Earth', 'STEREO-A', 'BepiColombo']
    date = '2021-10-28 15:15:00'
    sm = SolarMACH(date=date, body_list=body_list, coord_sys='Stonyhurst')
    assert np.round(sm.coord_table[sm.coord_table['Spacecraft/Body']=='STEREO-A']['Vsw'].values[0]) == 365.0
    assert sm.coord_table[sm.coord_table['Spacecraft/Body']=='BepiColombo']['Vsw'].values[0] == 400.0


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
    sm.plot(plot_spirals=True, plot_sun_body_line=True,
            reference_vsw=reference_vsw, transparent=False,
            show_earth_centered_coord=False, numbered_markers=True,
            long_sector=long_sector, long_sector_vsw=long_sector_vsw, long_sector_color=long_sector_color,
            background_spirals=background_spirals, outfile=filename)

    assert os.path.exists(os.getcwd()+os.sep+filename)


def test_solarmach_pfss():
    date = '2021-4-1 1:00:00'
    body_list = ['Earth', 'STEREO-A']
    vsw_list = [400, 400]   # position-sensitive solar wind speed per body in body_list
    sm = SolarMACH(date, body_list, vsw_list, reference_long=100, reference_lat=10)
    gong_map = get_gong_map(time=date, filepath=None)
    assert isinstance(gong_map, pfsspy.map.GongSynopticMap)
    pfss_solution = calculate_pfss_solution(gong_map=gong_map, rss=2.5)
    assert isinstance(pfss_solution, pfsspy.output.Output)
    fig, ax = sm.plot_pfss(rss=2.5, pfss_solution=pfss_solution, vary=True, return_plot_object=True,
                           numbered_markers=True, long_sector=[290, 328], long_sector_vsw=[400, 600],
                           long_sector_color='red', reference_vsw=400.0)
    assert isinstance(fig, matplotlib.figure.Figure)
