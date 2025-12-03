import datetime as dt
# import hashlib
import os
from pathlib import Path

import astropy
from astropy.time import Time
import astropy.units as u
import matplotlib
import numpy as np
import pandas as pd
import pytest
import sunkit_magex.pfss as pfsspy
import sunpy

from solarmach import (
    SolarMACH,
    get_sw_speed,
    print_body_list,
    sc_distance,
    sto2car_sun,
    car2sto_sun,
)
from solarmach.pfss_utilities import calculate_pfss_solution, get_gong_map


def test_print_body_list():
    df = print_body_list()
    assert isinstance(df, pd.core.frame.DataFrame)
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
Create/update hash library and baseline images for the following matplotlib tests by running e.g. the following command from the base package dir (replace py312 with installed Python version):
tox -e py312-test -- --mpl-generate-hash-library=solarmach/tests/figure_hashes_mpl_391.json --mpl-deterministic

Because for test_solarmach_pfss() the hash comparison always failed on GitHub Actions, fall back to plain image comparison mode for it.
To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=solarmach/tests/baseline
"""


@pytest.mark.parametrize("coord_sys, markers, fix_earth, long_offset", [('Carrington', 'numbers', True, 270), ('Stonyhurst', 'letters', False, 0)])
# @pytest.mark.mpl_image_compare(hash_library=Path(__file__).parent / 'figure_hashes_mpl_391.json', deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning:solarmach")
def test_solarmach_plot(coord_sys, markers, fix_earth, long_offset):
    body_list = ['STEREO-A']
    vsw_list = [400]
    date = '2021-10-28 15:15:00'
    reference_long = 273
    reference_lat = 9
    reference_vsw = 400
    filename = 'tmp_solarmach.png'
    long_sector=[290, 328]
    long_sector_vsw=[400, 600]
    long_sector_color='red'
    background_spirals=[6, 600]

    sm = SolarMACH(date=date, body_list=body_list, vsw_list=vsw_list, reference_long=reference_long, reference_lat=reference_lat, coord_sys=coord_sys)
    fig, ax = sm.plot(plot_spirals=True, plot_sun_body_line=True,
                      reference_vsw=reference_vsw, transparent=False,
                      markers=markers, fix_earth=fix_earth,
                      long_sector=long_sector, long_sector_vsw=long_sector_vsw, long_sector_color=long_sector_color,
                      background_spirals=background_spirals, outfile=filename, return_plot_object=True)
    assert os.path.exists(os.getcwd()+os.sep+filename)
    return fig


# Because for test_solarmach_pfss() the hash comparison always failed on GitHub Actions, fall back to plain image comparison mode for it.
# @pytest.mark.mpl_image_compare(hash_library=Path(__file__).parent / 'figure_hashes_mpl_391.json', deterministic=True, remove_text=True)
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning:solarmach")
def test_solarmach_pfss():
    date = '2021-4-1 1:00:00'
    body_list = ['Earth', 'PSP']
    vsw_list = [400, 400]   # position-sensitive solar wind speed per body in body_list
    sm = SolarMACH(date, body_list, vsw_list, reference_long=180, reference_lat=10, coord_sys='Carrington')
    gong_map = get_gong_map(time=date, filepath=None)
    assert isinstance(gong_map, sunpy.map.sources.gong.GONGSynopticMap)
    pfss_solution = calculate_pfss_solution(gong_map=gong_map, rss=2.5, coord_sys='Carrington')
    assert isinstance(pfss_solution, pfsspy.output.Output)
    fig, ax = sm.plot_pfss(rss=2.5, pfss_solution=pfss_solution, vary=True, return_plot_object=True,
                           markers='numbers', long_sector=[290, 328], long_sector_vsw=[400, 600],
                           long_sector_color='red', reference_vsw=400.0, outfile='tmp_solarmach_pfss.png')
    assert isinstance(fig, matplotlib.figure.Figure)

    assert sm.pfss_footpoints.shape == (17,2)
    assert np.round(sm.pfss_footpoints["Earth"].iloc[0][0], 6) == 244.356758
    assert np.round(sm.pfss_footpoints["Earth"].iloc[0][1], 6) == -27.489297

    # assert hashlib.sha1(pd.util.hash_pandas_object(sm.coord_table).values).hexdigest() == '0709d8b384c5b74b792ce725c4165a2741f88e3f'  # fails - bc. of slightly different values? in other tests, np.round was used...
    # assert hashlib.sha1(pd.util.hash_pandas_object(sm.pfss_table).values).hexdigest() == ''  # fails - bc. of nan's in DF?
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


def test_sto2car_scalar():
    long, lat = 10.0, 5.0
    dtime = Time("2023-01-01T00:00:00")
    car_long, car_lat = sto2car_sun(long, lat, dtime)
    assert np.isscalar(car_long)
    assert np.isscalar(car_lat)
    assert isinstance(car_long, float)
    assert isinstance(car_lat, float)
    # Carrington longitude should be in [0, 360)
    assert 0.0 <= car_long < 360.0
    # Latitude should be preserved (within a small tolerance)
    assert np.isclose(car_lat, lat, atol=1e-8)


def test_sto2car_array():
    longs = np.array([0.0, 90.0, 180.0])
    lats = np.array([0.0, 10.0, -10.0])
    dtime = Time("2023-01-01T12:00:00")
    car_longs, car_lats = sto2car_sun(longs, lats, dtime)
    assert isinstance(car_longs, np.ndarray)
    assert isinstance(car_lats, np.ndarray)
    assert car_longs.shape == longs.shape
    assert car_lats.shape == lats.shape
    # Carrington longitude should be in [0, 360)
    assert np.all((car_longs >= 0.0) & (car_longs < 360.0))
    # Latitudes should be preserved (within a small tolerance)
    assert np.allclose(car_lats, lats, atol=1e-8)


def test_sto2car_times():
    long, lat = 45.0, -20.0
    dtime = "2022-06-15T18:30:00"
    car_long, car_lat = sto2car_sun(long, lat, dtime)
    assert 0.0 <= car_long < 360.0
    assert np.isclose(car_lat, lat, atol=1e-8)
    dtime2 = Time("2022-06-15T18:30:00")
    car_long2, car_lat2 = sto2car_sun(long, lat, dtime2)
    dtime3 = dt.datetime(2022, 6, 15, 18, 30, 0)
    car_long3, car_lat3 = sto2car_sun(long, lat, dtime3)
    assert car_long == car_long2
    assert car_long2 == car_long3


def test_sto2car_latitude_bounds():
    for lat in [-90.0, 90.0]:
        car_long, car_lat = sto2car_sun(120.0, lat, "2021-01-01T00:00:00")
        assert np.isclose(car_lat, lat, atol=1e-8)


def test_sto2car_longitude_wrap():
    long = 370.0  # > 360
    lat = 0.0
    dtime = "2023-01-01T00:00:00"
    car_long, car_lat = sto2car_sun(long, lat, dtime)
    assert 0.0 <= car_long < 360.0
    car_long2, car_lat2 = sto2car_sun(long-360, lat, dtime)
    assert car_long == car_long2


def test_car2sto_scalar():
    long, lat = 120.0, -10.0
    dtime = Time("2023-01-01T00:00:00")
    sto_long, sto_lat = car2sto_sun(long, lat, dtime)
    assert isinstance(sto_long, float) or isinstance(sto_long, np.floating)
    assert isinstance(sto_lat, float) or isinstance(sto_lat, np.floating)
    # Should be within valid longitude/latitude ranges
    assert -180.0 <= sto_long <= 360.0
    assert -90.0 <= sto_lat <= 90.0


def test_car2sto_array():
    longs = np.array([0.0, 90.0, 180.0])
    lats = np.array([0.0, 10.0, -10.0])
    dtime = Time("2023-01-01T12:00:00")
    sto_long, sto_lat = car2sto_sun(longs, lats, dtime)
    assert isinstance(sto_long, np.ndarray)
    assert isinstance(sto_lat, np.ndarray)
    assert sto_long.shape == longs.shape
    assert sto_lat.shape == lats.shape


def test_car2sto_time_string():
    long, lat = 45.0, 0.0
    dtime = "2022-06-15T18:30:00"
    sto_long, sto_lat = car2sto_sun(long, lat, dtime)
    assert isinstance(sto_long, float)
    assert isinstance(sto_lat, float)


def test_car2sto_identity():
    dtime = Time("2023-01-01T00:00:00")
    L0 = sunpy.coordinates.sun.L0(dtime, light_travel_time_correction=False, nearest_point=False, aberration_correction=False)
    sto_long, sto_lat = car2sto_sun(L0.value, 0.0, dtime)
    # Stonyhurst longitude should be close to 0. It's not exactly 0 due to additional small corrections that are applied in L0.
    assert abs(sto_long) < 1e-3


def test_car2sto_invalid_time():
    with pytest.raises(Exception):
        car2sto_sun(0.0, 0.0, "not-a-valid-time")
