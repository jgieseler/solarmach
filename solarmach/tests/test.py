import os

import astropy
import astropy.units as u
import numpy as np
from solarmach import SolarMACH, print_body_list


def test_solarmach_initialize():
    body_list = ['STEREO-A']
    vsw_list = [400]   # position-sensitive solar wind speed per body in body_list
    date = '2021-10-28 15:15:00'
    reference_long = 273                             # Carrington longitude of reference (None to omit)
    reference_lat = 9                                # Carrington latitude of reference (None to omit)

    sm = SolarMACH(date, body_list, vsw_list, reference_long, reference_lat)

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
    assert np.round(sm.coord_table["Longitudinal separation between body's mangetic footpoint and reference_long"][0], 2) == 18.96


def test_solarmach_plot():
    body_list = ['STEREO-A']
    vsw_list = [400]   # position-sensitive solar wind speed per body in body_list
    date = '2021-10-28 15:15:00'
    reference_long = 273                             # Carrington longitude of reference (None to omit)
    reference_lat = 9                                # Carrington latitude of reference (None to omit)
    reference_vsw = 400
    filename = 'test.png'

    sm = SolarMACH(date, body_list, vsw_list, reference_long, reference_lat)
    sm.plot(plot_spirals=True, plot_sun_body_line=True,
            reference_vsw=reference_vsw, transparent=False,
            show_earth_centered_coord=False, numbered_markers=True,
            outfile=filename)

    assert os.path.exists(os.getcwd()+os.sep+filename)
