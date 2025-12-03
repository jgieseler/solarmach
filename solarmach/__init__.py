from .version import version as __version__

# __all__ = []  # defines which functions, variables etc. will be loaded when running "from solarmach import *"

import copy
import os

import astropy.constants as aconst
import astropy.units as u
import datetime as dt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
import scipy.constants as const
from astropy.coordinates import SkyCoord
from matplotlib.legend_handler import HandlerPatch
from sunpy import log
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.time import parse_time

from solarmach.pfss_utilities import calculate_pfss_solution, get_field_line_coords, get_gong_map, multicolorline, sphere, spheric2cartesian, vary_flines

# pd.options.display.max_rows = None
# pd.options.display.float_format = '{:.1f}'.format
# if needed, rather use the following to have the desired display:
"""
with pd.option_context('display.float_format', '{:0.2f}'.format):
    display(df)
"""

# initialize the body dictionary
body_dict = dict.fromkeys(['Earth', 'EARTH', 'earth', 399], [399, 'Earth', 'green'])
body_dict.update(dict.fromkeys(['ACE', 'ace', 'Advanced Composition Explorer', -92], [-92, 'ACE', 'dimgrey']))
body_dict.update(dict.fromkeys(['BepiColombo', 'Bepi Colombo', 'Bepi', 'MPO', -121], [-121, 'BepiColombo', 'orange']))
body_dict.update(dict.fromkeys(['Cassini', -82], [-82, 'Cassini', 'mediumvioletred']))
body_dict.update(dict.fromkeys(['Europa Clipper', 'Clipper', -159], [-159, 'Europa Clipper', 'dimgray']))
body_dict.update(dict.fromkeys(['JUICE', 'Juice', -28], [-28, 'JUICE', 'violet']))
body_dict.update(dict.fromkeys(['Juno', 'JUNO', -61], [-61, 'Juno', 'orangered']))
body_dict.update(dict.fromkeys(['Jupiter', 599], [599, 'Jupiter', 'navy']))
body_dict.update(dict.fromkeys(['L1', 31], [31, 'SEMB-L1', 'black']))
body_dict.update(dict.fromkeys(['L2', 32], [32, 'SEMB-L2', 'salmon']))
body_dict.update(dict.fromkeys(['L4', 34], [34, 'SEMB-L4', 'lightsteelblue']))
body_dict.update(dict.fromkeys(['L5', 35], [35, 'SEMB-L5', 'olive']))
body_dict.update(dict.fromkeys(['Mars', 499], [499, 'Mars', 'maroon']))
body_dict.update(dict.fromkeys(['Mars Express', -41], [-41, 'Mars Express', 'darkorange']))
body_dict.update(dict.fromkeys(['MAVEN', 'Maven', -202], [-202, 'MAVEN', 'brown']))
body_dict.update(dict.fromkeys(['Mercury', 199], [199, 'Mercury', 'darkturquoise']))
body_dict.update(dict.fromkeys(['MESSENGER', 'Messenger', -236], [-236, 'MESSENGER', 'olivedrab']))
body_dict.update(dict.fromkeys(['PSP', 'Parker Solar Probe', 'parkersolarprobe', 'ParkerSolarProbe', -96], [-96, 'Parker Solar Probe', 'purple']))
body_dict.update(dict.fromkeys(['Pioneer10', 'Pioneer 10', -23], [-23, 'Pioneer 10', 'teal']))
body_dict.update(dict.fromkeys(['Pioneer11', 'Pioneer 11', -24], [-24, 'Pioneer 11', 'darkblue']))
body_dict.update(dict.fromkeys(['Psyche', -255], [-255, 'Psyche', '#a53f5b']))  # dark pink
body_dict.update(dict.fromkeys(['Rosetta', -226], [-226, 'Rosetta', 'blueviolet']))
body_dict.update(dict.fromkeys(['SOHO', 'soho', 'SoHO', -21], [-21, 'SOHO', 'darkgreen']))
body_dict.update(dict.fromkeys(['Solar Orbiter', 'SolO', 'solo', 'SOLO', 'solarorbiter', 'SolarOrbiter', -144], [-144, 'Solar Orbiter', 'dodgerblue']))
body_dict.update(dict.fromkeys(['STEREO B', 'STEREO-B', 'STB', 'stb', -235], [-235, 'STEREO B', 'blue']))
body_dict.update(dict.fromkeys(['STEREO A', 'STEREO-A', 'STA', 'sta', -234], [-234, 'STEREO A', 'red']))
body_dict.update(dict.fromkeys(['Ulysses', -55], [-55, 'Ulysses', 'dimgray']))
body_dict.update(dict.fromkeys(['Venus', 299], [299, 'Venus', 'darkorchid']))
body_dict.update(dict.fromkeys(['Voyager1', 'Voyager 1', -31], [-31, 'Voyager 1', 'darkred']))
body_dict.update(dict.fromkeys(['Voyager2', 'Voyager 2', -32], [-32, 'Voyager 2', 'midnightblue']))
body_dict.update(dict.fromkeys(['WIND', 'Wind', 'wind', -8], [-8, 'Wind', 'slategray']))


def print_body_list():
    """
    Prints a selection of body keys and the corresponding body names which may
    be provided to the SolarMACH class.
    Visit https://ssd.jpl.nasa.gov/horizons/app.html for a complete list of
    available bodies.
    """
    data = pd.DataFrame\
        .from_dict(body_dict, orient='index', columns=['ID', 'Body', 'Color'])\
        .drop(columns=['ID', 'Color'])\
        .drop_duplicates()
    data.index.name = 'Key'
    return data


def get_sw_speed(body, dtime, trange=1, default_vsw=400.0, silent=False):
    """
    Obtains measured solar wind bulk speed. Downloads solar wind speed
    measurements for "body" from "trange" hours before "dtime" until "trange"
    hours after "dtime", then calculates 1-hour mean values, and finally
    returns that 1-hour mean measurements that is closest to "dtime".

    Parameters
    ----------
    body : str
        Name of body, e.g., planet or spacecraft
    dtime : datetime object or datetime-compatible str
        Date and time of measurement
    trange : int of float
        Timedelta for which measurements are obtainted before and after "dtime",
        i.e. dtime +- trange (in hours). Default value 1.
    default_vsw : float
        Default solar wind bulk speed in km/s that is returned if no
        measurements can be obtained. Default value 400.0
    silent : bool, optional
        If True, suppresses most print statements. Default is False. Use at own risk!

    Returns
    -------
    float
        solar wind bulk speed in km/s
    """
    # disable unused speasy data provider before importing to speed it up
    os.environ['SPEASY_CORE_DISABLED_PROVIDERS'] = "sscweb,archive,csa"

    try:
        import speasy as spz
    except ModuleNotFoundError:
        print(f"Couldn't load required module speasy, using default_vsw={default_vsw}. Install it with 'pip install speasy' to use this functionality.")
        return default_vsw

    try:
        # standardize body name (e.g. 'PSP' => 'Parker Solar Probe')
        body = body_dict[body][1]
    except KeyError:
        pass

    amda_tree = spz.inventories.data_tree.amda
    cda_tree = spz.inventories.data_tree.cda

    dataset = dict(ACE=cda_tree.ACE.SWE.AC_K1_SWE.Vp)  # https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_K1_SWE
    dataset['SOHO'] = cda_tree.SOHO.CELIAS_PM.SOHO_CELIAS_PM_5MIN.V_p
    dataset['Parker Solar Probe'] = amda_tree.Parameters.PSP.SWEAP_SPC.psp_spc_mom.psp_spc_vp_mom_nrm
    dataset['Solar Orbiter'] = amda_tree.Parameters.SolarOrbiter.SWAPAS.L2.so_pas_momgr1.pas_momgr1_v_rtn_tot
    dataset['STEREO A'] = amda_tree.Parameters.STEREO.STEREO_A.PLASTIC.sta_l2_pla.vpbulk_sta
    dataset['STEREO B'] = amda_tree.Parameters.STEREO.STEREO_B.PLASTIC.stb_l2_pla.vpbulk_stb
    dataset['Wind'] = amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_vmag

    # obsolete with useage of "df = df.iloc[:,0].resample('1h').mean()" below
    # sw_key = dict(ACE='component_0')  # Solar Wind Bulk Speed [Vp]
    # sw_key['Parker Solar Probe'] = '|vp_mom|'  # Velocity vector magnitude
    # sw_key['SOHO'] = 'Proton V'  # Proton speed, scalar
    # sw_key['Solar Orbiter'] = '|v_rtn|'  # Velocity magnitude in RTN frame
    # sw_key['STEREO A'] = '|v|'  # Scalar magnitude of the velocity in km/s
    # sw_key['STEREO B'] = '|v|'  # Scalar magnitude of the velocity in km/s
    # sw_key['Wind'] = '|v|'  # |v|

    if body in ['Earth', 'SEMB-L1']:
        if not silent:
            print(f"Using 'ACE' measurements for '{body}'.")
        body = 'ACE'
    elif body not in dataset.keys():
        if not silent:
            print(f"Body '{body}' not supported, assuming default Vsw value of {default_vsw} km/s.")
        return default_vsw

    try:
        dtime = parse_time(dtime).datetime  # dateutil.parser.parse(dtime)
    except ValueError:  # dateutil.parser.ParserError:
        print(f"Unable to extract datetime from '{dtime}'. Assuming default Vsw value of {default_vsw} km/s.")
        return default_vsw

    try:
        if dataset[body].spz_provider() == 'amda':
            df = spz.get_data(dataset[body], dtime-dt.timedelta(hours=trange), dtime+dt.timedelta(hours=trange), output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        elif dataset[body].spz_provider() == 'cda':
            df = spz.get_data(dataset[body], dtime-dt.timedelta(hours=trange), dtime+dt.timedelta(hours=trange)).replace_fillval_by_nan().to_dataframe()
        # OLD: df = df[sw_key[body]].resample('1h').mean()
        # This approach only takes the left-most column. All dataframe contain only a single column as of now. Be careful if this changes or new datasets are added!
        df = df.iloc[:, 0].resample('1h').mean()
        # drop NaN entries:
        df.dropna(inplace=True)
        if len(df) > 0:
            idx = df.iloc[df.index.get_indexer([dtime], method='nearest')]
            if idx.values[0] >= 0.0:
                return idx.values[0]
            else:
                if not silent:
                    print(f"No Vsw data found for '{body}' on {dtime}, assuming default Vsw value of {default_vsw} km/s.")
                return default_vsw
        else:
            if not silent:
                print(f"No Vsw data found for '{body}' on {dtime}, assuming default Vsw value of {default_vsw} km/s.")
            return default_vsw
    except AttributeError:
        if not silent:
            print(f"No Vsw data found for '{body}' on {dtime}, assuming default Vsw value of {default_vsw} km/s.")
        return default_vsw


def backmapping(body_pos, reference_long=None, target_solar_radius=1, vsw=400, **kwargs):
    """
    Determine the longitudinal separation angle of a given body and a given reference longitude

    Parameters
    ----------
    body_pos : astropy.coordinates.sky_coordinate.SkyCoord
        coordinates of the body
    reference_long: float
        longitude of reference point at Sun to which we determine the longitudinal separation
    target_solar_radius: float
        target solar radius to which to be backmapped. 0 corresponds to Sun's center, 1 to 1 solar radius, and e.g. 2.5 to the source surface.
    vsw: float
        solar wind speed (in km/s) used to determine the position of the magnetic footpoint of the body. Default is 400.

    Returns
    -------
    sep: float
        longitudinal separation of body magnetic footpoint and reference longitude in degrees
    alpha: float
        backmapping angle in degrees
    """
    if 'diff_rot' in kwargs.keys():
        diff_rot = kwargs['diff_rot']
    else:
        diff_rot = True

    # pos = body_pos
    # lon = pos.lon.value
    # lat = pos.lat.value
    # dist_body = pos.radius.value

    # take into account solar differential rotation wrt. latitude
    # omega = solar_diff_rot_old(lat, diff_rot=diff_rot)
    # omega = solar_diff_rot(lat*u.deg, diff_rot=diff_rot)
    # old:
    # omega = math.radians(360. / (25.38 * 24 * 60 * 60))  # rot-angle in rad/sec, sidereal period

    # tt = dist * AU / vsw
    # alpha = math.degrees(omega * tt)
    # alpha = math.degrees(omega * (dist_body-target_solar_radius*aconst.R_sun).to(u.km).value / vsw * np.cos(np.deg2rad(lat)))
    # alpha = (backmapping_angle(dist_body*u.AU, target_solar_radius*u.R_sun, lat*u.deg, vsw*u.km/u.s, diff_rot=diff_rot)).to(u.deg).value
    alpha = (backmapping_angle(body_pos.radius, target_solar_radius*u.R_sun, body_pos.lat, vsw*u.km/u.s, diff_rot=diff_rot))

    # diff = math.degrees(target_solar_radius*aconst.R_sun.to(u.km).value * omega / vsw * np.log(radius.to(u.km).value/(target_solar_radius*aconst.R_sun).to(u.km).value))

    if reference_long is not None:
        # sep = (lon + alpha) - reference_long
        sep = ((body_pos.lon + alpha) - reference_long*u.deg).to(u.deg).value
        if sep > 180.:
            sep = sep - 360

        if sep < -180.:
            sep = 360 - abs(sep)
    else:
        sep = np.nan

    return sep, alpha.to(u.deg).value


# def backmapping_old(body_pos, reference_long=None, target_solar_radius=1, vsw=400, **kwargs):
#     """
#     Determine the longitudinal separation angle of a given body and a given reference longitude

#     Parameters
#     ----------
#     body_pos : astropy.coordinates.sky_coordinate.SkyCoord
#         coordinates of the body
#     reference_long: float
#         Longitude of reference point at Sun to which we determine the longitudinal separation
#     target_solar_radius: float
#         Target solar radius to which to be backmapped. 0 corresponds to Sun's center, 1 to 1 solar radius, and e.g. 2.5 to the source surface.
#     vsw: float
#             solar wind speed (km/s) used to determine the position of the magnetic footpoint of the body. Default is 400.

#     Returns
#     -------
#         sep: float
#             longitudinal separation of body magnetic footpoint and reference longitude in degrees
#         alpha: float
#             backmapping angle
#     """
#     if 'diff_rot' in kwargs.keys():
#         diff_rot = kwargs['diff_rot']
#     else:
#         diff_rot = True

#     pos = body_pos
#     lon = pos.lon.value
#     lat = pos.lat.value
#     # dist = pos.radius.value
#     radius = pos.radius

#     # take into account solar differential rotation wrt. latitude
#     omega = solar_diff_rot_old(lat, diff_rot=diff_rot)
#     # old:
#     # omega = math.radians(360. / (25.38 * 24 * 60 * 60))  # rot-angle in rad/sec, sidereal period

#     # tt = dist * AU / vsw
#     # alpha = math.degrees(omega * tt)
#     alpha = math.degrees(omega * (radius-target_solar_radius*aconst.R_sun).to(u.km).value / vsw * np.cos(np.deg2rad(lat)))
#     # alpha = math.degrees((-1)*backmapping_angle(target_solar_radius*aconst.R_sun.to(u.km).value, radius.to(u.km).value, lat, vsw))

#     # diff = math.degrees(target_solar_radius*aconst.R_sun.to(u.km).value * omega / vsw * np.log(radius.to(u.km).value/(target_solar_radius*aconst.R_sun).to(u.km).value))

#     if reference_long is not None:
#         sep = (lon + alpha) - reference_long
#         if sep > 180.:
#             sep = sep - 360

#         if sep < -180.:
#             sep = 360 - abs(sep)
#     else:
#         sep = np.nan

#     return sep, alpha


# def solar_diff_rot_old(lat, **kwargs):
#     """
#     Calculate solar differential rotation wrt. latitude,
#     based on rLSQ method of Beljan et al. (2017),
#     doi: 10.1051/0004-6361/201731047

#     Parameters
#     ----------
#     lat : number (int, flotat)
#         Heliographic latitude in degrees

#     Returns
#     -------
#     numpy.float64
#         Solar angular rotation in rad/sec
#     """
#     if 'diff_rot' in kwargs.keys():
#         if kwargs['diff_rot'] is False:
#             lat = 0
#     return np.radians((14.50-2.87*np.sin(np.deg2rad(lat))**2)/(24*60*60))  # (14.50-2.87*np.sin(np.deg2rad(lat))**2) defines degrees/day


def solar_diff_rot(lat, **kwargs):
    """
    Calculate the solar differential rotation rate at a given latitude.
    Based on rLSQ method of Beljan et al. (2017), doi: 10.1051/0004-6361/201731047

    Parameters
    ----------
    lat : astropy.units.Quantity
        The latitude at which to calculate the differential rotation rate, e.g.,
        "23 * astropy.units.deg". If no units are provided, it will be treated as radians!

    Returns
    -------
    astropy.units.Quantity
        Solar angular rotation in deg/sec
    """
    if 'diff_rot' in kwargs.keys():
        if kwargs['diff_rot'] is False:
            lat = 0*u.deg
    return (14.50-2.87*np.sin(lat)**2)*u.deg/(24*60*60*u.s)  # (14.50-2.87*np.sin(np.deg2rad(lat))**2) defines degrees/day


# def backmapping_angle_old(distance, r, lat, vsw, **kwargs):
#     """
#     Calculates phi(r)-phi_0 as defined in Eq. (1) of https://doi.org/10.3389/fspas.2022.1058810

#     vsw = [km/s]
#     distance = [AU]
#     r = [AU]
#     lat = [deg]
#     omega = [rad/s]

#     returns [rad]
#     """
#     if 'diff_rot' in kwargs.keys():
#         diff_rot = kwargs['diff_rot']
#     else:
#         diff_rot = True
#     #
#     omega = solar_diff_rot_old(lat, diff_rot=diff_rot)
#     # AU = const.au / 1000  # km
#     # return omega / (vsw / AU) * (distance - r) * np.cos(np.deg2rad(lat))
#     return omega / (vsw * 1000) * (distance - r)*const.au * np.cos(np.deg2rad(lat))


def backmapping_angle(distance, r, lat, vsw, **kwargs):
    """
    Calculates the backmapping angle phi(r) - phi_0.

    This function computes the backmapping angle as defined in Eq. (1) of https://doi.org/10.3389/fspas.2022.1058810.

    Parameters
    ----------
    distance : astropy.units.Quantity
        Distance with astropy units.
    r : astropy.units.Quantity
        Radial distance with astropy units.
    lat : astropy.units.Quantity
        Latitude with astropy units.
    vsw : astropy.units.Quantity
        Solar wind speed with astropy units.
    **kwargs : dict, optional
        Additional keyword arguments:
            - diff_rot : bool, optional. If True, differential rotation is considered. Default is True.

    Returns
    -------
    angle : astropy.units.Quantity
        Backmapping angle with astropy units.
    """
    if 'diff_rot' in kwargs.keys():
        diff_rot = kwargs['diff_rot']
    else:
        diff_rot = True
    #
    omega = solar_diff_rot(lat, diff_rot=diff_rot)
    # AU = const.au / 1000  # km
    # return omega / (vsw / AU) * (distance - r) * np.cos(np.deg2rad(lat))
    return (omega / vsw * (distance - r) * np.cos(lat)).to(u.rad)


class SolarMACH():
    """
    Class handling selected bodies

    Parameters
    ----------
    date: string, datetime.datetime, datetime.date, numpy.datetime64, pandas.Timestamp, tuple
        date (and optional time) of interest in a format understood by https://docs.sunpy.org/en/stable/how_to/parse_time.html
    body_list: list
        list of body keys to be used. Keys can be string of int.
    vsw_list: list, optional
        list of solar wind bulk speeds in km/s at the position of the different bodies. Must have the same length as body_list.
        If empty list, obtaining actual measurements is tried. If this is not successful, a default value defined by default_vsw is used.
    default_vsw: int or float, optional
        Solar wind bulk speed in km/s to be used if vsw_list is not defined and no vsw measurements could be obtained. By default 400.0.
    coord_sys: string, optional
        Defines the coordinate system used: 'Carrington' (default) or 'Stonyhurst'. Note that the Carrington longitude is given for an observer at the Sun, not at Earth or any other body. When comparing with observations at different locations, those might need to be corrected for the light travel time.
    reference_long: float, optional
        Longitute of reference position at the Sun
    reference_lat: float, optional
        Latitude of referene position at the Sun
    silent : bool, optional
        If True, suppresses most print statements. Default is False. Use at own risk!
    """

    def __init__(self, date, body_list, vsw_list=[], reference_long=None, reference_lat=None, coord_sys='Carrington', default_vsw=400.0, silent=False, **kwargs):
        if 'diff_rot' in kwargs.keys():
            self.diff_rot = kwargs['diff_rot']
        else:
            self.diff_rot = True
        if 'target_solar_radius' in kwargs.keys():
            self.target_solar_radius = kwargs['target_solar_radius']
        else:
            self.target_solar_radius = 1

        # get initial sunpy logging level and disable unnecessary logging
        initial_log_level = log.getEffectiveLevel()
        log.setLevel('WARNING')

        body_list = list(dict.fromkeys(body_list))
        bodies = copy.deepcopy(body_dict)

        if coord_sys.lower().startswith('car'):
            coord_sys = 'Carrington'
        if coord_sys.lower().startswith('sto') or coord_sys.lower() == "earth":
            coord_sys = 'Stonyhurst'

        # parse input date & time
        self.date = parse_time(date)

        self.reference_long = reference_long
        self.reference_lat = reference_lat
        self.coord_sys = coord_sys

        try:
            pos_E = get_horizons_coord(399, self.date, None)  # (lon, lat, radius) in (deg, deg, AU)
        except (ValueError, RuntimeError):
            if not silent:
                print('')
                print('!!! No ephemeris found for Earth for date {self.date} - there probably is a problem with JPL HORIZONS.')
        if coord_sys=='Carrington':
            self.pos_E = pos_E.transform_to(frames.HeliographicCarrington(observer='Sun'))
        elif coord_sys=='Stonyhurst':
            self.pos_E = pos_E

        # standardize "undefined" vsw_list for further usage:
        if type(vsw_list) is type(None) or vsw_list is False:
            vsw_list=[]

        # make deep copy of vsw_list bc. otherwise it doesn't get reset in a new init:
        vsw_list2 = copy.deepcopy(vsw_list)

        if len(vsw_list2) == 0:
            if not silent:
                print('No solar wind speeds defined, trying to obtain measurements...')
            for body in body_list:
                vsw_list2.append(get_sw_speed(body=body, dtime=date, default_vsw=default_vsw, silent=silent))
            # vsw_list = np.zeros(len(body_list)) + 400

        random_cols = ['forestgreen', 'mediumblue', 'm', 'saddlebrown', 'tomato', 'olive', 'steelblue', 'darkmagenta',
                       'c', 'darkslategray', 'yellow', 'darkolivegreen']
        body_lon_list = []
        body_lat_list = []
        body_dist_list = []
        longsep_E_list = []
        latsep_E_list = []
        body_vsw_list = []
        footp_long_list = []
        longsep_list = []
        latsep_list = []
        footp_longsep_list = []

        for i, body in enumerate(body_list.copy()):
            if body in bodies:
                body_id = bodies[body][0]
                body_lab = bodies[body][1]
                body_color = bodies[body][2]

            else:
                body_id = body
                body_lab = str(body)
                body_color = random_cols[i]
                bodies.update(dict.fromkeys([body_id], [body_id, body_lab, body_color]))

            try:
                pos = get_horizons_coord(body_id, date, None)  # (lon, lat, radius) in (deg, deg, AU)
                if coord_sys=='Carrington':
                    pos = pos.transform_to(frames.HeliographicCarrington(observer='Sun'))
                bodies[body_id].append(pos)
                bodies[body_id].append(vsw_list2[i])

                longsep_E = pos.lon.value - self.pos_E.lon.value
                if longsep_E > 180:
                    longsep_E = longsep_E - 360.
                latsep_E = pos.lat.value - self.pos_E.lat.value

                body_lon_list.append(pos.lon.value)
                body_lat_list.append(pos.lat.value)
                body_dist_list.append(pos.radius.value)
                longsep_E_list.append(longsep_E)
                latsep_E_list.append(latsep_E)

                body_vsw_list.append(vsw_list2[i])

                sep, alpha = backmapping(pos, reference_long, target_solar_radius=self.target_solar_radius, vsw=vsw_list2[i], diff_rot=self.diff_rot)
                bodies[body_id].append(sep)

                body_footp_long = pos.lon.value + alpha
                if body_footp_long > 360:
                    body_footp_long = body_footp_long - 360
                footp_long_list.append(body_footp_long)

                if self.reference_long is not None:
                    bodies[body_id].append(sep)
                    long_sep = pos.lon.value - self.reference_long
                    if long_sep > 180:
                        long_sep = long_sep - 360.

                    longsep_list.append(long_sep)
                    footp_longsep_list.append(sep)

                if self.reference_lat is not None:
                    lat_sep = pos.lat.value - self.reference_lat
                    latsep_list.append(lat_sep)
            except (ValueError, RuntimeError):
                if not silent:
                    print('')
                    print('!!! No ephemeris for target "' + str(body) + '" for date ' + str(self.date))
                body_list.remove(body)

        body_dict_short = {sel_key: bodies[sel_key] for sel_key in body_list}
        self.body_dict = body_dict_short
        self.max_dist = np.max(body_dist_list)  # spherical radius
        self.max_dist_lat = body_lat_list[np.argmax(body_dist_list)]  # latitude connected to max spherical radius
        self.coord_table = pd.DataFrame(
            {'Spacecraft/Body': list(self.body_dict.keys()), f'{coord_sys} longitude (°)': body_lon_list,
             f'{coord_sys} latitude (°)': body_lat_list, 'Heliocentric distance (AU)': body_dist_list,
             "Longitudinal separation to Earth's longitude": longsep_E_list,
             "Latitudinal separation to Earth's latitude": latsep_E_list, 'Vsw': body_vsw_list,
             f'Magnetic footpoint longitude ({coord_sys})': footp_long_list})

        self.pfss_table = pd.DataFrame(
            {"Spacecraft/Body": list(self.body_dict.keys()),
             f"{coord_sys} longitude (°)": body_lon_list,
             f"{coord_sys} latitude (°)": body_lat_list,
             "Heliocentric_distance (R_Sun)": np.array(body_dist_list) * u.au.to(u.solRad),  # Quick conversion of AU -> Solar radii
             "Vsw": body_vsw_list
             }
        )

        if self.reference_long is not None:
            self.coord_table['Longitudinal separation between body and reference_long'] = longsep_list
            self.coord_table[
                "Longitudinal separation between body's magnetic footpoint and reference_long"] = footp_longsep_list
        if self.reference_lat is not None:
            self.coord_table['Latitudinal separation between body and reference_lat'] = latsep_list
        
        if self.reference_long is not None or self.reference_lat is not None:
            self.pfss_table = pd.concat([self.pfss_table, 
                                         pd.DataFrame({"Spacecraft/Body": ["Reference Point"],
                                                       f"{coord_sys} longitude (°)": [self.reference_long],
                                                       f"{coord_sys} latitude (°)": [self.reference_lat], 
                                                       "Heliocentric_distance (R_Sun)": [1],
                                                       "Vsw": [np.nan]})],
                                                       ignore_index=True)

        # Does this still have a use?
        pass
        self.coord_table.style.set_properties(**{'text-align': 'left'})

        # reset sunpy log level to initial state
        log.setLevel(initial_log_level)

    def plot(self, plot_spirals=True,
             plot_sun_body_line=False,
             show_earth_centered_coord=False,
             reference_vsw=400,
             transparent=False,
             markers=False,
             return_plot_object=False,
             fix_earth=True,
             long_offset=270,
             outfile='',
             figsize=(12, 8),
             dpi=200,
             long_sector=None,
             long_sector_vsw=None,
             long_sector_color='red',
             long_sector_alpha=0.5,
             background_spirals=None,
             numbered_markers=False,  # kept only for backward compatibility
             test_plotly=False,
             test_plotly_template='plotly',
             # x_offset=0.0,  # TODO: remove this option.
             # y_offset=0.0, # TODO: remove this option.
             test_plotly_legend=(1.0, 1.0),
             test_plotly_logo=(1.0, 0.0)):
        """
        Make a polar plot showing the Sun in the center (view from North) and the positions of the selected bodies

        Parameters
        ----------
        plot_spirals : bool, optional
            if True, the magnetic field lines connecting the bodies with the Sun are plotted
        plot_sun_body_line : bool, optional
            if True, straight lines connecting the bodies with the Sun are plotted
        show_earth_centered_coord : bool, optional
            Deprecated! With the introduction of coord_sys in class SolarMACH() this function is redundant and not functional any more!
        reference_vsw : int, optional
            if defined, defines solar wind speed for reference. if not defined, 400 km/s is used
        transparent : bool, optional
            if True, output image has transparent background
        markers : bool or string, optional
            if defined, body markers contain 'numbers' or 'letters' for better identification. If False (default), only geometric markers are used.
        return_plot_object : bool, optional
            if True, figure and axis object of matplotib are returned, allowing further adjustments to the figure
        fix_earth : bool, optional
            if True (default), Earth is always at the defined long_offset position (by default "6 o'clock", i.e., 270°). If False, the plot is oriented with 0° at the position defined with long_offset.
        long_offset : int or float, optional
            longitudinal offset for polar plot; defines for fix_earth=True (default) where Earth's longitude is (by default 270, i.e., at "6 o'clock"). For fix_earth=False it defines where 0° is located.
        outfile : string, optional
            if provided, the plot is saved with outfile as filename. supports png and pdf format.
        long_sector : list of 2 numbers, optional
            Start and stop longitude of a shaded area; e.g. [350, 20] to get a cone from 350 to 20 degree longitude (for long_sector_vsw=None).
        long_sector_vsw : list of 2 numbers, optional
            Solar wind speed used to calculate Parker spirals (at start and stop longitude provided by long_sector) between which a reference cone should be drawn; e.g. [400, 400] to assume for both edges of the fill area a Parker spiral produced by solar wind speeds of 400 km/s. If None, instead of Parker spirals straight lines are used, i.e. a simple cone wil be plotted. By default None.
        long_sector_color : string, optional
            String defining the matplotlib color used for the shading defined by long_sector. By default 'red'.
        long_sector_alpha : float, optional
            Float between 0.0 and 1.0, defining the matplotlib alpha used for the shading defined by long_sector. By default 0.5.W
        background_spirals : list of 2 numbers (and 3 optional strings), optional
            If defined, plot evenly distributed Parker spirals over 360°. background_spirals[0] defines the number of spirals, background_spirals[1] the solar wind speed in km/s used for their calculation. background_spirals[2], background_spirals[3], and background_spirals[4] optionally change the plotting line style, color, and alpha setting, respectively (default values ':', 'grey', and 0.1). Full example that plots 12 spirals (i.e., every 30°) using a solar wind speed of 400 km/s with solid red lines with alpha=0.2 is "background_spirals=[12, 400, '-', 'red', 0.2]"
        numbered_markers : bool, deprecated
            Deprecated option, use markers='numbers' instead!

        Returns
        -------
        matplotlib figure and axes or None
            Returns the matplotlib figure and axes if return_plot_object=True (by default set to False), else nothing.
        """
        hide_logo = False  # optional later keyword to hide logo on figure
        # AU = const.au / 1000  # km

        # save inital rcParams and update some of them:
        initial_rcparams = plt.rcParams.copy()
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['font.size'] = 15
        plt.rcParams['agg.path.chunksize'] = 20000

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize, dpi=dpi)
        self.ax = ax

        # build array of values for radius (in spherical coordinates!) given in AU!
        r_array = np.arange(0.007, (self.max_dist+0.1)/np.cos(np.deg2rad(self.max_dist_lat)) + 3.0, 0.001)
        # take into account solar differential rotation wrt. latitude. Thus move calculation of omega to the per-body section below
        # omega = np.radians(360. / (25.38 * 24 * 60 * 60))  # solar rot-angle in rad/sec, sidereal period

        E_long = self.pos_E.lon.value

        # catch old syntax
        if numbered_markers is True and not markers:
            markers='numbers'
            print('')
            print("WARNING: The usage of numbered_markers is deprecated and will be discontinued in the future! Use markers='numbers' instead.")
            print('')

        if markers:
            if markers.lower() in ['n', 'number']:
                markers='numbers'
            if markers.lower() in ['l', 'letter']:
                markers='letters'

        if test_plotly:
            import plotly.graph_objects as go
            pfig = go.Figure()

        rlabel_pos = E_long + 120
        ax.set_rlabel_position(rlabel_pos)
        if fix_earth:
            ax.set_theta_offset(np.deg2rad(long_offset - E_long))
        elif not fix_earth:
            ax.set_theta_offset(np.deg2rad(long_offset))
        ax.yaxis.get_major_locator().base.set_params(nbins=4)
        circle = plt.Circle((0., 0.),
                            self.max_dist + 0.29,
                            transform=ax.transData._b,
                            edgecolor="k",
                            facecolor=None,
                            fill=False, lw=3,
                            zorder=2.5)
        ax.add_patch(circle)

        # deactivate plotting of the outer circle that limits the plotting area bc. it sometimes vanishes. 
        # it's "replaced" by the plt.Circle above
        ax.spines['polar'].set_linewidth(0)

        # r-grid with different resolution depending on maximum distance body
        if self.max_dist < 2:
            ax.set_rgrids(np.arange(0, self.max_dist + 0.29, 0.5)[1:], angle=rlabel_pos)
        elif self.max_dist < 10:
            ax.set_rgrids(np.arange(0, self.max_dist + 0.29, 1.0)[1:], angle=rlabel_pos)

        # manually plot r-grid lines with different resolution depending on maximum distance body
        grid_radii = []
        if self.max_dist < 2:
            grid_radii = np.arange(0, self.max_dist + 0.29, 0.5)[1:]
        elif self.max_dist < 10:
            grid_radii = np.arange(0, self.max_dist + 0.29, 1.0)[1:]
        if len(grid_radii) > 0:
            grid_lines, grid_labels = ax.set_rgrids(grid_radii, angle=rlabel_pos)
            # overplot r-grid circles manually because there sometimes missing
            for grid_radius in grid_radii:
                ax.plot(np.linspace(0, 2*np.pi, 180), [grid_radius]*180,
                        color=grid_lines[0].get_color(),
                        lw=grid_lines[0].get_lw(),
                        ls=grid_lines[0].get_ls(),
                        zorder=grid_lines[0].get_zorder(),
                        )

        for i, body_id in enumerate(self.body_dict):
            body_lab = self.body_dict[body_id][1]
            body_color = self.body_dict[body_id][2]
            body_vsw = self.body_dict[body_id][4]
            body_pos = self.body_dict[body_id][3]

            pos = body_pos
            dist_body = pos.radius.value

            body_long = pos.lon.value
            body_lat = pos.lat.value

            # take into account solar differential rotation wrt. latitude
            # omega = solar_diff_rot_old(body_lat, diff_rot=self.diff_rot)
            # old:
            # omega = np.radians(360. / (25.38 * 24 * 60 * 60))  # solar rot-angle in rad/sec, sidereal period

            # plot body positions
            if markers:
                ax.plot(np.deg2rad(body_long), dist_body*np.cos(np.deg2rad(body_lat)), 'o', ms=15, color=body_color, label=body_lab)
                if markers.lower()=='letters':
                    if body_id[:6] == 'STEREO':
                        mark = str(body_id[-1])
                    elif body_id == 'Europa Clipper':
                        mark = 'C'
                    else:
                        mark = str(body_id[0])
                if markers.lower()=='numbers':
                    mark = i+1
                ax.annotate(mark, xy=(np.deg2rad(body_long), dist_body*np.cos(np.deg2rad(body_lat))), color='white',
                            fontsize="small", weight='heavy',
                            horizontalalignment='center',
                            verticalalignment='center')
            else:
                ax.plot(np.deg2rad(body_long), dist_body*np.cos(np.deg2rad(body_lat)), 's', color=body_color, label=body_lab)

            if plot_sun_body_line:
                # ax.plot(alpha_ref[0], 0.01, 0)
                ax.plot([np.deg2rad(body_long), np.deg2rad(body_long)], [0.01, dist_body*np.cos(np.deg2rad(body_lat))], ':', color=body_color)
            # plot the spirals
            if plot_spirals:
                # tt = dist_body * AU / body_vsw
                # alpha = np.degrees(omega * tt)
                # alpha_body = np.deg2rad(body_long) + omega / (body_vsw / AU) * (dist_body - r_array)
                # alpha_body = np.deg2rad(body_long) + omega / (body_vsw / AU) * (dist_body - r_array) * np.cos(np.deg2rad(body_lat))
                # alpha_body = np.deg2rad(body_long) + backmapping_angle2(dist_body, r_array, body_lat, body_vsw, diff_rot=self.diff_rot)
                alpha_body = (body_long*u.deg + backmapping_angle(dist_body*u.AU, r_array*u.AU, body_lat*u.deg, body_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value
                ax.plot(alpha_body, r_array * np.cos(np.deg2rad(body_lat)), color=body_color)

            if test_plotly:
                if plot_spirals:
                    pfig.add_trace(go.Scatterpolar(
                        r=r_array * np.cos(np.deg2rad(body_lat)),
                        theta=alpha_body,
                        mode='lines',
                        name=f'{body_id} magnetic field line',
                        showlegend=False,
                        line=dict(color=body_dict[body_id][2]),
                        thetaunit="radians"))

                if plot_sun_body_line:
                    pfig.add_trace(go.Scatterpolar(
                        r=[0.01, dist_body*np.cos(np.deg2rad(body_lat))],
                        theta=[np.deg2rad(body_long), np.deg2rad(body_long)],
                        mode='lines',
                        name=f'{body_id} direct line',
                        showlegend=False,
                        line=dict(color=body_dict[body_id][2], dash='dot'),
                        thetaunit="radians"))

                if markers:
                    if markers.lower()=='letters' or markers.lower()=='numbers':
                        str_number = f'<b>{mark}</b>'
                else:
                    str_number = None

                pfig.add_trace(go.Scatterpolar(
                    r=[dist_body*np.cos(np.deg2rad(body_lat))],
                    theta=[np.deg2rad(body_long)],
                    mode='markers+text',
                    name=body_id,
                    marker=dict(size=16, color=body_dict[body_id][2]),
                    # text=[f'<b>{body_id}</b>'],
                    # textposition="top center",
                    text=[str_number],
                    textfont=dict(color="white", size=14),
                    textposition="middle center",
                    thetaunit="radians"))

        if self.reference_long is not None:
            delta_ref = self.reference_long
            if delta_ref < 0.:
                delta_ref = delta_ref + 360.
            if self.reference_lat is None:
                ref_lat = 0.
            else:
                ref_lat = self.reference_lat
            # take into account solar differential rotation wrt. latitude
            # omega_ref = solar_diff_rot_old(ref_lat, diff_rot=self.diff_rot)

            # old eq. for alpha_ref contained redundant dist_e variable:
            # alpha_ref = np.deg2rad(delta_ref) + omega_ref / (reference_vsw / AU) * (dist_e / AU - r_array) - (omega_ref / (reference_vsw / AU) * (dist_e / AU))
            # alpha_ref = np.deg2rad(delta_ref) + omega_ref / (reference_vsw / AU) * (aconst.R_sun.to(u.AU).value - r_array)
            # alpha_ref = np.deg2rad(delta_ref) + omega_ref / (reference_vsw / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(ref_lat))
            alpha_ref = (delta_ref*u.deg + backmapping_angle(self.target_solar_radius*aconst.R_sun, r_array*u.AU, ref_lat*u.deg, reference_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

            # old arrow style:
            # arrow_dist = min([self.max_dist + 0.1, 2.])
            # ref_arr = plt.arrow(alpha_ref[0], 0.01, 0, arrow_dist, head_width=0.12, head_length=0.11, edgecolor='black',
            #                     facecolor='black', lw=2, zorder=5, overhang=0.2)
            arrow_dist = min([self.max_dist/3.2, 2.])
            # ref_arr = plt.arrow(alpha_ref[0], 0.01, 0, arrow_dist, head_width=0.2, head_length=0.07, edgecolor='black',
            #                     facecolor='black', lw=1.8, zorder=5, overhang=0.2)
            ref_arr = plt.arrow(np.deg2rad(delta_ref), 0.01, 0, arrow_dist, head_width=0.2, head_length=0.07, edgecolor='black',
                                facecolor='black', lw=1.8, zorder=5, overhang=0.2)
            if test_plotly:
                if test_plotly_template=="plotly_dark":
                    reference_color = "white"
                else:
                    reference_color = "black"
                pfig.add_trace(go.Scatterpolar(
                    r=[0.0, arrow_dist],
                    theta=[np.deg2rad(delta_ref), np.deg2rad(delta_ref)],
                    mode='lines+markers',
                    marker=dict(symbol="arrow", size=15, angleref="previous", color=reference_color),
                    name='reference long.',
                    showlegend=True,
                    line=dict(color=reference_color),
                    thetaunit="radians"))
            if plot_spirals:
                ax.plot(alpha_ref, r_array * np.cos(np.deg2rad(ref_lat)), '--k', label=f'field line connecting to\nref. long. (vsw={reference_vsw} km/s)')
                if test_plotly:
                    pfig.add_trace(go.Scatterpolar(
                        r=r_array * np.cos(np.deg2rad(ref_lat)),
                        theta=alpha_ref,
                        mode='lines',
                        name=f'field line connecting to<br>ref. long. (vsw={reference_vsw} km/s)',
                        showlegend=True,
                        line=dict(color=reference_color, dash="dash"),
                        thetaunit="radians"))

        if test_plotly:
            if markers:
                if markers.lower()=='letters' or markers.lower()=='numbers':
                    for i, body_id in enumerate(self.body_dict):
                        if self.reference_long is not None:
                            x_offset_ref = -0.035  # 0.004
                            y_offset_ref = 0.081
                            y_offset_per_i = -0.051
                        else:
                            x_offset_ref = 0.0
                            y_offset_ref = 0.0
                            y_offset_per_i = -0.0475
                        # These offset numbers probably need to be updated; it seems the markers are now too much in the upper left direction.
                        # They're not visible anymore for test_plotly_legend=[1.0, 1.0], so test for test_plotly_legend=[0.5, 0.5].
                        # Note that the offset effect changes with the size of the plotly figure (i.e., when resizing the browser window)!
                        x_offset = -0.11  # 0.05
                        y_offset = 0.124  # -0.0064

                        if markers.lower()=='letters':
                            if body_id[:6] == 'STEREO':
                                mark = str(body_id[-1])
                            elif body_id == 'Europa Clipper':
                                mark = 'C'
                            else:
                                mark = str(body_id[0])
                        if markers.lower()=='numbers':
                            mark = i+1

                        pfig.add_annotation(text=f'<b>{mark}</b>', xref="paper", yref="paper", xanchor="center", yanchor="top",
                                            x=test_plotly_legend[0]+x_offset+x_offset_ref, y=test_plotly_legend[1]+y_offset+y_offset_ref+y_offset_per_i*i,
                                            showarrow=False, font=dict(color="black", size=14))

            pfig.add_annotation(text='Solar-MACH', xref="paper", yref="paper",  # xanchor="center", yanchor="middle",
                                x=test_plotly_logo[0], y=test_plotly_logo[1]+0.05,
                                showarrow=False, font=dict(color="black", size=28, family='DejaVu Serif'), align="right")
            pfig.add_annotation(text='https://solar-mach.github.io', xref="paper", yref="paper",  # xanchor="center", yanchor="middle",
                                x=test_plotly_logo[0], y=test_plotly_logo[1],
                                showarrow=False, font=dict(color="black", size=18, family='DejaVu Serif'), align="right")

            # for template in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
            if not test_plotly_template:
                test_plotly_template = "plotly"
            polar_rotation = (long_offset - E_long)
            pfig.update_layout(template=test_plotly_template,
                               polar=dict(radialaxis_range=[0, self.max_dist + 0.3], angularaxis_rotation=polar_rotation),
                               modebar_add=["v1hovermode"],
                               modebar_remove=["select2d", "lasso2d"],
                               margin=dict(l=100, r=100, b=0, t=50),
                               # paper_bgcolor="LightSteelBlue",
                               legend=dict(yanchor="middle", y=test_plotly_legend[1], xanchor="center", x=test_plotly_legend[0]))
            # fig.show()
            # if using streamlit, send plot to streamlit output, else call plt.show()
            if _isstreamlit():
                import streamlit as st
                # import streamlit.components.v1 as components
                # st.plotly_chart(pfig, theme="streamlit")
                # components.html(pfig.to_html(include_mathjax='cdn'), height=500)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            else:
                pfig.show()

        if long_sector is not None:
            if type(long_sector) is list and np.array(long_sector).ndim==1:
                long_sector = [long_sector]
                long_sector_vsw = [long_sector_vsw]
                long_sector_color = [long_sector_color]
                long_sector_alpha = [long_sector_alpha]
            else:
                print("Non-standard 'long_sector'. It should be a 2-element list defining the start and end longitude of the cone in degrees; e.g. 'long_sector=[15,45]'. 'long_sector_XXX' options have to follow accordingly.")
            for i in range(len(long_sector)):
                t_long_sector = long_sector[i]
                t_long_sector_vsw = long_sector_vsw[i]
                t_long_sector_color = long_sector_color[i]
                t_long_sector_alpha = long_sector_alpha[i]
                delta_ref1 = t_long_sector[0]
                if delta_ref1 < 0.:
                    delta_ref1 = delta_ref1 + 360.
                delta_ref2 = t_long_sector[1]
                if delta_ref2 < 0.:
                    delta_ref2 = delta_ref2 + 360.

                # Check that we are considering the same rotation
                if delta_ref2 < delta_ref1:
                    delta_ref2 += 360

                long_sector_lat = [0, 0]  # maybe later add option to have different latitudes, so that the long_sector plane is out of the ecliptic
                # take into account solar differential rotation wrt. latitude
                # omega_ref1 = solar_diff_rot_old(long_sector_lat[0], diff_rot=self.diff_rot)
                # omega_ref2 = solar_diff_rot_old(long_sector_lat[1], diff_rot=self.diff_rot)

                # Build an r_array for the second spiral for while loop to iterate forwards
                r_array2 = np.copy(r_array)

                if t_long_sector_vsw is not None:
                    # Calculate the first spiral's angles along r
                    # alpha_ref1 = np.deg2rad(delta_ref1) + omega_ref1 / (t_long_sector_vsw[0] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(long_sector_lat[0]))
                    # alpha_ref2 = np.deg2rad(delta_ref2) + omega_ref2 / (t_long_sector_vsw[1] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array2) * np.cos(np.deg2rad(long_sector_lat[1]))
                    alpha_ref1 = (delta_ref1*u.deg + backmapping_angle(self.target_solar_radius*aconst.R_sun, r_array*u.AU, long_sector_lat[0]*u.deg, t_long_sector_vsw[0]*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value
                    alpha_ref2 = (delta_ref2*u.deg + backmapping_angle(self.target_solar_radius*aconst.R_sun, r_array2*u.AU, long_sector_lat[1]*u.deg, t_long_sector_vsw[1]*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

                    # # Save the last angle as a starting point for reference for the while loop
                    # alpha_init = alpha_ref2[-1]

                    # Check that reference angle of the first loop is ahead
                    if alpha_ref1[-1] > alpha_ref2[-1]:
                        alpha_ref1_comp = alpha_ref1[-1] - 2*np.pi
                    else:
                        alpha_ref1_comp = alpha_ref1[-1]

                    # While the second spiral is behind the first spiral in angle, extend the second spiral
                    while alpha_ref2[-1] > alpha_ref1_comp:
                        r_array2 = np.append(r_array2, r_array2[-1] + 0.1)
                        # alpha_ref2 = np.append(alpha_ref2, np.deg2rad(delta_ref2) + omega_ref2 / (t_long_sector_vsw[1] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array2[-1]) * np.cos(np.deg2rad(long_sector_lat[1])))
                        alpha_ref2 = np.append(alpha_ref2, (delta_ref2*u.deg + backmapping_angle(self.target_solar_radius*aconst.R_sun, r_array2[-1]*u.AU, long_sector_lat[1]*u.deg, t_long_sector_vsw[1]*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value)

                    # Interpolate the first spiral's angles to the coarser second spiral's angles (outside the plot)
                    alpha_ref1 = np.interp(r_array2, r_array, alpha_ref1)

                else:
                    # if no solar wind speeds for Parker spirals are provided, use straight lines:
                    alpha_ref1 = np.array([np.deg2rad(delta_ref1)] * len(r_array))
                    alpha_ref2 = np.array([np.deg2rad(delta_ref2)] * len(r_array))

                c1 = plt.polar(alpha_ref1, r_array2 * np.cos(np.deg2rad(long_sector_lat[0])), lw=0, color=t_long_sector_color, alpha=t_long_sector_alpha)[0]
                x1 = c1.get_xdata()
                y1 = c1.get_ydata()
                c2 = plt.polar(alpha_ref2, r_array2 * np.cos(np.deg2rad(long_sector_lat[1])), lw=0, color=t_long_sector_color, alpha=t_long_sector_alpha)[0]
                x2 = c2.get_xdata()
                # y2 = c2.get_ydata()

                # Check that plotted are is between the two spirals, and do not fill after potential crossing
                clause1 = x1 < x2
                clause2 = alpha_ref1[clause1] < alpha_ref2[clause1]

                # Take only the points that fill the above clauses
                y1_fill = y1[clause1][clause2]
                x1_fill = x1[clause1][clause2]
                x2_fill = x2[clause1][clause2]

                plt.fill_betweenx(y1_fill, x1_fill, x2_fill, lw=0, color=t_long_sector_color, alpha=t_long_sector_alpha)

        if background_spirals is not None:
            if type(background_spirals) is list and len(background_spirals)>=2:
                # maybe later add option to have a non-zero latitude, so that the field lines are out of the ecliptic
                background_spirals_lat = 0
                # take into account solar differential rotation wrt. latitude
                # omega_ref = solar_diff_rot_old(background_spirals_lat, diff_rot=self.diff_rot)

                if len(background_spirals)>=3:
                    background_spirals_ls = background_spirals[2]
                else:
                    background_spirals_ls = ':'

                if len(background_spirals)>=4:
                    background_spirals_c = background_spirals[3]
                else:
                    background_spirals_c = 'grey'

                if len(background_spirals)>=5:
                    background_spirals_alpha = background_spirals[4]
                else:
                    background_spirals_alpha = 0.5

                for l in np.arange(0, 360, 360/background_spirals[0]):
                    # alpha_ref = np.deg2rad(l) + omega_ref / (background_spirals[1] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(background_spirals_lat))
                    alpha_ref = (l*u.deg + backmapping_angle(self.target_solar_radius*aconst.R_sun, r_array*u.AU, background_spirals_lat*u.deg, background_spirals[1]*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value
                    ax.plot(alpha_ref, r_array * np.cos(np.deg2rad(background_spirals_lat)), ls=background_spirals_ls, c=background_spirals_c, alpha=background_spirals_alpha)
            else:
                print("Ill-defined 'background_spirals'. It should be a list with at least 2 elements defining the number of field lines and the solar wind speed used for them in km/s; e.g. 'background_spirals=[10, 400]'")

        def legend_arrow(width, height, **_):
            return mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True,
                                       head_width=0.75 * height)

        # leg1 = ax.legend(loc=(1.2, 0.7), fontsize=13)
        leg1 = ax.legend(bbox_to_anchor=(1.1, 1.05), loc="upper left", fontsize=13, numpoints=1,
                         handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=legend_arrow), })

        if markers:
            offset = matplotlib.text.OffsetFrom(leg1, (0.0, 1.0))
            for i, body_id in enumerate(self.body_dict):
                if outfile.split('.')[-1] == 'pdf':
                    yoffset = i*19.25  # 18.5 19.5
                else:
                    yoffset = i*18.7  # 18.5 19.5
                if markers.lower()=='letters':
                    if body_id[:6] == 'STEREO':
                        mark = str(body_id[-1])
                    elif body_id == 'Europa Clipper':
                        mark = 'C'
                    else:
                        mark = str(body_id[0])
                if markers.lower()=='numbers':
                    mark = i+1
                ax.annotate(mark, xy=(1, 1), xytext=(18.3, -11-yoffset), color='white',
                            fontsize="small", weight='heavy', textcoords=offset,
                            horizontalalignment='center',
                            verticalalignment='center', zorder=100)

        if self.reference_long is not None:
            # leg2 = ax.legend([ref_arr], ['reference long.'], loc=(1.2, 0.6),
            #                  handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=legend_arrow), },
            #                  fontsize=13)
            # ax.add_artist(leg1)

            def add_arrow_to_legend(legend):
                ax = legend.axes

                handles, labels = ax.get_legend_handles_labels()
                handles.append(ref_arr)
                labels.append('reference long.')

                legend._legend_box = None
                legend._init_legend_box(handles, labels)
                legend._set_loc(legend._loc)
                legend.set_title(legend.get_title().get_text())

            add_arrow_to_legend(leg1)

        # replace 'SEMB-L1' in legend with 'L1' if present
        for text in leg1.get_texts():
            if text.get_text()[:6] == 'SEMB-L':
                text.set_text(text.get_text()[-2:])

        # for Stonyhurst, define the longitude from -180 to 180 (instead of 0 to 360)
        if self.coord_sys=='Stonyhurst':
            ax.set_xticks(np.pi/180. * np.linspace(180, -180, 8, endpoint=False))
            ax.set_thetalim(-np.pi, np.pi)

        ax.set_rmax(self.max_dist + 0.3)
        ax.set_rmin(0.01)

        ax.set_title(str(self.date.to_value('iso', subfmt='date_hm')) + ' (UTC)\n', pad=30)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        if show_earth_centered_coord:
            print("The option 'show_earth_centered_coord' is deprecated! Please initialize SolarMACH with coord_sys='Stonyhurst' to get an Earth-centered coordinate system.")
            # pos1 = ax.get_position()  # get the original position of the polar plot
            # offset = 0.12
            # pos2 = [pos1.x0 - offset / 2, pos1.y0 - offset / 2, pos1.width + offset, pos1.height + offset]
            # ax2 = self._polar_twin(ax, E_long, pos2, long_offset)

        ax.tick_params(axis='x', pad=10)

        if not hide_logo:
            ax.text(0.83, 0.16, 'Solar-MACH',
                    fontfamily='DejaVu Serif', fontsize=23,
                    ha='right', va='bottom', transform=fig.transFigure)
            ax.text(0.83, 0.12, 'https://solar-mach.github.io',
                    fontfamily='DejaVu Sans', fontsize=13,
                    ha='right', va='bottom', transform=fig.transFigure)

        if transparent:
            fig.patch.set_alpha(0.0)

        if outfile != '':
            plt.savefig(outfile, bbox_inches="tight")
        # st.pyplot(fig, dpi=200)

        # restore initial rcParams that have been saved at the beginning of this function:
        plt.rcParams.update(initial_rcparams)

        # don't display figure if saving as pdf file
        if outfile.split('.')[-1] != 'pdf':
            # if using streamlit, send plot to streamlit output, else call plt.show()
            if _isstreamlit():
                import streamlit as st
                st.pyplot(fig, width="content")  # , dpi=200)
            else:
                plt.show()

        if return_plot_object:
            # TODO: not really straightforward; change in future
            if not test_plotly:
                return fig, ax
            else:
                return pfig

    # def _polar_twin(self, ax, E_long, position, long_offset):
    #     """
    #     add an additional axes which is needed to plot additional longitudinal tickmarks with Earth at longitude 0
    #     not used any more!
    #     """
    #     ax2 = ax.figure.add_axes(position, projection='polar',
    #                              label='twin', frameon=False,
    #                              theta_direction=ax.get_theta_direction(),
    #                              theta_offset=E_long)

    #     ax2.set_rmax(self.max_dist + 0.3)
    #     ax2.yaxis.set_visible(False)
    #     ax2.set_theta_zero_location("S")
    #     ax2.tick_params(axis='x', colors='darkgreen', pad=10)
    #     ax2.set_xticks(np.pi/180. * np.linspace(180, -180, 8, endpoint=False))
    #     ax2.set_thetalim(-np.pi, np.pi)
    #     ax2.set_theta_offset(np.deg2rad(long_offset - E_long))
    #     gridlines = ax2.xaxis.get_gridlines()
    #     for xax in gridlines:
    #         xax.set_color('darkgreen')

    #     return ax2

    def plot_pfss(self,
                  pfss_solution,
                  rss=2.5,
                  figsize=(15, 10),
                  dpi=200,
                  return_plot_object=False,
                  vary=False,
                  n_varies=1,
                  long_offset=270,
                  reference_vsw=400.,
                  markers=False,
                  plot_spirals=True,
                  long_sector=None,
                  long_sector_vsw=None,
                  long_sector_color=None,
                  hide_logo=False,
                  numbered_markers=False,  # kept only for backward compatibility
                  outfile=''):
        """
        Plot the Potential Field Source Surface (PFSS) solution on a polar plot with logarithmic r-axis outside the PFSS.
        Tracks an open field line down to the photosphere given a point on the PFSS.

        Parameters
        ----------
        pfss_solution : object
            The PFSS solution object containing the magnetic field data.
        rss : float, optional
            The source surface radius in solar radii. Default is 2.5.
        figsize : tuple, optional
            The size of the figure in inches. Default is (15, 10).
        dpi : int, optional
            The resolution of the figure in dots per inch. Default is 200.
        return_plot_object: bool, optional
            if True, figure and axis object of matplotib are returned, allowing further adjustments to the figure
        vary : bool, optional
            If True, plot varied field lines. Default is False.
        n_varies : int, optional
            Number of varied field lines to plot if vary is True. Default is 1.
        long_offset : float, optional
            Longitude offset for the plot in degrees. Default is 270.
        reference_vsw : float, optional
            Solar wind speed for the reference point in km/s. Default is 400.
        markers : bool or str, optional
            If True or 'letters'/'numbers', plot markers at body positions. Default is False.
        plot_spirals : bool, optional
            If True, plot Parker spirals. Default is True.
        long_sector : list or tuple, optional
            A 2-element list defining the start and end longitude of the cone in degrees. Default is None.
        long_sector_vsw : list or tuple, optional
            Solar wind speeds for the Parker spirals in the long sector. Default is None.
        long_sector_color : str, optional
            Color for the long sector. Default is None.
        hide_logo : bool, optional
            If True, hide the Solar-MACH logo. Default is False.
        numbered_markers: bool, deprecated
            Deprecated option, use markers='numbers' instead!
        outfile : str, optional
            If provided, save the plot to the specified file. Default is ''.

        Returns
        -------
        matplotlib figure and axes or None
            Returns the matplotlib figure and axes if return_plot_object=True (by default set to False), else nothing.

        Raises
        ------
        Exception
            If the PFSS solution and the SolarMACH object use different coordinate systems.

        Notes
        -----
        This function plots the PFSS solution on a polar plot, including the source surface, solar surface, Parker spirals, and field lines. It also supports plotting varied field lines, long sectors, and markers for different bodies. The plot can be saved to a file or displayed using matplotlib or streamlit.
        """
        # check that PFSS solution and SolarMACH object use the same coordinate system
        if not pfss_solution.coordinate_frame.name==self.pos_E.name:
            raise Exception("The provided PFSS solution and the SolarMACH object use different coordinate systems! Aborting.")

        # Constants
        AU = const.au / 1000  # km
        sun_radius = aconst.R_sun.value  # meters

        # r_scaler scales distances from astronomical units to solar radii. unit = [solar radii / AU]
        r_scaler = (AU*1000)/sun_radius

        # carrington longitude of the Earth
        E_long = self.pos_E.lon.value

        # catch old syntax
        if numbered_markers is True and not markers:
            markers='numbers'
            print('')
            print("WARNING: The usage of numbered_markers is deprecated and will be discontinued in the future! Use markers='numbers' instead.")
            print('')

        if markers:
            if markers.lower() in ['n', 'number']:
                markers='numbers'
            if markers.lower() in ['l', 'letter']:
                markers='letters'

        # save inital rcParams and update some of them:
        initial_rcparams = plt.rcParams.copy()
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['font.size'] = 15
        plt.rcParams['agg.path.chunksize'] = 20000

        # init the figure and axes
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize, dpi=dpi)

        # maximum distance anything will be plotted
        # r_max = r_scaler * 5  # 5 AU = 1075 in units of solar radii
        r_max = np.max([r_scaler * 2 * self.max_dist, 5 * r_scaler])  # either twice the actual maximal radius, or minimal 5 AU

        # setting the title
        ax.set_title(str(self.date.to_value('iso', subfmt='date_hm')) + ' (UTC)\n', pad=30)  # , fontsize=26)

        # Plot the source_surface and solar surface
        full_circle_radians = 2*np.pi*np.linspace(0, 1, 200)
        ax.plot(full_circle_radians, np.ones(200)*rss, c='k', ls='--', zorder=3)
        ax.plot(full_circle_radians, np.ones(200), c='darkorange', lw=2.5, zorder=1)

        # Plot the 30 and 60 deg lines on the Sun
        ax.plot(full_circle_radians, np.ones(len(full_circle_radians))*0.866, c='darkgray', lw=1.5, ls=":", zorder=3)  # cos(30deg) = 0.866(O)
        ax.plot(full_circle_radians, np.ones(len(full_circle_radians))*0.500, c='darkgray', lw=1.5, ls=":", zorder=3)  # cos(60deg) = 0.5(0)

        # Plot the gridlines for 10 and 100 solar radii, because this sometimes fails bythe .grid() -method for unkown reason
        ax.plot(full_circle_radians, np.ones(len(full_circle_radians))*10, c="gray", lw=0.6, ls='-', zorder=1)
        ax.plot(full_circle_radians, np.ones(len(full_circle_radians))*100, c="gray", lw=0.6, ls='-', zorder=1)

        # Gather field line objects, photospheric footpoints and magnetic polarities in these lists
        # fieldlines is a class attribute, so that the field lines can be easily 3D plotted with another method
        self.fieldlines = []
        photospheric_footpoints = []
        fieldline_polarities = []

        # Collect the pfss-fieldline footpoints to a dictionary -> to be assembled into a pd DataFrame 
        # at the end.
        pfss_footpoints_dict = {}

        # The radial coordinates for reference parker spiral (plot even outside the figure boundaries to avert visual bugs)
        # reference_array = np.linspace(rss, r_max+200, int(1e3))
        reference_array = np.linspace(rss, r_max, int(1e3))

        # Longitudinal and latitudinal separation angles to Earth's magnetic footpoint
        lon_sep_angles = np.array([])
        lat_sep_angles = np.array([])

        for i, body_id in enumerate(tqdm(self.body_dict)):

            body_lab = self.body_dict[body_id][1]
            body_color = self.body_dict[body_id][2]
            body_vsw = self.body_dict[body_id][4]
            body_pos = self.body_dict[body_id][3]

            pos = body_pos
            dist_body = pos.radius.value

            body_long = pos.lon.value
            body_lat = pos.lat.value

            # take into account solar differential rotation wrt. latitude
            # omega = solar_diff_rot_old(body_lat, diff_rot=self.diff_rot)

            # The radial coordinates (outside source surface) for each object
            # r_array = np.linspace(r_scaler*dist_body*np.cos(np.deg2rad(body_lat)), rss, 1000)
            r_array = np.linspace(r_scaler*dist_body, rss, 1000)

            # plot body positions
            if markers:
                ax.plot(np.deg2rad(body_long), r_scaler*dist_body*np.cos(np.deg2rad(body_lat)), 'o', ms=15, color=body_color, label=body_lab)
                if markers.lower()=='letters':
                    if body_id[:6] == 'STEREO':
                        mark = str(body_id[-1])
                    elif body_id == 'Europa Clipper':
                        mark = 'C'
                    else:
                        mark = str(body_id[0])
                if markers.lower()=='numbers':
                    mark = i+1
                ax.annotate(mark, xy=(np.deg2rad(body_long), r_scaler*dist_body*np.cos(np.deg2rad(body_lat))), color='white',
                            fontsize="small", weight='heavy',
                            horizontalalignment='center',
                            verticalalignment='center')
            else:
                ax.plot(np.deg2rad(body_long), r_scaler*dist_body*np.cos(np.deg2rad(body_lat)), 's', color=body_color, label=body_lab)

            # The angular coordinates are calculated here
            # alpha = longitude + (omega)*(distance-r)/sw
            # alpha_body = np.deg2rad(body_long) + omega / (1000*body_vsw / sun_radius) * (r_scaler*dist_body - r_array) * np.cos(np.deg2rad(body_lat))
            alpha_body = (body_long*u.deg + backmapping_angle(dist_body*u.AU, r_array*u.R_sun, body_lat*u.deg, body_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

            # Plotting the spirals
            if plot_spirals:
                ax.plot(alpha_body, r_array * np.cos(np.deg2rad(body_lat)), color=body_color)

            # To this list we later collect pfss-extrapolated footpoints
            pfss_footpoints = []

            # Acquire an array of (r,lon,lat) coordinates of the open field lines under the pfss
            # based on the footpoint(s) of the sc
            if vary:

                # Triplets contain 35 tuples of (r,lon,lat)
                fline_triplets, fline_objects, varyfline_triplets, varyfline_objects = vary_flines(alpha_body[-1], np.deg2rad(body_lat), pfss_solution, n_varies, rss)

                # Collect field line objects to a list
                self.fieldlines.append(fline_objects[0])
                for varyfline in varyfline_objects:
                    self.fieldlines.append(varyfline)

                # Plot the color coded varied field lines and collect the footpoints to the list
                for triplet in varyfline_triplets:
                    v_fl_r   = triplet[0]
                    v_fl_lon = triplet[1]
                    v_fl_lat = triplet[2]

                    pfss_footpoints.append((v_fl_lon[0], v_fl_lat[0]))

                    fieldline = multicolorline(np.deg2rad(v_fl_lon), np.cos(np.deg2rad(v_fl_lat))*v_fl_r, ax=ax, cvals=v_fl_lat, vmin=-90, vmax=90)

            else:
                # If no varying, then just get one field line from get_field_line_coords()
                # Note that even in the case of a singular fieldline object, this function returns a list
                fline_triplets, fline_objects = get_field_line_coords(alpha_body[-1], np.deg2rad(body_lat), pfss_solution, rss)

                # Collect field line objects to a list
                self.fieldlines.append(fline_objects[0])

            # The middlemost field lines are always plotted regardless if varying or no
            fl_r   = fline_triplets[0][0]
            fl_lon = fline_triplets[0][1]
            fl_lat = fline_triplets[0][2]

            # Plot the color coded field line
            fieldline = multicolorline(np.deg2rad(fl_lon), np.cos(np.deg2rad(fl_lat))*fl_r, ax=ax, cvals=fl_lat, vmin=-90, vmax=90)

            # Finally, save the photospheric footpoint of the middlemost field lines as a tuple and magnetic polarity as +1/-1
            photospheric_footpoints.append((fl_lon[0], fl_lat[0]))
            pfss_footpoints.append((fl_lon[0], fl_lat[0]))
            fieldline_polarities.append(int(fline_objects[0].polarity))

            # Save Earth's magnetic footpoint for later comparison:
            if body_lab == "Earth":
                earth_footpoint = (fl_lon[0], fl_lat[0])
            
            # Finally save all the collected footpoints to the dictionary
            pfss_footpoints_dict[body_id] = pfss_footpoints

        # Calculate footpoint separation angles to Earth's footpoint
        if "Earth" in self.body_dict:
            for footpoint in photospheric_footpoints:
                lon_sep = earth_footpoint[0] - footpoint[0]
                lon_sep = lon_sep if lon_sep < 180 else lon_sep - 360  # Here check that the separation isn't over half a circle
                lat_sep = earth_footpoint[1] - footpoint[1]

                lon_sep_angles = np.append(lon_sep_angles, lon_sep)
                lat_sep_angles = np.append(lat_sep_angles, lat_sep)

            if self.reference_long:
                ref_earth_sep_lon = earth_footpoint[0] - self.reference_long if earth_footpoint[0] - self.reference_long < 180 else earth_footpoint[0] - self.reference_long - 360
                ref_earth_sep_lat = earth_footpoint[1] - self.reference_lat if self.reference_lat else earth_footpoint[1]
                lon_sep_angles = np.append(lon_sep_angles, ref_earth_sep_lon)
                lat_sep_angles = np.append(lat_sep_angles, ref_earth_sep_lat)

            self.pfss_table["Footpoint lon separation to Earth's footpoint lon"] = lon_sep_angles
            self.pfss_table["Footpoint lat separation to Earth's footpoint lat"] = lat_sep_angles

        # Reference longitude and corresponding parker spiral arm
        if self.reference_long:
            delta_ref = self.reference_long
            if delta_ref < 0.:
                delta_ref = delta_ref + 360.
            if self.reference_lat is None:
                ref_lat = 0.
            else:
                ref_lat = self.reference_lat

            # take into account solar differential rotation wrt. latitude
            # omega_ref = solar_diff_rot_old(ref_lat, diff_rot=self.diff_rot)

            # Track up from the reference point a fluxtube
            # Start tracking from the height of 0.1 solar radii
            ref_triplets, ref_objects, varyref_triplets, varyref_objects = vary_flines(np.deg2rad(delta_ref), np.deg2rad(ref_lat), pfss_solution, n_varies, 1.1)

            # Plot the color coded field line
            fieldline = multicolorline(np.deg2rad(ref_triplets[0][1]), np.cos(np.deg2rad(ref_triplets[0][2]))*ref_triplets[0][0], ax=ax, cvals=ref_triplets[0][2], vmin=-90, vmax=90)

            # ... And also plot the color coded flux tube
            for triplet in varyref_triplets:
                v_fl_r   = triplet[0]
                v_fl_lon = triplet[1]
                v_fl_lat = triplet[2]

                fieldline = multicolorline(np.deg2rad(v_fl_lon), np.cos(np.deg2rad(v_fl_lat))*v_fl_r, ax=ax, cvals=v_fl_lat, vmin=-90, vmax=90)

            # Collect reference flux tube to their own list of fieldlines
            self.reference_fieldlines = []
            self.reference_fieldlines.append(ref_objects[0])

            # Boolean switch to keep track what kind of arrow/spiral to draw for reference point
            open_mag_flux_near_ref_point = False

            varyref_objects_longitudes = []
            # Loop the fieldlines, collect them to the list and find the extreme values of longitude at the ss
            for ref_vary in varyref_objects:
                self.reference_fieldlines.append(ref_vary)

                # There still may be closed field lines here, despite trying to avert them in vary_flines() -function. Check here
                # that they do not contribute to the max longitude reach at the ss:
                if ref_vary.polarity == 0:
                    continue
                else:
                    open_mag_flux_near_ref_point = True

                # Check the orientation of the field line; is the first index at the photosphere or the last?
                idx = 0 if ref_vary.coords.radius.value[0] > ref_vary.coords.radius.value[-1] else -1

                # Collect the longitudinal values from the uptracked fluxtube at the source surface height
                varyref_objects_longitudes.append(ref_vary.coords.lon.value[idx])

            """
            # These are test-cases for the following code to select the boundaries of the longitudinal range.
            varyref_objects_longitudes = [-170, 180, 160]  # To be used with Stonyhurst coordinates
            varyref_objects_longitudes = [-30, 0, 15]  # To be used with Stonyhurst coordinates
            varyref_objects_longitudes = [-30, 0, 30, 140]  # To be used with Stonyhurst coordinates
            varyref_objects_longitudes = [-30, 0, 30, 160]  # To be used with Stonyhurst coordinates
            print(varyref_objects_longitudes)
            """

            arrow_dist = rss-0.80
            if open_mag_flux_near_ref_point:
                self.reference_long_min = min(varyref_objects_longitudes)
                self.reference_long_max = max(varyref_objects_longitudes)

                # TODO: IMPROVE!
                # The following is a rather severe if-statement because it renders situations with a real londitudinal spread of bigger than 180° unusable. Unfortunately, there is no better solution as of now.
                if self.reference_long_max-self.reference_long_min > 180:
                    varyref_objects_longitudes2 = []
                    for lon in varyref_objects_longitudes:
                        if (lon > 180) and (self.coord_sys=='Carrington'):
                            varyref_objects_longitudes2.append(lon-360)
                        elif (lon < 0) and (self.coord_sys=='Stonyhurst'):
                            varyref_objects_longitudes2.append(lon+360)
                        else:
                            varyref_objects_longitudes2.append(lon)
                    self.reference_long_max = max(varyref_objects_longitudes2)
                    self.reference_long_min = min(varyref_objects_longitudes2)

                ref_arr = plt.arrow(np.deg2rad(self.reference_long_min), 1, 0, arrow_dist, head_width=0.05, head_length=0.2, edgecolor='black',
                                    facecolor='black', lw=0, zorder=7, overhang=0.1)
                ref_arr = plt.arrow(np.deg2rad(self.reference_long_max), 1, 0, arrow_dist, head_width=0.05, head_length=0.2, edgecolor='black',
                                    facecolor='black', lw=0, zorder=7, overhang=0.1)

                reference_legend_label = f"reference long.\nsector:\n({np.round(self.reference_long_min, 1)}, {np.round(self.reference_long_max, 1)})"
                if (self.reference_long_min < 0) & (self.coord_sys=='Carrington'):
                    reference_legend_label = f"reference long.\nsector:\n({np.round(360+self.reference_long_min, 1)}, {np.round(self.reference_long_max, 1)})"

            else:
                # Set the reach of the flux tube to nan, since it doesn't even reach up to the source surface
                self.reference_long_min, self.reference_long_max = np.nan, np.nan

                ref_arr = plt.arrow(np.deg2rad(self.reference_long), 1, 0, arrow_dist, head_width=0.1, head_length=0.5, edgecolor='black',
                                    facecolor='black', lw=1., zorder=7, overhang=0.5)

                reference_legend_label = f"reference long.\n{self.reference_long} deg"

            # These two spirals and the space between them gets drawn only if we plot spirals and open magnetic flux was found near the ref point
            if plot_spirals and open_mag_flux_near_ref_point:

                # Calculate spirals for the flux tube boundaries
                # alpha_ref_min = np.deg2rad(self.reference_long_min) + omega_ref / (1000*reference_vsw / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(ref_lat))
                # alpha_ref_max = np.deg2rad(self.reference_long_max) + omega_ref / (1000*reference_vsw / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(ref_lat))
                alpha_ref_min = (self.reference_long_min*u.deg + backmapping_angle(rss*u.R_sun, reference_array*u.R_sun, ref_lat*u.deg, reference_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value
                alpha_ref_max = (self.reference_long_max*u.deg + backmapping_angle(rss*u.R_sun, reference_array*u.R_sun, ref_lat*u.deg, reference_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

                # Construct a second r_array for the second spiral for while loop to iterate forwards.
                # This copy of an array will be used to plot both spiral later.
                reference_array2 = np.copy(reference_array)

                # Check that reference angle of the first loop is ahead
                if alpha_ref_min[-1] > alpha_ref_max[-1]:
                    alpha_ref_min_comp = alpha_ref_min[-1] - 2*np.pi
                else:
                    alpha_ref_min_comp = alpha_ref_min[-1]
                
                # While the second spiral is behind the first spiral in angle, extend the second spiral
                while alpha_ref_max[-1] > alpha_ref_min_comp:
                    reference_array2 = np.append(reference_array2, reference_array2[-1] + 1)
                    # alpha_ref_max = np.append(alpha_ref_max, (delta_ref2*u.deg + backmapping_angle(rss*u.R_sun, reference_array2[-1]*u.R_sun, long_sector_lat[1]*u.deg, long_sector_vsw[1]*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value)
                    alpha_ref_max = np.append(alpha_ref_max, (self.reference_long_max*u.deg + backmapping_angle(rss*u.R_sun, reference_array2[-1]*u.R_sun, ref_lat*u.deg, reference_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value)

                # Finally interpolate the first spiral's angles to the coarser second spiral's angles (outside the plot)
                alpha_ref_min = np.interp(reference_array2, reference_array, alpha_ref_min)

                # Introduce r axis to plot that is common between these if and else blocks
                r_to_plot = reference_array2

                # Plot the spirals
                # min_edge = plt.polar(alpha_ref_min, reference_array * np.cos(np.deg2rad(ref_lat)), lw=0.7, color="grey", alpha=0.45)[0]
                # max_edge = plt.polar(alpha_ref_max, reference_array * np.cos(np.deg2rad(ref_lat)), lw=0.7, color="grey", alpha=0.45)[0]
                min_edge = plt.polar(alpha_ref_min, r_to_plot * np.cos(np.deg2rad(ref_lat)), lw=0.7, color="grey", alpha=0.45)[0]
                max_edge = plt.polar(alpha_ref_max, r_to_plot * np.cos(np.deg2rad(ref_lat)), lw=0.7, color="grey", alpha=0.45)[0]

                # Extract 'x' and 'y' values
                x1 = min_edge.get_xdata()
                y1 = min_edge.get_ydata()
                x2 = max_edge.get_xdata()

                # Check that plotted are is between the two spirals, and do not fill after potential crossing
                clause1 = x1 < x2
                clause2 = alpha_ref_min[clause1] < alpha_ref_max[clause1]

                # Take as a selection only the points that fill the above clauses
                y1_fill = y1[clause1][clause2]
                x1_fill = x1[clause1][clause2]
                x2_fill = x2[clause1][clause2]

                # plt.fill_betweenx(y1, x1, x2, lw=0, color="grey", alpha=0.35)
                plt.fill_betweenx(y1_fill, x1_fill, x2_fill, lw=0, color="grey", alpha=0.35)

            # Here we plot spirals (open magnetic flux was not necessarily found) -> draw only one spiral
            if plot_spirals:

                # alpha_ref_single = np.deg2rad(self.reference_long) + omega_ref / (1000*reference_vsw / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(ref_lat))
                alpha_ref_single = (self.reference_long*u.deg + backmapping_angle(rss*u.R_sun, reference_array*u.R_sun, ref_lat*u.deg, reference_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value
                ax.plot(alpha_ref_single, reference_array * np.cos(np.deg2rad(ref_lat)), color="grey",
                        # label=f'field line connecting to\nref. long. (vsw={reference_vsw} km/s)'
                        )

        if long_sector is not None:
            if isinstance(long_sector, (list, tuple)) and len(long_sector)==2:

                delta_ref1 = long_sector[0]
                if delta_ref1 < 0.:
                    delta_ref1 = delta_ref1 + 360.
                delta_ref2 = long_sector[1]
                if delta_ref2 < 0.:
                    delta_ref2 = delta_ref2 + 360.

                # maybe later add option to have different latitudes, so that the long_sector plane is out of the ecliptic
                long_sector_lat = [0, 0]

                # take into account solar differential rotation wrt. latitude
                # omega_ref1 = solar_diff_rot_old(long_sector_lat[0], diff_rot=self.diff_rot)
                # omega_ref2 = solar_diff_rot_old(long_sector_lat[1], diff_rot=self.diff_rot)

                if long_sector_vsw is not None:

                    # Calculate the spirals' angles along r
                    # alpha_ref1 = np.deg2rad(delta_ref1) + omega_ref1 / (1000*long_sector_vsw[0] / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(long_sector_lat[0]))
                    # alpha_ref2 = np.deg2rad(delta_ref2) + omega_ref2 / (1000*long_sector_vsw[1] / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(long_sector_lat[1]))
                    alpha_ref1 = (delta_ref1*u.deg + backmapping_angle(rss*u.R_sun, reference_array*u.R_sun, long_sector_lat[0]*u.deg, long_sector_vsw[0]*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value
                    alpha_ref2 = (delta_ref2*u.deg + backmapping_angle(rss*u.R_sun, reference_array*u.R_sun, long_sector_lat[1]*u.deg, long_sector_vsw[1]*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

                    # Construct a second r_array for the second spiral for while loop to iterate forwards.
                    # This copy of an array will be used to plot both spiral later.
                    reference_array2 = np.copy(reference_array)

                    # Check that reference angle of the first loop is ahead
                    if alpha_ref1[-1] > alpha_ref2[-1]:
                        alpha_ref1_comp = alpha_ref1[-1] - 2*np.pi
                    else:
                        alpha_ref1_comp = alpha_ref1[-1]

                    # While the second spiral is behind the first spiral in angle, extend the second spiral
                    while alpha_ref2[-1] > alpha_ref1_comp:
                        reference_array2 = np.append(reference_array2, reference_array2[-1] + 1)
                        # alpha_ref2 = np.append(alpha_ref2, np.deg2rad(delta_ref2) + omega_ref2 / (1000*long_sector_vsw[1] / sun_radius) * (rss - reference_array2[-1]) * np.cos(np.deg2rad(long_sector_lat[1])))
                        alpha_ref2 = np.append(alpha_ref2, (delta_ref2*u.deg + backmapping_angle(rss*u.R_sun, reference_array2[-1]*u.R_sun, long_sector_lat[1]*u.deg, long_sector_vsw[1]*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value)

                    # Finally interpolate the first spiral's angles to the coarser second spiral's angles (outside the plot)
                    alpha_ref1 = np.interp(reference_array2, reference_array, alpha_ref1)

                    # Introduce r axis to plot that is common between these if and else blocks
                    r_to_plot = reference_array2

                else:
                    # if no solar wind speeds for Parker spirals are provided, use straight lines:
                    # alpha_ref1 = [np.deg2rad(delta_ref1)] * len(reference_array)
                    # alpha_ref2 = [np.deg2rad(delta_ref2)] * len(reference_array)

                    # Vectorize the previous implementation for added performance
                    alpha_ref1 = np.ones(shape=len(reference_array)) * np.deg2rad(delta_ref1)
                    alpha_ref2 = np.ones(shape=len(reference_array)) * np.deg2rad(delta_ref2)

                    # Another reference to r_array to unify this if/else -block's output
                    r_to_plot = reference_array

                c1 = plt.polar(alpha_ref1, r_to_plot * np.cos(np.deg2rad(long_sector_lat[0])), lw=0, color=long_sector_color, alpha=0)[0]
                x1 = c1.get_xdata()
                y1 = c1.get_ydata()
                c2 = plt.polar(alpha_ref2, r_to_plot * np.cos(np.deg2rad(long_sector_lat[1])), lw=0, color=long_sector_color, alpha=0)[0]
                x2 = c2.get_xdata()
                # y2 = c2.get_ydata()

                # Check that plotted are is between the two spirals, and do not fill after potential crossing
                if long_sector_vsw:
                    clause1 = x1 < x2
                    clause2 = alpha_ref1[clause1] < alpha_ref2[clause1]

                    # Take as a selection only the points that fill the above clauses
                    y1_fill = y1[clause1][clause2]
                    x1_fill = x1[clause1][clause2]
                    x2_fill = x2[clause1][clause2]
                else:
                    y1_fill = y1
                    x1_fill = x1
                    x2_fill = x2

                plt.fill_betweenx(y1_fill, x1_fill, x2_fill, lw=0, color=long_sector_color, alpha=0.40)

            else:
                print("Ill-defined 'long_sector'. It should be a 2-element list defining the start and end longitude of the cone in degrees; e.g. 'long_sector=[15,45]'")

        leg1 = ax.legend(loc=(1.05, 0.8), fontsize=13, numpoints=1)

        if markers:
            offset = matplotlib.text.OffsetFrom(leg1, (0.0, 1.0))
            for i, body_id in enumerate(self.body_dict):
                if outfile.split('.')[-1] == 'pdf':
                    yoffset = i*19.25  # 18.5 19.5
                else:
                    yoffset = i*18.7  # 18.5 19.5
                if markers.lower()=='letters':
                    if body_id[:6] == 'STEREO':
                        mark = str(body_id[-1])
                    elif body_id == 'Europa Clipper':
                        mark = 'C'
                    else:
                        mark = str(body_id[0])
                if markers.lower()=='numbers':
                    mark = i+1
                ax.annotate(mark, xy=(1, 1), xytext=(18.3, -11-yoffset), color='white',
                            fontsize="small", weight='heavy', textcoords=offset,
                            horizontalalignment='center',
                            verticalalignment='center', zorder=100)

        if self.reference_long:
            def legend_arrow(width, height, **_):
                return mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True,
                                           head_width=0.75 * height)

            _leg2 = ax.legend([ref_arr], [reference_legend_label], loc=(1.05, 0.6),
                             handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=legend_arrow), },
                             fontsize=15)
            ax.add_artist(leg1)

        # replace 'SEMB-L1' in legend with 'L1' if present
        for text in leg1.get_texts():
            if text.get_text()[:6] == 'SEMB-L':
                text.set_text(text.get_text()[-2:])

        # for Stonyhurst, define the longitude from -180 to 180 (instead of 0 to 360)
        if self.coord_sys=='Stonyhurst':
            ax.set_xticks(np.pi/180. * np.linspace(180, -180, 8, endpoint=False))
            ax.set_thetalim(-np.pi, np.pi)

        # Spin the angular coordinate so that earth is at 6 o'clock
        ax.set_theta_offset(np.deg2rad(long_offset - E_long))

        # For some reason we need to specify 'ylim' here
        ax.set_ylim(0, r_max)
        ax.set_rscale('symlog', linthresh=rss)
        ax.set_rmax(r_max)
        ax.set_rticks([1.0, rss, 10.0, 100.0])
        rlabel_pos = E_long + 120  # -22.5
        ax.set_rlabel_position(rlabel_pos)  # Move radial labels away from plotted line
        # ax.tick_params(which='major', labelsize=22,)

        rlabels = ['1', str(np.round(rss, 2)), r'$10^1$', r'$10^2\ \mathrm{R}_{\odot}$ ']
        ax.set_yticklabels(rlabels)

        # Drawing a circle around the plot, because sometimes for unkown reason the plot boundary is not drawn.
        ax.plot(np.linspace(0, 2*np.pi, 180),
                [r_max]*180,
                color="black",
                lw=3,
                )

        # Cut off unnecessary margins from the plot
        plt.tight_layout()

        ax.tick_params(axis='x', pad=10)

        if not hide_logo:
            txt_x_begin, txt_y_begin = 0.94, 0.05
            ax.text(txt_x_begin, txt_y_begin, 'Solar-MACH',
                    fontfamily='DejaVu Serif', fontsize=24,
                    ha='right', va='bottom', transform=fig.transFigure)
            ax.text(txt_x_begin, txt_y_begin-0.04, 'https://solar-mach.github.io',
                    fontfamily='DejaVu Sans', fontsize=14,
                    ha='right', va='bottom', transform=fig.transFigure)

        # Create the colorbar displaying values of the last fieldline plotted
        _cb = fig.colorbar(fieldline, ax=ax, location="left", anchor=(1.4, 1.2), pad=0.12, shrink=0.6, ticks=[-90, -60, -30, 0, 30, 60, 90])

        # Colorbar is the last object created -> it is the final index in the list of axes
        cb_ax = fig.axes[-1]
        cb_ax.set_ylabel('Heliographic latitude [deg]', fontsize=16)  # 20

        # Add footpoints, magnetic polarities and the reach of reference_long flux tube to PFSS_table
        if self.reference_long:
            photospheric_footpoints.append(self.reference_long)
            fieldline_polarities.append(ref_objects[0].polarity)
            self.pfss_table["Reference flux tube lon range"] = [np.nan if i<len(self.body_dict) else (self.reference_long_min, self.reference_long_max) for i in range(len(self.body_dict)+1)]

        self.pfss_table["Magnetic footpoint (PFSS)"] = photospheric_footpoints
        self.pfss_table["Magnetic polarity"] = fieldline_polarities

        # Assemble the dataframe that contains the pfss-extrapolated magnetic fieldline footpoints
        self.produce_pfss_footpoints_df(footpoints_dict=pfss_footpoints_dict)

        # Update solar wind speed to the reference point
        if reference_vsw:
            self.pfss_table.loc[self.pfss_table["Spacecraft/Body"]=="Reference_point", "Vsw"] = reference_vsw

        if outfile != '':
            plt.savefig(outfile, bbox_inches="tight")

        # don't display figure if saving as pdf file
        if outfile.split('.')[-1] != 'pdf':
            # if using streamlit, send plot to streamlit output, else call plt.show()
            if _isstreamlit():
                import streamlit as st
                st.pyplot(fig, width="content")  # , dpi=200)
            else:
                plt.show()

        # restore initial rcParams that have been saved at the beginning of this function:
        plt.rcParams.update(initial_rcparams)

        if return_plot_object:
            return fig, ax

    def plot_pfss_3d(self, active_area=(None, None, None, None), color_code='object', rss=2.5,
                plot_spirals=True, plot_sun_body_line=False, plot_vertical_line=False,
                markers=False, numbered_markers=False, plot_equatorial_plane=True, plot_3d_grid=True,
                reference_vsw=400, zoom_out=False, return_plot_object=False):
        """
        Plots a 3D visualization of the Potential Field Source Surface (PFSS) model using Plotly.

        Parameters
        ----------
        active_area : tuple, optional
            A tuple specifying the active area in the format (lonmax, lonmin, latmax, latmin). Default is (None, None, None, None).
        color_code : str, optional
            Specifies the color coding for the field lines. Options are 'object' or 'polarity'. Default is 'object'.
        rss : float, optional
            The source surface radius in solar radii. Default is 2.5.
        plot_spirals : bool, optional
            If True, plots the Parker spirals. Default is True.
        plot_sun_body_line : bool, optional
            If True, plots the direct line from the Sun to the body. Default is False.
        plot_vertical_line : bool, optional
            If True, plots vertical lines from the heliographic equatorial plane to each body. Default is False.
        markers : bool or str, optional
            If True or 'letters'/'numbers', plot markers at body positions. Default is False.
        plot_equatorial_plane : bool, optional
            If True, plots the equatorial plane. Default is True.
        plot_3d_grid : bool, optional
            If True, plots grid and axis for x, y, z. Default is True.
        reference_vsw : int, optional
            The solar wind speed for the reference field line in km/s. Default is 400.
        zoom_out : bool, optional
            If True, zooms out the plot to show the entire field of view. Default is False.
        return_plot_object : bool, optional
            if True, figure object of plotly is returned, allowing further adjustments to the figure
        numbered_markers : bool, deprecated
            Deprecated option, use markers='numbers' instead!

        Returns
        -------
        plotly figure or None
            Returns the plotly figure if return_plot_object=True (by default set to False), else nothing.
        """
        import plotly.graph_objects as go
        from astropy.constants import R_sun
        from plotly.graph_objs.scatter3d import Line

        hide_logo = False  # optional later keyword to hide logo on figure

        # catch old syntax
        if numbered_markers is True and not markers:
            markers='numbers'
            print('')
            print("WARNING: The usage of numbered_markers is deprecated and will be discontinued in the future! Use markers='numbers' instead.")
            print('')

        if markers:
            if markers.lower() in ['n', 'number']:
                markers='numbers'
            if markers.lower() in ['l', 'letter']:
                markers='letters'

        AU = const.au / 1000  # km

        # scale from AU/km to solar radii/km
        # r_array = r_array * AU / R_sun.to(u.km).value
        max_dist2 = self.max_dist * AU / R_sun.to(u.km).value

        # Flare site (or whatever area of interest) is plotted at this height
        FLARE_HEIGHT = 1.005

        object_names = list(self.body_dict.keys())

        # choose the color coding as either polarity or object
        if color_code=='object':

            # Number of objects, modulator that is the amount of field lines per object and color list that holds the corresponding color names
            num_objects = len(self.body_dict)
            modulator = len(self.fieldlines)//num_objects
            color_list = [self.body_dict[body_id][2] for body_id in self.body_dict]

            # Plotly doesn't like color 'b', so check if that exists and change it to more specific identifier
            for i, color in enumerate(color_list):
                if color=='b':
                    color_list[i] = "blue"

        elif color_code=='polarity':

            colors = {0: 'black',
                      -1: 'blue',
                      1: 'red'}

        else:
            raise Exception(f"Invalid color_code=={color_code}. Choose either 'polarity' or 'object'.")

        # create the sun object, a sphere, for plotting
        sun = sphere(radius=1, clr='#ffff55')  # '#ffff00'

        # and add it to the list of traces
        # traces are all the objects that will be plotted
        traces = [sun]

        # go through field lines, assign a color to them and append them to the list of traces
        for i, field_line in enumerate(self.fieldlines):
            coords = field_line.coords
            coords.representation_type = "cartesian"

            if color_code=="polarity":
                color = colors.get(field_line.polarity)
            if color_code=='object':
                color = color_list[i//modulator]

            # New object's lines being plotted
            if i%modulator==0:
                line_label = object_names[i//modulator]
                show_in_legend = True
            else:
                show_in_legend = False

            # never show field lines in legend
            show_in_legend = False

            fieldline_trace = go.Scatter3d(x=coords.x/R_sun, y=coords.y/R_sun, z=coords.z/R_sun,
                                           mode='lines',
                                           line=Line(color=color, width=3.5),
                                           name=line_label,
                                           showlegend=show_in_legend
                                           )

            traces.append(fieldline_trace)

        # If there is a reference_longitude that was plotted, add it to the list of names
        if self.reference_long:

            for i, field_line in enumerate(self.reference_fieldlines):

                coords = field_line.coords
                coords.representation_type = "cartesian"

                if color_code=="polarity":
                    color = colors.get(field_line.polarity)
                if color_code=='object':
                    color = "black"

                # New object's lines being plotted
                if i==0:
                    if self.reference_lat is None:
                        ref_lat = 0
                    else:
                        ref_lat = self.reference_lat
                    line_label = f"Reference_point: {self.reference_long, ref_lat}"
                    show_in_legend = True
                else:
                    show_in_legend = False

                # never show field lines in legend
                show_in_legend = False

                fieldline_trace = go.Scatter3d(x=coords.x/R_sun, y=coords.y/R_sun, z=coords.z/R_sun,
                                               mode='lines',
                                               line=Line(color=color, width=3.5),
                                               name=line_label,
                                               showlegend=show_in_legend
                                               )

                traces.append(fieldline_trace)

        if active_area[0]:

            # the flare area is bound by the extreme values of longitude and latitude
            lonmax, lonmin = np.deg2rad(active_area[0]), np.deg2rad(active_area[1])
            latmax, latmin = np.deg2rad(active_area[2]), np.deg2rad(active_area[3])

            # the perimeter of flare area in four segments
            perimeter1 = (np.linspace(lonmin, lonmax, 10), [latmax]*10)
            perimeter2 = (np.linspace(lonmin, lonmax, 10), [latmin]*10)
            perimeter3 = ([lonmax]*10, np.linspace(latmin, latmax, 10))
            perimeter4 = ([lonmin]*10, np.linspace(latmin, latmax, 10))

            # the perimeter in terms of elevation and azimuthal angles
            perimeter_phis = np.append(np.append(np.append(perimeter1[0], perimeter2[0]), perimeter3[0]), perimeter4[0])
            perimeter_thetas = np.append(np.append(np.append(perimeter1[1], perimeter2[1]), perimeter3[1]), perimeter4[1])

            # the perimeter in terms of cartesian components
            perimeter_cartesian = spheric2cartesian([FLARE_HEIGHT]*40, theta=perimeter_thetas, phi=perimeter_phis)

            # flare area object
            active_area = go.Scatter3d(x=perimeter_cartesian[0], y=perimeter_cartesian[1], z=perimeter_cartesian[2],
                                       mode='lines',
                                       line=Line(color='purple', width=5.5),
                                       name="Active Area"
                                       )

            traces.append(active_area)

        # the 0-latitude line, i.e. the equator
        equator_sphericals = (np.ones(101)*FLARE_HEIGHT, np.zeros(101), np.linspace(0, 2*np.pi, 101))
        equator_cartesians = spheric2cartesian(equator_sphericals[0], equator_sphericals[1], equator_sphericals[2])

        equator_line = go.Scatter3d(x=equator_cartesians[0], y=equator_cartesians[1], z=equator_cartesians[2],
                                    mode='lines',
                                    line=Line(color='gray', width=5.5),
                                    name="Solar equator",
                                    showlegend=False,
                                    )

        traces.append(equator_line)

        # create the figure
        fig = go.Figure(data=traces)

        # TODO: is r_array falsly projected to the ecliptic here again???
        # build array of values for radius (in spherical coordinates!) given in AU!
        # r_array = np.arange(0.007, (self.max_dist+0.1)/np.cos(np.deg2rad(self.max_dist_lat)) + 3.0, 0.001)
        # r_array = np.arange(0.007, (max_dist2+0.1)/np.cos(np.deg2rad(self.max_dist_lat)) + 3.0, 0.001)  # Define with lower "resolution"!
        r_array = np.arange(0.007, (max_dist2+0.29*const.au/R_sun.to(u.m).value)/np.cos(np.deg2rad(self.max_dist_lat)) + 3.0, 0.05)

        for i, body_id in enumerate(self.body_dict):
            # body_lab = self.body_dict[body_id][1]
            # body_color = self.body_dict[body_id][2]
            body_vsw = self.body_dict[body_id][4]
            body_pos = self.body_dict[body_id][3]

            pos = body_pos
            # dist_body = pos.radius.value
            dist_body = (pos.radius.to(u.m)/R_sun).value

            body_long = pos.lon.value
            body_lat = pos.lat.value

            # take into account solar differential rotation wrt. latitude
            # omega = solar_diff_rot_old(body_lat, diff_rot=self.diff_rot)

            # TODO: np.cos(np.deg2rad(body_lat) correct????
            # alpha_body = np.deg2rad(body_long) + omega / (body_vsw / AU) * (dist_body - r_array) * np.cos(np.deg2rad(body_lat))
            # alpha_body = np.deg2rad(body_long) + omega / (body_vsw / R_sun.to(u.km).value) * (dist_body - r_array) * np.cos(np.deg2rad(body_lat))
            alpha_body = (body_long*u.deg + backmapping_angle(dist_body*u.R_sun, r_array*u.R_sun, body_lat*u.deg, body_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

            if plot_spirals:
                phi = np.ones(len(r_array))*np.deg2rad(body_lat)
                # x, y, z = spheric2cartesian(r_array * np.cos(np.deg2rad(body_lat)), phi, alpha_body)
                x, y, z = spheric2cartesian(r_array[(r_array>=rss) & (r_array<=max_dist2+0.29*const.au/R_sun.to(u.m).value)],
                                            phi[(r_array>=rss) & (r_array<=max_dist2+0.29*const.au/R_sun.to(u.m).value)],
                                            alpha_body[(r_array>=rss) & (r_array<=max_dist2+0.29*const.au/R_sun.to(u.m).value)])

                fig.add_trace(go.Scatter3d(x=x,
                                           y=y,
                                           z=z,
                                           mode='lines',
                                           name=f'{body_id} magnetic field line',
                                           showlegend=False,
                                           line=dict(color=body_dict[body_id][2]),
                                           # thetaunit="radians"
                                           ))

            if plot_sun_body_line:
                x, y, z = spheric2cartesian([0.01, dist_body], [0.01, np.deg2rad(body_lat)], [np.deg2rad(body_long), np.deg2rad(body_long)])

                fig.add_trace(go.Scatter3d(x=x,
                                           y=y,
                                           z=z,
                                           mode='lines',
                                           name=f'{body_id} direct line',
                                           showlegend=False,
                                           line=dict(color=body_dict[body_id][2], dash='dot'),
                                           # thetaunit="radians"
                                           ))

            if plot_vertical_line:
                x, y, z = spheric2cartesian([dist_body*np.cos(np.deg2rad(body_lat)), dist_body], [0.0, np.deg2rad(body_lat)], [np.deg2rad(body_long), np.deg2rad(body_long)])

                fig.add_trace(go.Scatter3d(x=x,
                                           y=y,
                                           z=z,
                                           mode='lines',
                                           name=f'{body_id} direct line',
                                           showlegend=False,
                                           line=dict(width=5, color=body_dict[body_id][2], dash='dot'),
                                           # thetaunit="radians"
                                           ))
                fig.add_trace(go.Scatter3d(x=[x[0]],
                                           y=[y[0]],
                                           z=[z[0]],
                                           mode='markers',
                                           name=body_id,
                                           showlegend=False,
                                           marker=dict(symbol='circle', size=3, color=body_dict[body_id][2]),
                                           ))

            if markers:
                if markers.lower()=='numbers':
                    str_number = f'<b>{i+1}</b>'
                if markers.lower()=='letters':
                    if body_id[:6] == 'STEREO':
                        str_number = f'<b>{body_id[-1]}</b>'
                    elif body_id == 'Europa Clipper':
                        str_number = '<b>C</b>'
                    else:
                        str_number = f'<b>{body_id[0]}</b>'
                symbol = 'circle'
            else:
                str_number = None
                symbol = 'square'

            # SkyCoord transformed to cartesian correspond to HEEQ for Stonyhurst
            fig.add_trace(go.Scatter3d(x=[(body_pos.cartesian.x.to(u.m)/R_sun).value],
                                       y=[(body_pos.cartesian.y.to(u.m)/R_sun).value],
                                       z=[(body_pos.cartesian.z.to(u.m)/R_sun).value],
                                       mode='markers+text',
                                       name=body_id,
                                       marker=dict(symbol=symbol, size=10, color=body_dict[body_id][2]),
                                       # text=[f'<b>{body_id}</b>'],
                                       text=[str_number],
                                       textfont=dict(color="white", size=14),
                                       textposition="middle center",
                                       # thetaunit="radians"
                                       ))

        if self.reference_long is not None:
            delta_ref = self.reference_long
            if delta_ref < 0.:
                delta_ref = delta_ref + 360.
            if self.reference_lat is None:
                ref_lat = 0.
            else:
                ref_lat = self.reference_lat

            # omega_ref = solar_diff_rot_old(ref_lat, diff_rot=self.diff_rot)
            # alpha_ref = np.deg2rad(delta_ref) + omega_ref / (reference_vsw / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(ref_lat))
            alpha_ref = (delta_ref*u.deg + backmapping_angle(self.target_solar_radius*aconst.R_sun, r_array*u.R_sun, ref_lat*u.deg, reference_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

            arrow_dist = min([max_dist2/3.2, 2.])
            if zoom_out:
                arrow_dist = min([max_dist2/3.2, 2.*(aconst.au/aconst.R_sun).value])
            x, y, z = spheric2cartesian([0.0, arrow_dist], [np.deg2rad(ref_lat), np.deg2rad(ref_lat)], [np.deg2rad(delta_ref), np.deg2rad(delta_ref)])

            # arrow plotting based on plotly hack provided through
            # https://stackoverflow.com/a/66792953/2336056
            arrow_tip_ratio = 0.4
            arrow_starting_ratio = 0.95

            # plot arrow line
            fig.add_trace(go.Scatter3d(x=x,
                                       y=y,
                                       z=z,
                                       mode='lines',
                                       # marker=dict(symbol="arrow", size=15, angleref="previous", color="black"),  # only works in plotly 2d plots
                                       name=f'reference<br>(long={self.reference_long}°, lat={ref_lat}°)',
                                       showlegend=True,
                                       line=dict(color="black", width=3),
                                       # thetaunit="radians"
                                       ))
            # plot arrow head
            fig.add_trace(go.Cone(x=[x[0] + arrow_starting_ratio*(x[1] - x[0])],
                                  y=[y[0] + arrow_starting_ratio*(y[1] - y[0])],
                                  z=[z[0] + arrow_starting_ratio*(z[1] - z[0])],
                                  u=[arrow_tip_ratio*(x[1] - x[0])],
                                  v=[arrow_tip_ratio*(y[1] - y[0])],
                                  w=[arrow_tip_ratio*(z[1] - z[0])],
                                  name=f'reference<br>(long={self.reference_long}°, lat={ref_lat}°)',
                                  showlegend=False,
                                  showscale=False,
                                  colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(0,0,0)']]
                                  ))

            if plot_spirals:
                phi = np.ones(len(r_array))*np.deg2rad(ref_lat)
                x, y, z = spheric2cartesian(r_array[r_array<=max_dist2+0.29*const.au/R_sun.to(u.m).value],
                                            phi[r_array<=max_dist2+0.29*const.au/R_sun.to(u.m).value],
                                            alpha_ref[r_array<=max_dist2+0.29*const.au/R_sun.to(u.m).value])

                fig.add_trace(go.Scatter3d(x=x,
                                           y=y,
                                           z=z,
                                           mode='lines',
                                           name=f'field line connecting to<br>reference (vsw={reference_vsw} km/s) ',
                                           showlegend=True,
                                           line=dict(color="black", dash="dot"),
                                           # thetaunit="radians"
                                           ))
            #     ax.plot(alpha_ref, r_array * np.cos(np.deg2rad(ref_lat)), '--k', label=f'field line connecting to\nref. long. (vsw={reference_vsw} km/s)')

        if not plot_3d_grid:
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

        if self.max_dist < 2:
            ring_steps = 0.5
        elif self.max_dist < 10:
            ring_steps = 1
        elif self.max_dist < 50:
            ring_steps = 5
        elif self.max_dist < 50:
            ring_steps = 10
        else:
            ring_steps = 50

        if plot_equatorial_plane:
            # fig.add_trace(go.Surface(x=np.linspace(-200, 200, 100),
            #                          y=np.linspace(-200, 200, 100),
            #                          z=np.zeros((100, 100)),
            #                          hoverinfo='skip',
            #                          colorscale='gray', showscale=False, opacity=0.2))

            # add rings
            def add_ring(fig, radius, line=dict(color="black", dash="dot")):
                angle = np.linspace(0, 2*np.pi, 150)
                x = radius*np.cos(angle)
                y = radius*np.sin(angle)
                z = np.zeros((len(x)))
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                           mode='lines',
                                           line=line,
                                           showlegend=False,
                                           ))
                return

            add_ring(fig, max_dist2 + 0.29*const.au/R_sun.to(u.m).value, line=dict(width=5, color="black"))
            for rr in np.arange(0, max_dist2 + 0.29*const.au/R_sun.to(u.m).value, ring_steps*const.au/R_sun.to(u.m).value)[1:]:
                rr = int(rr)
                add_ring(fig, rr, line=dict(color="lightgray"))
                # x2, y2, z2 = spheric2cartesian([rr+ring_steps/5], [np.deg2rad(0)], [np.deg2rad(120)])
                x2, y2, z2 = spheric2cartesian([rr+ring_steps/5*const.au/R_sun.to(u.m).value], [np.deg2rad(0)], [np.deg2rad(120)])
                fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='text',
                                           marker=dict(symbol=symbol, size=1, color='red'),
                                           text=[f'{rr} R<sub>☉</sub>'],
                                           textfont=dict(color="black", size=16),
                                           textposition="middle center",
                                           showlegend=False,
                                           ))

            # if max_dist2 < 2*const.au/R_sun.to(u.m).value:
            #     for rr in np.arange(0, max_dist2 + 0.29*const.au/R_sun.to(u.m).value, 0.5*const.au/R_sun.to(u.m).value)[1:]:
            #         add_ring(fig, rr, line=dict(color="lightgray"))
            #         x2, y2, z2 = spheric2cartesian([rr+0.1*const.au/R_sun.to(u.m).value], [np.deg2rad(0)], [np.deg2rad(120)])
            #         fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='text',
            #                                    marker=dict(symbol=symbol, size=1, color='red'),
            #                                    text=[f'{rr}'],
            #                                    textfont=dict(color="black", size=16),
            #                                    textposition="middle center",
            #                                    showlegend=False,
            #                                    ))
            # else:
            #     if max_dist2 < 10*const.au/R_sun.to(u.m).value:
            #         for rr in np.arange(0, max_dist2 + 0.29*const.au/R_sun.to(u.m).value, 1.0*const.au/R_sun.to(u.m).value)[1:]:
            #             add_ring(fig, rr, line=dict(color="lightgray"))
            #             x2, y2, z2 = spheric2cartesian([rr+0.1*const.au/R_sun.to(u.m).value], [np.deg2rad(0)], [np.deg2rad(120)])
            #             fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='text',
            #                                        marker=dict(symbol=symbol, size=1, color='red'),
            #                                        text=[f'{rr}'],
            #                                        textfont=dict(color="black", size=16),
            #                                        textposition="middle center",
            #                                        showlegend=False,
            #                                        ))

            # add spokes
            for s_long in np.arange(0, 360, 45):
                x, y, z = spheric2cartesian([0.01, max_dist2+0.29*const.au/R_sun.to(u.m).value], [0.0, 0.0], [np.deg2rad(s_long), np.deg2rad(s_long)])

                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                           showlegend=False,
                                           line=dict(color='lightgray'),
                                           ))

                x2, y2, z2 = spheric2cartesian([0.01, max_dist2+(0.29+ring_steps/3)*const.au/R_sun.to(u.m).value+0.1*const.au/R_sun.to(u.m).value], [0.0, 0.0], [np.deg2rad(s_long), np.deg2rad(s_long)])
                fig.add_trace(go.Scatter3d(x=[x2[-1]], y=[y2[-1]], z=[z2[-1]], mode='text',
                                           marker=dict(symbol=symbol, size=1, color='red'),
                                           text=[f'{s_long}°'],
                                           textfont=dict(color="black", size=16),
                                           textposition="middle center",
                                           showlegend=False,
                                           ))

        stitle = str(self.date.to_value('iso', subfmt='date_hm'))
        fig.update_layout(title=dict(text=stitle+' (UTC)', x=0.5, xref="paper", xanchor="center", font=dict(size=22, weight="normal"), automargin=True, yref='paper'),
                          legend=dict(itemsizing='constant', xref="paper", yref="paper", yanchor="top", y=1.0, xanchor="right", x=1.3, font=dict(size=16)))

        if not hide_logo:
            logo_x = 1.3
            logo_y = 0.05
            fig.add_annotation(x=logo_x, y=logo_y,
                               xref="paper", yref="paper",
                               xanchor="right",
                               yanchor="bottom",
                               font=dict(color="black", size=23, family="DejaVu Serif"),
                               text="Solar-MACH",
                               showarrow=False
                               )
            fig.add_annotation(x=logo_x, y=logo_y,
                               xref="paper", yref="paper",
                               xanchor="right",
                               yanchor="top",
                               font=dict(color="black", size=13, family="DejaVu Sans"),
                               text="https://solar-mach.github.io",
                               showarrow=False
                               )

        xyz_range = 2.5
        if zoom_out:
            xyz_range = max_dist2+ring_steps*const.au/R_sun.to(u.m).value

        # additional figure settings, like aspect mode, extreme values of axes etc...
        fig.update_layout(scene_aspectmode='cube')
        fig.update_layout(scene=dict(xaxis=dict(title="X / R<sub>☉</sub>", nticks=4, range=[-xyz_range, xyz_range],),
                                     yaxis=dict(title="Y / R<sub>☉</sub>", nticks=4, range=[-xyz_range, xyz_range],),
                                     zaxis=dict(title="Z / R<sub>☉</sub>", nticks=4, range=[-xyz_range, xyz_range],),
                                     xaxis_tickfont=dict(weight=500, size=14),
                                     yaxis_tickfont=dict(weight=500, size=14),
                                     zaxis_tickfont=dict(weight=500, size=14),
                                     xaxis_title_font=dict(weight=500, size=16),
                                     yaxis_title_font=dict(weight=500, size=16),
                                     zaxis_title_font=dict(weight=500, size=16),
                                     ),
                          width=1024, height=1024,
                          margin=dict(r=20, l=10, b=10, t=10)
                          )

        config = {'toImageButtonOptions': {'format': 'png',  # one of png, svg, jpeg, webp
                                           'filename': 'Solar-MACH_'+(stitle.replace(' ', '_')).replace(':', '-')+'_PFSS',
                                           # 'height': 500,
                                           # 'width': 700,
                                           # 'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
                                           }
                  }

        if _isstreamlit():
            # fig.update_layout(width=700, height=700)
            import streamlit as st
            # import streamlit.components.v1 as components
            # components.html(fig.to_html(include_mathjax='cdn'), height=700)
            st.plotly_chart(fig.update_layout(width=700, height=700),
                            theme=None,  # "streamlit",
                            use_container_width=True,
                            config=config)
        else:
            fig.show(config=config)

        if return_plot_object:
            return fig
        else:
            return

    # for backward compatibility, copy the function under the old name too
    pfss_3d = copy.copy(plot_pfss_3d)

    def plot_3d(self, plot_spirals=True, plot_sun_body_line=True, plot_vertical_line=False, markers=False, numbered_markers=False, plot_equatorial_plane=True, plot_3d_grid=True, reference_vsw=400, return_plot_object=False):
        """
        Generates a 3D plot of the solar system with various optional features.

        Parameters
        ----------
        plot_spirals : bool, optional
            If True, plots the magnetic field lines as spirals. Default is True.
        plot_sun_body_line : bool, optional
            If True, plots direct lines from the Sun to each body. Default is True.
        plot_vertical_line : bool, optional
            If True, plots vertical lines from the heliographic equatorial plane to each body. Default is False.
        markers : bool or str, optional
            If True or 'letters'/'numbers', plot markers at body positions. Default is False.
        plot_equatorial_plane : bool, optional
            If True, plots the equatorial plane. Default is True.
        plot_3d_grid : bool, optional
            If True, plots grid and axis for x, y, z. Default is True.
        reference_vsw : int, optional
            The reference solar wind speed in km/s. Default is 400.
        return_plot_object : bool, optional
            if True, figure object of plotly is returned, allowing further adjustments to the figure
        numbered_markers : bool, deprecated
            Deprecated option, use markers='numbers' instead!

        Returns
        -------
        plotly figure or None
            Returns the plotly figure if return_plot_object=True (by default set to False), else nothing.
        """

        import plotly.graph_objects as go
        # from astropy.constants import R_sun
        # from plotly.graph_objs.scatter3d import Line

        hide_logo = False  # optional later keyword to hide logo on figure
        # AU = const.au / 1000  # km
        # sun_radius = aconst.R_sun.value  # meters

        # catch old syntax
        if numbered_markers is True and not markers:
            markers='numbers'
            print('')
            print("WARNING: The usage of numbered_markers is deprecated and will be discontinued in the future! Use markers='numbers' instead.")
            print('')

        if markers:
            if markers.lower() in ['n', 'number']:
                markers='numbers'
            if markers.lower() in ['l', 'letter']:
                markers='letters'

        # build array of values for radius (in spherical coordinates!) given in AU!
        r_array = np.arange(0.007, (self.max_dist+0.1)/np.cos(np.deg2rad(self.max_dist_lat)) + 3.0, 0.001)

        # create the sun object, a sphere, for plotting
        # use 10*R_sun to have it visilble!
        sun = sphere(radius=(10*u.solRad).to(u.AU).value, clr='#ffff55')  # '#ffff00'

        # create the figure
        fig = go.Figure([sun])

        # additional figure settings, like aspect mode, extreme values of axes etc...
        fig.update_layout()

        for i, body_id in enumerate(self.body_dict):
            # body_lab = self.body_dict[body_id][1]
            # body_color = self.body_dict[body_id][2]
            body_vsw = self.body_dict[body_id][4]
            body_pos = self.body_dict[body_id][3]

            pos = body_pos
            dist_body = pos.radius.value

            body_long = pos.lon.value
            body_lat = pos.lat.value

            # take into account solar differential rotation wrt. latitude
            # omega = solar_diff_rot_old(body_lat, diff_rot=self.diff_rot)

            # TODO: np.cos(np.deg2rad(body_lat) correct????
            # alpha_body = np.deg2rad(body_long) + omega / (body_vsw / AU) * (dist_body - r_array) * np.cos(np.deg2rad(body_lat))
            alpha_body = (body_long*u.deg + backmapping_angle(dist_body*u.AU, r_array*u.AU, body_lat*u.deg, body_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

            if plot_spirals:
                spiral_width = 3
                phi = np.ones(len(r_array))*np.deg2rad(body_lat)
                # x, y, z = spheric2cartesian(r_array * np.cos(np.deg2rad(body_lat)), phi, alpha_body)
                x, y, z = spheric2cartesian(r_array[r_array<=self.max_dist+0.29],
                                            phi[r_array<=self.max_dist+0.29],
                                            alpha_body[r_array<=self.max_dist+0.29])

                fig.add_trace(go.Scatter3d(x=x,
                                           y=y,
                                           z=z,
                                           mode='lines',
                                           name=f'{body_id} magnetic field line',
                                           showlegend=False,
                                           line=dict(width=spiral_width, color=body_dict[body_id][2]),
                                           # thetaunit="radians"
                                           ))

            if plot_sun_body_line:
                x, y, z = spheric2cartesian([0.01, dist_body], [0.01, np.deg2rad(body_lat)], [np.deg2rad(body_long), np.deg2rad(body_long)])

                fig.add_trace(go.Scatter3d(x=x,
                                           y=y,
                                           z=z,
                                           mode='lines',
                                           name=f'{body_id} direct line',
                                           showlegend=False,
                                           line=dict(color=body_dict[body_id][2], dash='dot'),
                                           # thetaunit="radians"
                                           ))

            if plot_vertical_line:
                x, y, z = spheric2cartesian([dist_body*np.cos(np.deg2rad(body_lat)), dist_body], [0.0, np.deg2rad(body_lat)], [np.deg2rad(body_long), np.deg2rad(body_long)])

                fig.add_trace(go.Scatter3d(x=x,
                                           y=y,
                                           z=z,
                                           mode='lines',
                                           name=f'{body_id} direct line',
                                           showlegend=False,
                                           line=dict(width=5, color=body_dict[body_id][2], dash='dot'),
                                           # thetaunit="radians"
                                           ))
                fig.add_trace(go.Scatter3d(x=[x[0]],
                                           y=[y[0]],
                                           z=[z[0]],
                                           mode='markers',
                                           name=body_id,
                                           showlegend=False,
                                           marker=dict(symbol='circle', size=3, color=body_dict[body_id][2]),
                                           ))

            if markers:
                if markers.lower()=='numbers':
                    str_number = f'<b>{i+1}</b>'
                if markers.lower()=='letters':
                    if body_id[:6] == 'STEREO':
                        str_number = f'<b>{body_id[-1]}</b>'
                    elif body_id == 'Europa Clipper':
                        str_number = '<b>C</b>'
                    else:
                        str_number = f'<b>{body_id[0]}</b>'
                symbol = 'circle'
            else:
                str_number = None
                symbol = 'square'

            # customdata=[[dist_body], [body_long], [body_lat]]
            # SkyCoord transformed to cartesian correspond to HEEQ for Stonyhurst
            fig.add_trace(go.Scatter3d(x=[body_pos.cartesian.x.value],
                                       y=[body_pos.cartesian.y.value],
                                       z=[body_pos.cartesian.z.value],
                                       mode='markers+text',
                                       name=body_id,
                                       marker=dict(symbol=symbol, size=10, color=body_dict[body_id][2]),
                                       # text=[f'<b>{body_id}</b>'],
                                       text=[str_number],
                                       textfont=dict(color="white", size=14),
                                       textposition="middle center",
                                       # customdata=[[dist_body], [body_long], [body_lat]],
                                       # hovertemplate='r:%{customdata[0]:.3f} <br>t: %{customdata[1]:.3f} <br>p: %{customdata[2]:.3f} ',
                                       # thetaunit="radians"
                                       ))

        if self.reference_long is not None:
            delta_ref = self.reference_long
            if delta_ref < 0.:
                delta_ref = delta_ref + 360.
            if self.reference_lat is None:
                ref_lat = 0.
            else:
                ref_lat = self.reference_lat

            # omega_ref = solar_diff_rot_old(ref_lat, diff_rot=self.diff_rot)
            # alpha_ref = np.deg2rad(delta_ref) + omega_ref / (reference_vsw / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(ref_lat))
            alpha_ref = (delta_ref*u.deg + backmapping_angle(self.target_solar_radius*aconst.R_sun, r_array*u.AU, ref_lat*u.deg, reference_vsw*u.km/u.s, diff_rot=self.diff_rot)).to(u.rad).value

            arrow_dist = min([self.max_dist/3.2, 2.])
            x, y, z = spheric2cartesian([0.0, arrow_dist], [np.deg2rad(ref_lat), np.deg2rad(ref_lat)], [np.deg2rad(delta_ref), np.deg2rad(delta_ref)])

            # arrow plotting based on plotly hack provided through
            # https://stackoverflow.com/a/66792953/2336056
            arrow_tip_ratio = 0.4
            arrow_starting_ratio = 0.95

            # plot arrow line
            fig.add_trace(go.Scatter3d(x=x,
                                       y=y,
                                       z=z,
                                       mode='lines',
                                       # marker=dict(symbol="arrow", size=15, angleref="previous", color="black"),  # only works in plotly 2d plots
                                       name=f'reference<br>(long={self.reference_long}°, lat={ref_lat}°)',
                                       showlegend=True,
                                       line=dict(color="black", width=3),
                                       # thetaunit="radians"
                                       ))
            # plot arrow head
            fig.add_trace(go.Cone(x=[x[0] + arrow_starting_ratio*(x[1] - x[0])],
                                  y=[y[0] + arrow_starting_ratio*(y[1] - y[0])],
                                  z=[z[0] + arrow_starting_ratio*(z[1] - z[0])],
                                  u=[arrow_tip_ratio*(x[1] - x[0])],
                                  v=[arrow_tip_ratio*(y[1] - y[0])],
                                  w=[arrow_tip_ratio*(z[1] - z[0])],
                                  name=f'reference<br>(long={self.reference_long}°, lat={ref_lat}°)',
                                  showlegend=False,
                                  showscale=False,
                                  colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(0,0,0)']]
                                  ))

            if plot_spirals:
                phi = np.ones(len(r_array))*np.deg2rad(ref_lat)
                x, y, z = spheric2cartesian(r_array[r_array<=self.max_dist+0.29],
                                            phi[r_array<=self.max_dist+0.29],
                                            alpha_ref[r_array<=self.max_dist+0.29])

                fig.add_trace(go.Scatter3d(x=x,
                                           y=y,
                                           z=z,
                                           mode='lines',
                                           name=f'field line connecting to<br>reference (vsw={reference_vsw} km/s) ',
                                           showlegend=True,
                                           line=dict(width=spiral_width, color="black", dash="dot"),
                                           # thetaunit="radians"
                                           ))
            #     ax.plot(alpha_ref, r_array * np.cos(np.deg2rad(ref_lat)), '--k', label=f'field line connecting to\nref. long. (vsw={reference_vsw} km/s)')

        if not plot_3d_grid:
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

        if self.max_dist < 2:
            ring_steps = 0.5
        elif self.max_dist < 10:
            ring_steps = 1
        elif self.max_dist < 50:
            ring_steps = 5
        elif self.max_dist < 50:
            ring_steps = 10
        else:
            ring_steps = 50

        if plot_equatorial_plane:
            # fig.add_trace(go.Surface(x=np.linspace(-200, 200, 100),
            #                          y=np.linspace(-200, 200, 100),
            #                          z=np.zeros((100, 100)),
            #                          hoverinfo='skip',
            #                          colorscale='gray', showscale=False, opacity=0.2))

            # add rings
            def add_ring(fig, radius, line=dict(color="black", dash="dot")):
                angle = np.linspace(0, 2*np.pi, 150)
                x = radius*np.cos(angle)
                y = radius*np.sin(angle)
                z = np.zeros((len(x)))
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                           mode='lines',
                                           line=line,
                                           showlegend=False,
                                           ))
                return

            add_ring(fig, self.max_dist + 0.29, line=dict(width=5, color="black"))

            for rr in np.arange(0, self.max_dist + 0.29, ring_steps)[1:]:
                if isinstance(ring_steps, int):
                    rr = int(rr)
                add_ring(fig, rr, line=dict(color="lightgray"))
                x2, y2, z2 = spheric2cartesian([rr+ring_steps/5], [np.deg2rad(0)], [np.deg2rad(120)])
                fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='text',
                                           marker=dict(symbol=symbol, size=1, color='red'),
                                           text=[f'{rr}'],
                                           textfont=dict(color="black", size=16),
                                           textposition="middle center",
                                           showlegend=False,
                                           ))

            # add spokes
            for s_long in np.arange(0, 360, 45):
                x, y, z = spheric2cartesian([0.01, self.max_dist+0.29], [0.0, 0.0], [np.deg2rad(s_long), np.deg2rad(s_long)])

                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                           showlegend=False,
                                           line=dict(color='lightgray'),
                                           ))

                x2, y2, z2 = spheric2cartesian([0.01, self.max_dist+0.29+ring_steps/3], [0.0, 0.0], [np.deg2rad(s_long), np.deg2rad(s_long)])
                fig.add_trace(go.Scatter3d(x=[x2[-1]], y=[y2[-1]], z=[z2[-1]], mode='text',
                                           marker=dict(symbol=symbol, size=1, color='red'),
                                           text=[f'{s_long}°'],
                                           textfont=dict(color="black", size=16),
                                           textposition="middle center",
                                           showlegend=False,
                                           ))

        stitle = str(self.date.to_value('iso', subfmt='date_hm'))
        fig.update_layout(title=dict(text=stitle+' (UTC)', x=0.5, xref="paper", xanchor="center", font=dict(size=22, weight="normal"), automargin=True, yref='paper'),
                          legend=dict(itemsizing='constant', xref="paper", yref="paper", yanchor="top", y=1.0, xanchor="right", x=1.3, font=dict(size=16)))

        if not hide_logo:
            logo_x = 1.3
            logo_y = 0.05
            fig.add_annotation(x=logo_x, y=logo_y,
                               xref="paper", yref="paper",
                               xanchor="right",
                               yanchor="bottom",
                               font=dict(color="black", size=23, family="DejaVu Serif"),
                               text="Solar-MACH",
                               showarrow=False
                               )
            fig.add_annotation(x=logo_x, y=logo_y,
                               xref="paper", yref="paper",
                               xanchor="right",
                               yanchor="top",
                               font=dict(color="black", size=13, family="DejaVu Sans"),
                               text="https://solar-mach.github.io",
                               showarrow=False
                               )

        fig.update_layout(scene_aspectmode='cube',
                          scene=dict(xaxis=dict(title="X / AU", nticks=4, range=[-(self.max_dist+ring_steps), self.max_dist+ring_steps],),
                                     yaxis=dict(title="Y / AU", nticks=4, range=[-(self.max_dist+ring_steps), self.max_dist+ring_steps],),
                                     zaxis=dict(title="Z / AU", nticks=4, range=[-(self.max_dist+ring_steps), self.max_dist+ring_steps],),
                                     xaxis_tickfont=dict(weight=500, size=14),
                                     yaxis_tickfont=dict(weight=500, size=14),
                                     zaxis_tickfont=dict(weight=500, size=14),
                                     xaxis_title_font=dict(weight=500, size=16),
                                     yaxis_title_font=dict(weight=500, size=16),
                                     zaxis_title_font=dict(weight=500, size=16),
                                     ),
                          width=1024, height=1024,
                          margin=dict(r=20, l=10, b=10, t=10),
                          )

        config = {'toImageButtonOptions': {'format': 'png',  # one of png, svg, jpeg, webp
                                           'filename': 'Solar-MACH_3D_'+(stitle.replace(' ', '_')).replace(':', '-')+'_3D',
                                           # 'height': 700,
                                           # 'width': 700,
                                           # 'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor - doesn't seem to work; just scales the whole figure!
                                           }
                  }

        if _isstreamlit():
            # fig.update_layout(width=700, height=700)
            import streamlit as st
            # import streamlit.components.v1 as components
            # components.html(fig.to_html(include_mathjax='cdn'), height=700)
            st.plotly_chart(fig.update_layout(width=700, height=700),
                            theme=None,  # "streamlit",
                            use_container_width=True,
                            config=config)
        else:
            fig.show(config=config)

        if return_plot_object:
            return fig
        else:
            return


    def produce_pfss_footpoints_df(self, footpoints_dict:dict) -> None:
        """
        Produces a dataframe that contains the footpoints of 
        the pfss-extrapolated fieldlines. Attaches this dataframe to the class
        variable called 'pfss_footpoints'.

        If the input dictionary is somehow invalid for a dataframe, an empty dataframe
        will be initialized instead.

        Parameter:
        ----------
        footpoints_dict : {dict} A dictionary that contains lists of (longitude,latitude)
        pairs mapped by the object names.
        """

        try:
            df = pd.DataFrame(data=footpoints_dict)
        except ValueError as ve:
            print(f"Something went wrong with collecting photospheric footpoints to pfss_footpoints.\n({ve})")
            # An empty placeholder dataframe
            df = pd.DataFrame(columns=self.body_dict.keys())

        df.index.name = "Fieldline #"
        self.pfss_footpoints = df

def sc_distance(sc1, sc2, dtime):
    """
    Obtain absolute distance between two bodies in 3d for a given datetime.

    Parameters
    ----------
    sc1 : str
        Name of body 1, e.g., planet or spacecraft
    sc2 : str
        Name of body 2, e.g., planet or spacecraft
    dtime : datetime object or datetime-compatible str
        Date (and time) of distance determination

    Returns
    -------
    astropy.units.Quantity
        Absolute distance between body 1 and 2 in AU.
    """
    # parse datetime:
    if type(dtime) is str:
        try:
            obstime = parse_time(dtime)
        except ValueError:
            print(f"Unable to extract datetime from '{dtime}'. Please try a different format.")
            return np.nan*u.AU
    else:
        obstime = dtime

    # standardize body names (e.g. 'PSP' => 'Parker Solar Probe')
    try:
        sc1 = body_dict[sc1][1]
    except KeyError:
        pass
    #
    try:
        sc2 = body_dict[sc2][1]
    except KeyError:
        pass

    try:
        sc1_coord = get_horizons_coord(sc1, obstime, None)
    except (ValueError, RuntimeError):
        print(f"Unable to obtain position for '{sc1}' at {obstime}. Please try a different name or date.")
        return np.nan*u.AU
    #
    try:
        sc2_coord = get_horizons_coord(sc2, obstime, None)
    except (ValueError, RuntimeError):
        print(f"Unable to obtain position for '{sc2}' at {obstime}. Please try a different name or date.")
        return np.nan*u.AU

    return sc1_coord.separation_3d(sc2_coord)


def sto2car_sun(long, lat, dtime):
    """
    Converts heliographic Stonyhurst coordinates to heliographic Carrington coordinates for Sun as the observer.

    Parameters
    ----------
    long : float or array-like
        Longitude(s) in degrees in the Stonyhurst frame.
    lat : float or array-like
        Latitude(s) in degrees in the Stonyhurst frame.
    dtime : str or astropy.time.Time
        Observation time corresponding to the coordinates.

    Returns
    -------
    tuple
        A tuple containing:
            - Carrington longitude(s) in degrees (float or array-like)
            - Carrington latitude(s) in degrees (float or array-like)
    """ 
    coord = SkyCoord(long*u.deg, lat*u.deg, aconst.R_sun, frame=frames.HeliographicStonyhurst, obstime=dtime)
    coord_trans = coord.transform_to(frames.HeliographicCarrington(observer='Sun'))
    return coord_trans.lon.value, coord_trans.lat.value


def car2sto_sun(long, lat, dtime):
    """
    Converts heliographic Carrington coordinates to heliographic Stonyhurst coordinates for Sun as the observer.

    Parameters
    ----------
    long : float or array-like
        Longitude(s) in degrees in the Carrington frame.
    lat : float or array-like
        Latitude(s) in degrees in the Carrington frame.
    dtime : str or astropy.time.Time
        Observation time corresponding to the coordinates.

    Returns
    -------
    tuple
        A tuple containing:
            - Stonyhurst longitude(s) in degrees (float or array-like)
            - Stonyhurst latitude(s) in degrees (float or array-like)
    """
    coord = SkyCoord(long*u.deg, lat*u.deg, aconst.R_sun, frame=frames.HeliographicCarrington, observer='Sun', obstime=dtime)
    coord_trans = coord.transform_to(frames.HeliographicStonyhurst)
    return coord_trans.lon.value, coord_trans.lat.value


def _isstreamlit():
    """
    Function to check whether python code is run within streamlit

    Returns
    -------
    use_streamlit : boolean
        True if code is run within streamlit, else False
    """
    # https://discuss.streamlit.io/t/how-to-check-if-code-is-run-inside-streamlit-and-not-e-g-ipython/23439
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx(suppress_warning=True):
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit


if _isstreamlit():
    from stqdm import stqdm as tqdm
else:
    from tqdm.auto import tqdm
