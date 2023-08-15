# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pkg_resources import get_distribution, DistributionNotFound  # type: ignore
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

import copy
import dateutil.parser  # type: ignore
import math

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
body_dict.update(dict.fromkeys(['JUICE', 'Juice', -28], [-28, 'JUICE', 'violet']))
body_dict.update(dict.fromkeys(['Juno', 'JUNO', -61], [-61, 'Juno', 'orangered']))
body_dict.update(dict.fromkeys(['Jupiter', 599], [599, 'Jupiter', 'navy']))
body_dict.update(dict.fromkeys(['L1', 31], [31, 'SEMB-L1', 'black']))
body_dict.update(dict.fromkeys(['Mars', 499], [499, 'Mars', 'maroon']))
body_dict.update(dict.fromkeys(['Mars Express', -41], [-41, 'Mars Express', 'darkorange']))
body_dict.update(dict.fromkeys(['MAVEN', 'Maven', -202], [-202, 'MAVEN', 'brown']))
body_dict.update(dict.fromkeys(['Mercury', 199], [199, 'Mercury', 'darkturquoise']))
body_dict.update(dict.fromkeys(['MESSENGER', 'Messenger', -236], [-236, 'MESSENGER', 'olivedrab']))
body_dict.update(dict.fromkeys(['PSP', 'Parker Solar Probe', 'parkersolarprobe', 'ParkerSolarProbe', -96], [-96, 'Parker Solar Probe', 'purple']))
body_dict.update(dict.fromkeys(['Pioneer10', 'Pioneer 10', -23], [-23, 'Pioneer 10', 'teal']))
body_dict.update(dict.fromkeys(['Pioneer11', 'Pioneer 11', -24], [-24, 'Pioneer 11', 'darkblue']))
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


def get_sw_speed(body, dtime, trange=1, default_vsw=400.0):
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
        i.e. dtime +- trange (in hours). Default value 2.
    default_vsw : float
        Default solar wind bulk speed in km/s that is returned if no
        measurements can be obtained. Default value 400.0

    Returns
    -------
    float
        solar wind bulk speed in km/s
    """
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
    dataset['Solar Orbiter'] = amda_tree.Parameters.SolarOrbiter.PAS.L2.so_pas_momgr1.pas_momgr1_v_rtn_tot
    dataset['STEREO A'] = amda_tree.Parameters.STEREO.STEREO_A.PLASTIC.sta_l2_pla.vpbulk_sta
    dataset['STEREO B'] = amda_tree.Parameters.STEREO.STEREO_B.PLASTIC.stb_l2_pla.vpbulk_stb
    dataset['Wind'] = amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_vmag

    sw_key = dict(ACE='SW Bulk Speed')  # Solar Wind Bulk Speed [Vp]
    sw_key['Parker Solar Probe'] = 'psp_spc_vp_mom_nrm'  # Velocity vector magnitude
    sw_key['SOHO'] = 'Proton V'  # Proton speed, scalar
    sw_key['Solar Orbiter'] = 'pas_momgr1_v_rtn_tot'  # Velocity magnitude in RTN frame
    sw_key['STEREO A'] = 'vpbulk_sta'  # Scalar magnitude of the velocity in km/s
    sw_key['STEREO B'] = 'vpbulk_stb'  # Scalar magnitude of the velocity in km/s
    sw_key['Wind'] = 'wnd_swe_vmag'  # |v|

    if body in ['Earth', 'SEMB-L1']:
        print(f"Using 'ACE' measurements for '{body}'.")
        body = 'ACE'
    elif body not in dataset.keys():
        print(f"Body '{body}' not supported, assuming default Vsw value of {default_vsw} km/s.")
        return default_vsw

    if type(dtime) == str:
        try:
            dtime = dateutil.parser.parse(dtime)
        except dateutil.parser.ParserError:
            print(f"Unable to extract datetime from '{dtime}'. Assuming default Vsw value of {default_vsw} km/s.")
            return default_vsw

    try:
        df = spz.get_data(dataset[body], dtime-dt.timedelta(hours=trange), dtime+dt.timedelta(hours=trange)).to_dataframe()
        df = df[sw_key[body]].resample('1H').mean()
        # drop NaN entries:
        df.dropna(inplace=True)
        if len(df) > 0:
            idx = df[df.index.get_indexer([dtime], method='nearest')]
            return idx.values[0]
        else:
            print(f"No Vsw data found for '{body}' on {dtime}, assuming default Vsw value of {default_vsw} km/s.")
            return default_vsw
    except AttributeError:
        print(f"No Vsw data found for '{body}' on {dtime}, assuming default Vsw value of {default_vsw} km/s.")
        return default_vsw


class SolarMACH():
    """
    Class handling selected bodies

    Parameters
    ----------
    date: str
    body_list: list
        list of body keys to be used. Keys can be string of int.
    vsw_list: list, optional
        list of solar wind bulk speeds in km/s at the position of the different bodies. Must have the same length as body_list.
        If empty list, obtaining actual measurements is tried. If this is not successful, a default value defined by default_vsw is used.
    default_vsw: int or float, optional
        Solar wind bulk speed in km/s to be used if vsw_list is not defined and no vsw measurements could be obtained. By default 400.0.
    coord_sys: string, optional
        Defines the coordinate system used: 'Carrington' (default) or 'Stonyhurst'
    reference_long: float, optional
        Longitute of reference position at the Sun
    reference_lat: float, optional
        Latitude of referene position at the Sun
    """

    def __init__(self, date, body_list, vsw_list=[], reference_long=None, reference_lat=None, coord_sys='Carrington', default_vsw=400.0, **kwargs):
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

        self.date = date
        self.reference_long = reference_long
        self.reference_lat = reference_lat
        self.coord_sys = coord_sys

        pos_E = get_horizons_coord(399, self.date, None)  # (lon, lat, radius) in (deg, deg, AU)
        if coord_sys=='Carrington':
            self.pos_E = pos_E.transform_to(frames.HeliographicCarrington(observer='Sun'))
        elif coord_sys=='Stonyhurst':
            self.pos_E = pos_E

        # make deep copy of vsw_list bc. otherwise it doesn't get reset in a new init:
        vsw_list2 = copy.deepcopy(vsw_list)

        if len(vsw_list2) == 0:
            print('No solar wind speeds defined, trying to obtain measurements...')
            for body in body_list:
                vsw_list2.append(get_sw_speed(body=body, dtime=date, default_vsw=default_vsw))
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

                sep, alpha = self.backmapping(pos, reference_long, target_solar_radius=self.target_solar_radius, vsw=vsw_list2[i])
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
            except ValueError:
                print('')
                print('!!! No ephemeris for target "' + str(body) + '" for date ' + self.date)
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
                "Longitudinal separation between body's mangetic footpoint and reference_long"] = footp_longsep_list
            self.pfss_table.loc[len(self.pfss_table.index)] = ["Reference Point", self.reference_long, self.reference_lat, 1, np.NaN]
        if self.reference_lat is not None:
            self.coord_table['Latitudinal separation between body and reference_lat'] = latsep_list
            self.pfss_table.loc[self.pfss_table["Spacecraft/Body"]=="Reference Point", f"{coord_sys} latitude (°)"] = self.reference_lat

        # Does this still have a use?
        pass
        self.coord_table.style.set_properties(**{'text-align': 'left'})

        # reset sunpy log level to initial state
        log.setLevel(initial_log_level)

    def backmapping(self, body_pos, reference_long, target_solar_radius=1, vsw=400):
        """
        Determine the longitudinal separation angle of a given body and a given reference longitude

        Parameters
        ----------
        body_pos : astropy.coordinates.sky_coordinate.SkyCoord
            coordinates of the body
        reference_long: float
            Longitude of reference point at Sun to which we determine the longitudinal separation
        target_solar_radius: float
            Target solar radius to which to be backmapped. 0 corresponds to Sun's center, 1 to 1 solar radius, and e.g. 2.5 to the source surface.
        vsw: float
             solar wind speed (km/s) used to determine the position of the magnetic footpoint of the body. Default is 400.

        Returns
        -------
            sep: float
                longitudinal separation of body magnetic footpoint and reference longitude in degrees
            alpha: float
                backmapping angle
        """
        # AU = const.au / 1000  # km

        pos = body_pos
        lon = pos.lon.value
        lat = pos.lat.value
        # dist = pos.radius.value
        radius = pos.radius

        # take into account solar differential rotation wrt. latitude
        omega = self.solar_diff_rot(lat)
        # old:
        # omega = math.radians(360. / (25.38 * 24 * 60 * 60))  # rot-angle in rad/sec, sidereal period

        # tt = dist * AU / vsw
        # alpha = math.degrees(omega * tt)
        alpha = math.degrees(omega * (radius-target_solar_radius*aconst.R_sun).to(u.km).value / vsw * np.cos(np.deg2rad(lat)))

        if reference_long is not None:
            sep = (lon + alpha) - reference_long
            if sep > 180.:
                sep = sep - 360

            if sep < -180.:
                sep = 360 - abs(sep)
        else:
            sep = np.nan

        return sep, alpha

    def solar_diff_rot(self, lat):
        """
        Calculate solar differential rotation wrt. latitude,
        based on rLSQ method of Beljan et al. (2017),
        doi: 10.1051/0004-6361/201731047

        Parameters
        ----------
        lat : number (int, flotat)
            Heliographic latitude in degrees

        Returns
        -------
        numpy.float64
            Solar angular rotation in rad/sec
        """
        # (14.50-2.87*np.sin(np.deg2rad(lat))**2) defines degrees/day
        if self.diff_rot is False:
            lat = 0
        return np.radians((14.50-2.87*np.sin(np.deg2rad(lat))**2)/(24*60*60))

    def plot(self, plot_spirals=True,
             plot_sun_body_line=False,
             show_earth_centered_coord=False,
             reference_vsw=400,
             transparent=False,
             numbered_markers=False,
             return_plot_object=False,
             long_offset=270,
             outfile='',
             figsize=(12, 8),
             dpi=200,
             long_sector=None,
             long_sector_vsw=None,
             long_sector_color='red',
             long_sector_alpha=0.5,
             background_spirals=None,
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
        plot_spirals: bool
            if True, the magnetic field lines connecting the bodies with the Sun are plotted
        plot_sun_body_line: bool
            if True, straight lines connecting the bodies with the Sun are plotted
        show_earth_centered_coord: bool
            Deprecated! With the introduction of coord_sys in class SolarMACH() this function is redundant and not functional any more!
        reference_vsw: int
            if defined, defines solar wind speed for reference. if not defined, 400 km/s is used
        transparent: bool
            if True, output image has transparent background
        numbered_markers: bool
            if True, body markers contain numbers for better identification
        return_plot_object: bool
            if True, figure and axis object of matplotib are returned, allowing further adjustments to the figure
        long_offset: int or float
            longitudinal offset for polar plot; defines where Earth's longitude is (by default 270, i.e., at "6 o'clock")
        outfile: string
            if provided, the plot is saved with outfile as filename
        long_sector: list of 2 numbers, optional
            Start and stop longitude of a shaded area; e.g. [350, 20] to get a cone from 350 to 20 degree longitude (for long_sector_vsw=None).
        long_sector_vsw: list of 2 numbers, optional
            Solar wind speed used to calculate Parker spirals (at start and stop longitude provided by long_sector) between which a reference cone should be drawn; e.g. [400, 400] to assume for both edges of the fill area a Parker spiral produced by solar wind speeds of 400 km/s. If None, instead of Parker spirals straight lines are used, i.e. a simple cone wil be plotted. By default None.
        long_sector_color: string, optional
            String defining the matplotlib color used for the shading defined by long_sector. By default 'red'.
        long_sector_alpha: float, optional
            Float between 0.0 and 1.0, defining the matplotlib alpha used for the shading defined by long_sector. By default 0.5.W
        background_spirals: list of 2 numbers (and 3 optional strings), optional
            If defined, plot evenly distributed Parker spirals over 360°. background_spirals[0] defines the number of spirals, background_spirals[1] the solar wind speed in km/s used for their calculation. background_spirals[2], background_spirals[3], and background_spirals[4] optionally change the plotting line style, color, and alpha setting, respectively (default values ':', 'grey', and 0.1). Full example that plots 12 spirals (i.e., every 30°) using a solar wind speed of 400 km/s with solid red lines with alpha=0.2: background_spirals=[12, 400, '-', 'red', 0.2]
        """
        hide_logo = False  # optional later keyword to hide logo on figure
        AU = const.au / 1000  # km

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

        if test_plotly:
            import plotly.graph_objects as go
            pfig = go.Figure()

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
            omega = self.solar_diff_rot(body_lat)
            # old:
            # omega = np.radians(360. / (25.38 * 24 * 60 * 60))  # solar rot-angle in rad/sec, sidereal period

            # plot body positions
            if numbered_markers:
                ax.plot(np.deg2rad(body_long), dist_body*np.cos(np.deg2rad(body_lat)), 'o', ms=15, color=body_color, label=body_lab)
                ax.annotate(i+1, xy=(np.deg2rad(body_long), dist_body*np.cos(np.deg2rad(body_lat))), color='white',
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
                alpha_body = np.deg2rad(body_long) + omega / (body_vsw / AU) * (dist_body - r_array) * np.cos(np.deg2rad(body_lat))
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

                if numbered_markers:
                    str_number = f'<b>{i+1}</b>'
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
            omega_ref = self.solar_diff_rot(ref_lat)

            # old eq. for alpha_ref contained redundant dist_e variable:
            # alpha_ref = np.deg2rad(delta_ref) + omega_ref / (reference_vsw / AU) * (dist_e / AU - r_array) - (omega_ref / (reference_vsw / AU) * (dist_e / AU))
            # alpha_ref = np.deg2rad(delta_ref) + omega_ref / (reference_vsw / AU) * (aconst.R_sun.to(u.AU).value - r_array)
            alpha_ref = np.deg2rad(delta_ref) + omega_ref / (reference_vsw / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(ref_lat))

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
            if numbered_markers:
                for i, body_id in enumerate(self.body_dict):
                    if self.reference_long is not None:
                        x_offset_ref = -0.035  # 0.004
                        y_offset_ref = 0.081
                        y_offset_per_i = -0.051
                    else:
                        x_offset_ref = 0.0
                        y_offset_ref = 0.0
                        y_offset_per_i = -0.0475
                    x_offset = -0.11  # 0.05
                    y_offset = 0.124  # -0.0064
                    pfig.add_annotation(text=f'<b>{i+1}</b>', xref="paper", yref="paper", xanchor="center", yanchor="top",
                                        x=test_plotly_legend[0]+x_offset+x_offset_ref, y=test_plotly_legend[1]+y_offset+y_offset_ref+y_offset_per_i*i,
                                        showarrow=False, font=dict(color="white", size=14))

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
                # st.plotly_chart(pfig, theme="streamlit")
                st.components.v1.html(pfig.to_html(include_mathjax='cdn'), height=500)
            else:
                pfig.show()

        if long_sector is not None:
            if type(long_sector) == list and np.array(long_sector).ndim==1:
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
                omega_ref1 = self.solar_diff_rot(long_sector_lat[0])
                omega_ref2 = self.solar_diff_rot(long_sector_lat[1])

                # Build an r_array for the second spiral for while loop to iterate forwards
                r_array2 = np.copy(r_array)

                if t_long_sector_vsw is not None:
                    # Calculate the first spiral's angles along r
                    alpha_ref1 = np.deg2rad(delta_ref1) + omega_ref1 / (t_long_sector_vsw[0] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(long_sector_lat[0]))
                    alpha_ref2 = np.deg2rad(delta_ref2) + omega_ref2 / (t_long_sector_vsw[1] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array2) * np.cos(np.deg2rad(long_sector_lat[1]))

                    # Save the last angle as a starting point for reference for the while loop
                    alpha_init = alpha_ref2[-1]

                    # Check that reference angle of the first loop is ahead
                    if alpha_ref1[-1] > alpha_ref2[-1]:
                        alpha_ref1_comp = alpha_ref1[-1] - 2*np.pi
                    else:
                        alpha_ref1_comp = alpha_ref1[-1]

                    # While the second spiral is behind the first spiral in angle, extend the second spiral
                    while alpha_ref2[-1] > alpha_ref1_comp:
                        r_array2 = np.append(r_array2, r_array2[-1] + 0.1)
                        alpha_ref2 = np.append(alpha_ref2, np.deg2rad(delta_ref2) + omega_ref2 / (t_long_sector_vsw[1] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array2[-1]) * np.cos(np.deg2rad(long_sector_lat[1])))

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
                y2 = c2.get_ydata()

                # Check that plotted are is between the two spirals, and do not fill after potential crossing
                clause1 = x1 < x2
                clause2 = alpha_ref1[clause1] < alpha_ref2[clause1]

                # Take only the points that fill the above clauses
                y1_fill = y1[clause1][clause2]
                x1_fill = x1[clause1][clause2]
                x2_fill = x2[clause1][clause2]

                plt.fill_betweenx(y1_fill, x1_fill, x2_fill, lw=0, color=t_long_sector_color, alpha=t_long_sector_alpha)

        if background_spirals is not None:
            if type(background_spirals) == list and len(background_spirals)>=2:
                # maybe later add option to have a non-zero latitude, so that the field lines are out of the ecliptic
                background_spirals_lat = 0
                # take into account solar differential rotation wrt. latitude
                omega_ref = self.solar_diff_rot(background_spirals_lat)

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
                    alpha_ref = np.deg2rad(l) + omega_ref / (background_spirals[1] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(background_spirals_lat))
                    ax.plot(alpha_ref, r_array * np.cos(np.deg2rad(background_spirals_lat)), ls=background_spirals_ls, c=background_spirals_c, alpha=background_spirals_alpha)
            else:
                print("Ill-defined 'background_spirals'. It should be a list with at least 2 elements defining the number of field lines and the solar wind speed used for them in km/s; e.g. 'background_spirals=[10, 400]'")

        leg1 = ax.legend(loc=(1.2, 0.7), fontsize=13)

        if numbered_markers:
            offset = matplotlib.text.OffsetFrom(leg1, (0.0, 1.0))
            for i, body_id in enumerate(self.body_dict):
                yoffset = i*18.7  # 18.5 19.5
                ax.annotate(i+1, xy=(1, 1), xytext=(18.3, -11-yoffset), color='white',
                            fontsize="small", weight='heavy', textcoords=offset,
                            horizontalalignment='center',
                            verticalalignment='center', zorder=100)

        if self.reference_long is not None:
            def legend_arrow(width, height, **_):
                return mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True,
                                           head_width=0.75 * height)

            leg2 = ax.legend([ref_arr], ['reference long.'], loc=(1.2, 0.6),
                             handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=legend_arrow), },
                             fontsize=13)
            ax.add_artist(leg1)

        # replace 'SEMB-L1' in legend with 'L1' if present
        for text in leg1.get_texts():
            if text.get_text() == 'SEMB-L1':
                text.set_text('L1')

        # for Stonyhurst, define the longitude from -180 to 180 (instead of 0 to 360)
        # NB: this remove the rgridlines for unknown reasons! deactivated for now
        # if self.coord_sys=='Stonyhurst':
        #     ax.set_xticks(np.pi/180. * np.linspace(180, -180, 8, endpoint=False))
        #     ax.set_thetalim(-np.pi, np.pi)

        rlabel_pos = E_long + 120
        ax.set_rlabel_position(rlabel_pos)
        ax.set_theta_offset(np.deg2rad(long_offset - E_long))
        ax.set_rmax(self.max_dist + 0.3)
        ax.set_rmin(0.01)
        ax.yaxis.get_major_locator().base.set_params(nbins=4)
        circle = plt.Circle((0., 0.),
                            self.max_dist + 0.29,
                            transform=ax.transData._b,
                            edgecolor="k",
                            facecolor=None,
                            fill=False, lw=2)
        ax.add_patch(circle)

        # manually plot r-grid lines with different resolution depending on maximum distance body
        if self.max_dist < 2:
            ax.set_rgrids(np.arange(0, self.max_dist + 0.29, 0.5)[1:], angle=rlabel_pos)
        else:
            if self.max_dist < 10:
                ax.set_rgrids(np.arange(0, self.max_dist + 0.29, 1.0)[1:], angle=rlabel_pos)

        ax.set_title(self.date + '\n', pad=60)

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
            ax.text(0.94, 0.16, 'Solar-MACH',
                    fontfamily='DejaVu Serif', fontsize=28,
                    ha='right', va='bottom', transform=fig.transFigure)
            ax.text(0.94, 0.12, 'https://solar-mach.github.io',
                    fontfamily='DejaVu Sans', fontsize=18,
                    ha='right', va='bottom', transform=fig.transFigure)

        if transparent:
            fig.patch.set_alpha(0.0)

        if outfile != '':
            plt.savefig(outfile, bbox_inches="tight")
        # st.pyplot(fig, dpi=200)

        # restore initial rcParams that have been saved at the beginning of this function:
        plt.rcParams.update(initial_rcparams)

        # if using streamlit, send plot to streamlit output, else call plt.show()
        if _isstreamlit():
            import streamlit as st
            st.pyplot(fig)  # , dpi=200)
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
                  rss=2.5,
                  pfss_solution=None,
                  figsize=(15, 10),
                  dpi=200,
                  return_plot_object=False,
                  vary=False, n_varies=1,
                  long_offset=270,
                  reference_vsw=400.,
                  numbered_markers=False,
                  plot_spirals=True,
                  long_sector=None,
                  long_sector_vsw=None,
                  long_sector_color=None,
                  hide_logo=False,
                  outfile=''):
        """
        Produces a figure of the heliosphere in polar coordinates with logarithmic r-axis outside the pfss.
        Also tracks an open field line down to photosphere given a point on the pfss.

        rss : float
                source surface height of the potential field in solar radii. default 2.5
        pfss_solution : .fits
                the pfss model calculated from magnetographic maps
        reference_longitude : int or floar
                draws an arrow pointing radially outward, input in degrees

        """

        if not pfss_solution:
            raise Exception("A PFSS solution is required for solar magnetic field extrapolation!")

        # Constants
        AU = const.au / 1000  # km
        sun_radius = aconst.R_sun.value  # meters

        # r_scaler scales distances from astronomical units to solar radii. unit = [solar radii / AU]
        r_scaler = (AU*1000)/sun_radius

        # carrington longitude of the Earth
        E_long = self.pos_E.lon.value

        # save inital rcParams and update some of them:
        initial_rcparams = plt.rcParams.copy()
        plt.rcParams['axes.linewidth'] = 2.5
        plt.rcParams['font.size'] = 15
        plt.rcParams['agg.path.chunksize'] = 20000

        # init the figure and axes
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize, dpi=dpi)

        # maximum distance anything will be plotted
        r_max = r_scaler * 5  # 5 AU = 1075 in units of solar radii

        # setting the title
        ax.set_title(self.date + '\n', pad=30, fontsize=26)

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

        # The radial coordinates for reference parker spiral (plot even outside the figure boundaries to avert visual bugs)
        reference_array = np.linspace(rss, r_max+200, int(1e3))

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
            omega = self.solar_diff_rot(body_lat)

            # The radial coordinates (outside source surface) for each object
            r_array = np.linspace(r_scaler*dist_body*np.cos(np.deg2rad(body_lat)), rss, 1000)

            # plot body positions
            if numbered_markers:
                ax.plot(np.deg2rad(body_long), r_scaler*dist_body*np.cos(np.deg2rad(body_lat)), 'o', ms=15, color=body_color, label=body_lab)
                ax.annotate(i+1, xy=(np.deg2rad(body_long), r_scaler*dist_body*np.cos(np.deg2rad(body_lat))), color='white',
                            fontsize="small", weight='heavy',
                            horizontalalignment='center',
                            verticalalignment='center')
            else:
                ax.plot(np.deg2rad(body_long), r_scaler*dist_body*np.cos(np.deg2rad(body_lat)), 's', color=body_color, label=body_lab)

            # Plotting the spirals
            if plot_spirals:

                # The angular coordinates are calculated here
                # alpha = longitude + (omega)*(distance-r)/sw
                alpha_body = np.deg2rad(body_long) + omega / (1000*body_vsw / sun_radius) * (r_scaler*dist_body - r_array) * np.cos(np.deg2rad(body_lat))
                ax.plot(alpha_body, r_array * np.cos(np.deg2rad(body_lat)), color=body_color)

            # Acquire an array of (r,lon,lat) coordinates of the open field lines under the pfss
            # based on the footpoint(s) of the sc
            if vary:

                # Triplets contain 35 tuples of (r,lon,lat)
                fline_triplets, fline_objects, varyfline_triplets, varyfline_objects = vary_flines(alpha_body[-1], np.deg2rad(body_lat), pfss_solution, n_varies, rss)

                # Collect field line objects to a list
                self.fieldlines.append(fline_objects[0])
                for varyfline in varyfline_objects:
                    self.fieldlines.append(varyfline)

                # Plot the color coded varied field lines
                for triplet in varyfline_triplets:
                    v_fl_r   = triplet[0]
                    v_fl_lon = triplet[1]
                    v_fl_lat = triplet[2]

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
            fieldline_polarities.append(int(fline_objects[0].polarity))

            # Save Earth's magnetic footpoint for later comparison:
            if body_lab == "Earth":
                earth_footpoint = (fl_lon[0], fl_lat[0])

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
            omega_ref = self.solar_diff_rot(ref_lat)

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

            # Also init extreme values for the longitudinal span of the uptracked flux tube
            self.reference_long_min, self.reference_long_max = 360, 0

            # Boolean switch to keep track what kind of arrow/spiral to draw for reference point
            open_mag_flux_near_ref_point = False

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

                # Collect the longitudinal extreme values from the uptracked fluxtube at the source surface height
                if ref_vary.coords.lon.value[idx] < self.reference_long_min:
                    self.reference_long_min = ref_vary.coords.lon.value[idx]
                if ref_vary.coords.lon.value[idx] > self.reference_long_max:
                    self.reference_long_max = ref_vary.coords.lon.value[idx]

            arrow_dist = rss-0.80
            if open_mag_flux_near_ref_point:
                ref_arr = plt.arrow(np.deg2rad(self.reference_long_min), 1, 0, arrow_dist, head_width=0.05, head_length=0.2, edgecolor='black',
                                    facecolor='black', lw=0, zorder=7, overhang=0.1)
                ref_arr = plt.arrow(np.deg2rad(self.reference_long_max), 1, 0, arrow_dist, head_width=0.05, head_length=0.2, edgecolor='black',
                                    facecolor='black', lw=0, zorder=7, overhang=0.1)

                reference_legend_label = f"reference long.\nsector:\n({np.round(self.reference_long_min,1)}, {np.round(self.reference_long_max,1)})"

            else:
                # Set the reach of the flux tube to nan, since it doesn't even reach up to the source surface
                self.reference_long_min, self.reference_long_max = np.NaN, np.NaN

                ref_arr = plt.arrow(np.deg2rad(self.reference_long), 1, 0, arrow_dist, head_width=0.1, head_length=0.5, edgecolor='black',
                                    facecolor='black', lw=1., zorder=7, overhang=0.5)

                reference_legend_label = f"reference long.\n{self.reference_long} deg"

            # These two spirals and the space between them gets drawn only if we plot spirals and open magnetic flux was found near the ref point
            if plot_spirals and open_mag_flux_near_ref_point:

                # Calculate spirals for the flux tube boundaries
                alpha_ref_min = np.deg2rad(self.reference_long_min) + omega_ref / (1000*reference_vsw / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(ref_lat))
                alpha_ref_max = np.deg2rad(self.reference_long_max) + omega_ref / (1000*reference_vsw / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(ref_lat))

                # Plot the spirals
                min_edge = plt.polar(alpha_ref_min, reference_array * np.cos(np.deg2rad(ref_lat)), lw=0.7, color="grey", alpha=0.45)[0]
                max_edge = plt.polar(alpha_ref_max, reference_array * np.cos(np.deg2rad(ref_lat)), lw=0.7, color="grey", alpha=0.45)[0]

                # Extract 'x' and 'y' values
                x1 = min_edge.get_xdata()
                y1 = min_edge.get_ydata()
                x2 = max_edge.get_xdata()

                plt.fill_betweenx(y1, x1, x2, lw=0, color="grey", alpha=0.35)

            # Here we plot spirals but open magnetic flux was not found -> draw only one spiral
            if plot_spirals:

                alpha_ref_single = np.deg2rad(self.reference_long) + omega_ref / (1000*reference_vsw / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(ref_lat))
                ax.plot(alpha_ref_single, reference_array * np.cos(np.deg2rad(ref_lat)), color="grey")

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
                omega_ref1 = self.solar_diff_rot(long_sector_lat[0])
                omega_ref2 = self.solar_diff_rot(long_sector_lat[1])

                if long_sector_vsw is not None:

                    # Calculate the spirals' angles along r
                    alpha_ref1 = np.deg2rad(delta_ref1) + omega_ref1 / (1000*long_sector_vsw[0] / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(long_sector_lat[0]))
                    alpha_ref2 = np.deg2rad(delta_ref2) + omega_ref2 / (1000*long_sector_vsw[1] / sun_radius) * (rss - reference_array) * np.cos(np.deg2rad(long_sector_lat[1]))

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
                        alpha_ref2 = np.append(alpha_ref2, np.deg2rad(delta_ref2) + omega_ref2 / (1000*long_sector_vsw[1] / sun_radius) * (rss - reference_array2[-1]) * np.cos(np.deg2rad(long_sector_lat[1])))

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
                y2 = c2.get_ydata()

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

        leg1 = ax.legend(loc=(1.05, 0.8), fontsize=13)

        if numbered_markers:
            offset = matplotlib.text.OffsetFrom(leg1, (0.0, 1.0))
            for i, body_id in enumerate(self.body_dict):
                yoffset = i*18.7  # 18.5 19.5
                ax.annotate(i+1, xy=(1, 1), xytext=(18.3, -11-yoffset), color='white',
                            fontsize="small", weight='heavy', textcoords=offset,
                            horizontalalignment='center',
                            verticalalignment='center', zorder=100)

        if self.reference_long:
            def legend_arrow(width, height, **_):
                return mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True,
                                           head_width=0.75 * height)

            leg2 = ax.legend([ref_arr], [reference_legend_label], loc=(1.05, 0.6),
                             handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=legend_arrow), },
                             fontsize=15)
            ax.add_artist(leg1)

        # replace 'SEMB-L1' in legend with 'L1' if present
        for text in leg1.get_texts():
            if text.get_text() == 'SEMB-L1':
                text.set_text('L1')

        # for Stonyhurst, define the longitude from -180 to 180 (instead of 0 to 360)
        # NB: this remove the rgridlines for unknown reasons! deactivated for now
        # if self.coord_sys=='Stonyhurst':
        #     ax.set_xticks(np.pi/180. * np.linspace(180, -180, 8, endpoint=False))
        #     ax.set_thetalim(-np.pi, np.pi)

        # Spin the angular coordinate so that earth is at 6 o'clock
        ax.set_theta_offset(np.deg2rad(long_offset - E_long))

        # For some reason we need to specify 'ylim' here
        ax.set_ylim(0, r_max)
        ax.set_rscale('symlog', linthresh=rss)
        ax.set_rmax(r_max)
        ax.set_rticks([1.0, rss, 10.0, 100.0])
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.tick_params(which='major', labelsize=22,)

        rlabels = ['1', str(rss), r'$10^1$', r'$10^2$']
        ax.set_yticklabels(rlabels)

        # Drawing a circle around the plot, because sometimes for unkown reason the plot boundary is not drawn.
        # Here circle_radius corresponds to the boundary of the plot, and it was empirically found.
        circle_radius = 9.35

        circle = plt.Circle((0., 0.),
                            circle_radius,
                            transform=ax.transData._b,
                            edgecolor="k",
                            facecolor=None,
                            fill=False, lw=2)
        ax.add_patch(circle)

        # Cut off unnecessary margins from the plot
        plt.tight_layout()

        if not hide_logo:
            txt_x_begin, txt_y_begin = 0.98, 0.16
            ax.text(txt_x_begin, txt_y_begin, 'Solar-MACH',
                    fontfamily='DejaVu Serif', fontsize=28,
                    ha='right', va='bottom', transform=fig.transFigure)
            ax.text(txt_x_begin, txt_y_begin-0.04, 'https://solar-mach.github.io',
                    fontfamily='DejaVu Sans', fontsize=18,
                    ha='right', va='bottom', transform=fig.transFigure)

        # Create the colorbar displaying values of the last fieldline plotted
        cb = fig.colorbar(fieldline, ax=ax, location="left", anchor=(1.4, 1.2), shrink=0.6)

        # Colorbar is the last object created -> it is the final index in the list of axes
        cb_ax = fig.axes[-1]
        cb_ax.set_ylabel('Heliographic latitude [deg]', fontsize=20)

        # Add footpoints, magnetic polarities and the reach of reference_long flux tube to PFSS_table
        if self.reference_long:
            photospheric_footpoints.append(self.reference_long)
            fieldline_polarities.append(ref_objects[0].polarity)
            self.pfss_table["Reference flux tube lon range"] = [np.NaN if i<len(self.body_dict) else (self.reference_long_min, self.reference_long_max) for i in range(len(self.body_dict)+1)]

        self.pfss_table["Magnetic footpoint (PFSS)"] = photospheric_footpoints
        self.pfss_table["Magnetic polarity"] = fieldline_polarities

        # Update solar wind speed to the reference point
        if reference_vsw:
            self.pfss_table.loc[self.pfss_table["Spacecraft/Body"]=="Reference_point", "Vsw"] = reference_vsw

        if outfile != '':
            plt.savefig(outfile, bbox_inches="tight")

        # if using streamlit, send plot to streamlit output, else call plt.show()
        if _isstreamlit():
            import streamlit as st
            st.pyplot(fig)  # , dpi=200)
        else:
            plt.show()

        if return_plot_object:
            return fig, ax

    def pfss_3d(self, active_area=(None, None, None, None), color_code='object'):
        """
        https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html
        https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
        https://plotly.com/python-api-reference/plotly.graph_objects.html
        https://fslab.org/XPlot/chart/plotly-3d-line-plots.html

        Opens up an interactive 3d-plot of the Sun and FieldLine objects. Also shows the flare area if provided.

        params
        -------
        active_area: (lonmax, lonmin, latmax, latmin)
                    A tuple of the max and min of longitude and latitude in degrees

        color_code: str
                    Choose either 'polarity' or 'object'
        """

        import plotly.express as px
        import plotly.graph_objects as go
        from astropy.constants import R_sun
        from plotly.graph_objs.scatter3d import Line

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
                                    line=Line(color='black', width=5.5),
                                    name="Solar equator"
                                    )

        traces.append(equator_line)

        # create the figure
        fig = go.Figure(data=traces)

        # additional figure settings, like aspect mode, extreme values of axes etc...
        fig.update_layout(scene_aspectmode='cube')
        fig.update_layout(scene=dict(
                          xaxis=dict(nticks=4, range=[-2.5, 2.5],),
                          yaxis=dict(nticks=4, range=[-2.5, 2.5],),
                          zaxis=dict(nticks=4, range=[-2.5, 2.5],),),
                          width=1280, height=720,
                          margin=dict(r=20, l=10, b=10, t=10))

        if _isstreamlit():
            fig.update_layout(width=700, height=700)
            import streamlit as st
            # st.plotly_chart(pfig, theme="streamlit")
            st.components.v1.html(fig.to_html(include_mathjax='cdn'), height=700)
        else:
            fig.show()

        return


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
        if not get_script_run_ctx():
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
