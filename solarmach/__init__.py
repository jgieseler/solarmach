# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

import math
from copy import deepcopy

import astropy.constants as aconst
import astropy.units as u
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

# pd.options.display.max_rows = None
# pd.options.display.float_format = '{:.1f}'.format
# if needed, rather use the following to have the desired display:
"""
with pd.option_context('display.float_format', '{:0.2f}'.format):
    display(df)
"""


# initialize the body dictionary
body_dict = dict.fromkeys(['Mercury', 199], [199, 'Mercury', 'darkturquoise'])
body_dict.update(dict.fromkeys(['Venus', 299], [299, 'Venus', 'darkorchid']))
body_dict.update(dict.fromkeys(['Earth', 'EARTH', 'earth', 399], [399, 'Earth', 'green']))
body_dict.update(dict.fromkeys(['Mars', 499], [499, 'Mars', 'maroon']))
body_dict.update(dict.fromkeys(['Jupiter', 599], [599, 'Jupiter', 'navy']))

body_dict.update(dict.fromkeys(['L1', 31], [31, 'SEMB-L1', 'black']))
body_dict.update(dict.fromkeys(['ACE', 'Advanced Composition Explorer', -92], [-92, 'ACE', 'dimgrey']))

body_dict.update(dict.fromkeys(['STEREO B', 'STEREO-B', 'STB', 'stb', -235], [-235, 'STEREO B', 'b']))
body_dict.update(dict.fromkeys(['STEREO A', 'STEREO-A', 'STA', 'sta', -234], [-234, 'STEREO A', 'red']))
body_dict.update(dict.fromkeys(['SOHO', 'soho', -21], [-21, 'SOHO', 'darkgreen']))
body_dict.update(dict.fromkeys(['Solar Orbiter', 'SolO', 'solarorbiter', 'SolarOrbiter', -144], [-144, 'Solar Orbiter', 'dodgerblue']))
body_dict.update(dict.fromkeys(['PSP', 'Parker Solar Probe', 'parkersolarprobe', 'ParkerSolarProbe', -96], [-96, 'Parker Solar Probe', 'purple']))
body_dict.update(dict.fromkeys(['BepiColombo', 'Bepi Colombo', 'Bepi', 'MPO', -121], [-121, 'BepiColombo', 'orange']))
body_dict.update(dict.fromkeys(['MAVEN', -202], [-202, 'MAVEN', 'brown']))
body_dict.update(dict.fromkeys(['Mars Express', -41], [-41, 'Mars Express', 'darkorange']))
body_dict.update(dict.fromkeys(['MESSENGER', -236], [-236, 'MESSENGER', 'olivedrab']))
body_dict.update(dict.fromkeys(['JUICE', -28], [-28, 'JUICE', 'violet']))
body_dict.update(dict.fromkeys(['Juno', -61], [-61, 'Juno', 'orangered']))
body_dict.update(dict.fromkeys(['Cassini', -82], [-82, 'Cassini', 'mediumvioletred']))
body_dict.update(dict.fromkeys(['Rosetta', -226], [-226, 'Rosetta', 'blueviolet']))
body_dict.update(dict.fromkeys(['Pioneer10', -23], [-23, 'Pioneer 10', 'teal']))
body_dict.update(dict.fromkeys(['Pioneer11', -24], [-24, 'Pioneer 11', 'darkblue']))
body_dict.update(dict.fromkeys(['Ulysses', -55], [-55, 'Ulysses', 'dimgray']))
body_dict.update(dict.fromkeys(['Voyager1', -31], [-31, 'Voyager 1', 'darkred']))
body_dict.update(dict.fromkeys(['Voyager2', -32], [-32, 'Voyager 2', 'midnightblue']))


def print_body_list():
    """
    prints a selection of body keys and the corresponding body names which may be provided to the
    SolarMACH class
    """
    # print('Please visit https://ssd.jpl.nasa.gov/horizons.cgi?s_target=1#top for a complete list of available bodies')
    data = pd.DataFrame\
        .from_dict(body_dict, orient='index', columns=['ID', 'Body', 'Color'])\
        .drop(columns=['ID', 'Color'])\
        .drop_duplicates()
    data.index.name = 'Key'
    return data


class SolarMACH():
    """
    Class which handles the selected bodies

    Parameters
    ----------
    date: str
    body_list: list
        list of body keys to be used. Keys can be string of int.
    vsw_list: list, optional
        list of solar wind speeds at the position of the different bodies. Must have the same length as body_list.
        Default is an epmty list leading to vsw=400km/s used for every body.
    coord_sys: string, optional
        Defines the coordinate system used: 'Carrington' (default) or 'Stonyhurst'
    reference_long: float, optional
        Longitute of reference position at the Sun
    reference_lat: float, optional
        Latitude of referene position at the Sun
    """

    def __init__(self, date, body_list, vsw_list=[], reference_long=None, reference_lat=None, coord_sys='Carrington', **kwargs):
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
        bodies = deepcopy(body_dict)

        if coord_sys.lower().startswith('car'):
            coord_sys = 'Carrington'
        if coord_sys.lower().startswith('sto') or coord_sys.lower() == 'Earth':
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

        if len(vsw_list) == 0:
            vsw_list = np.zeros(len(body_list)) + 400

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
                bodies[body_id].append(vsw_list[i])

                longsep_E = pos.lon.value - self.pos_E.lon.value
                if longsep_E > 180:
                    longsep_E = longsep_E - 360.
                latsep_E = pos.lat.value - self.pos_E.lat.value

                body_lon_list.append(pos.lon.value)
                body_lat_list.append(pos.lat.value)
                body_dist_list.append(pos.radius.value)
                longsep_E_list.append(longsep_E)
                latsep_E_list.append(latsep_E)

                body_vsw_list.append(vsw_list[i])

                sep, alpha = self.backmapping(pos, reference_long, target_solar_radius=self.target_solar_radius, vsw=vsw_list[i])
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

        if self.reference_long is not None:
            self.coord_table['Longitudinal separation between body and reference_long'] = longsep_list
            self.coord_table[
                "Longitudinal separation between body's mangetic footpoint and reference_long"] = footp_longsep_list
        if self.reference_lat is not None:
            self.coord_table['Latitudinal separation between body and reference_lat'] = latsep_list

        # Does this still have a use?
        pass
        self.coord_table.style.set_properties(**{'text-align': 'left'})

        # reset sunpy log level to initial state
        log.setLevel(initial_log_level)

    def backmapping(self, body_pos, reference_long, target_solar_radius=1, vsw=400):
        """
        Determine the longitudinal separation angle of a given spacecraft and a given reference longitude

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

        out:
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
             background_spirals=None):
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
        r_array = np.arange(0.007, (self.max_dist*3)/np.cos(np.deg2rad(self.max_dist_lat)) + 0.3, 0.001)
        # take into account solar differential rotation wrt. latitude. Thus move calculation of omega to the per-body section below
        # omega = np.radians(360. / (25.38 * 24 * 60 * 60))  # solar rot-angle in rad/sec, sidereal period

        E_long = self.pos_E.lon.value

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

            if plot_spirals:
                ax.plot(alpha_ref, r_array * np.cos(np.deg2rad(ref_lat)), '--k', label=f'field line connecting to\nref. long. (vsw={reference_vsw} km/s)')

        if long_sector is not None:
            if type(long_sector) == list and len(long_sector)==2:
                # long_sector_width = abs(180 - abs(abs(self.long_sector[0] - self.long_sector[1]) - 180))
                # cone_dist = self.max_dist+0.3
                # plt.bar(np.deg2rad(self.long_sector[0]), cone_dist, width=np.deg2rad(long_sector_width), align='edge', bottom=0.0, color=self.long_sector_color, alpha=0.5)

                delta_ref1 = long_sector[0]
                if delta_ref1 < 0.:
                    delta_ref1 = delta_ref1 + 360.
                delta_ref2 = long_sector[1]
                if delta_ref2 < 0.:
                    delta_ref2 = delta_ref2 + 360.

                long_sector_lat = [0, 0]  # maybe later add option to have different latitudes, so that the long_sector plane is out of the ecliptic
                # take into account solar differential rotation wrt. latitude
                omega_ref1 = self.solar_diff_rot(long_sector_lat[0])
                omega_ref2 = self.solar_diff_rot(long_sector_lat[1])

                if long_sector_vsw is not None:
                    alpha_ref1 = np.deg2rad(delta_ref1) + omega_ref1 / (long_sector_vsw[0] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(long_sector_lat[0]))
                    alpha_ref2 = np.deg2rad(delta_ref2) + omega_ref2 / (long_sector_vsw[1] / AU) * (self.target_solar_radius*aconst.R_sun.to(u.AU).value - r_array) * np.cos(np.deg2rad(long_sector_lat[1]))
                else:
                    # if no solar wind speeds for Parker spirals are provided, use straight lines:
                    alpha_ref1 = [np.deg2rad(delta_ref1)] * len(r_array)
                    alpha_ref2 = [np.deg2rad(delta_ref2)] * len(r_array)

                c1 = plt.polar(alpha_ref1, r_array * np.cos(np.deg2rad(long_sector_lat[0])), lw=0, color=long_sector_color, alpha=0.5)[0]
                x1 = c1.get_xdata()
                y1 = c1.get_ydata()
                c2 = plt.polar(alpha_ref2, r_array * np.cos(np.deg2rad(long_sector_lat[1])), lw=0, color=long_sector_color, alpha=0.5)[0]
                x2 = c2.get_xdata()
                y2 = c2.get_ydata()

                plt.fill_betweenx(y1, x1, x2, lw=0, color=long_sector_color, alpha=0.5)
            else:
                print("Ill-defined 'long_sector'. It should be a 2-element list defining the start and end longitude of the cone in degrees; e.g. 'long_sector=[15,45]'")

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
            return fig, ax

    def _polar_twin(self, ax, E_long, position, long_offset):
        """
        add an additional axes which is needed to plot additional longitudinal tickmarks with Earth at longitude 0
        not used any more!
        """
        ax2 = ax.figure.add_axes(position, projection='polar',
                                 label='twin', frameon=False,
                                 theta_direction=ax.get_theta_direction(),
                                 theta_offset=E_long)

        ax2.set_rmax(self.max_dist + 0.3)
        ax2.yaxis.set_visible(False)
        ax2.set_theta_zero_location("S")
        ax2.tick_params(axis='x', colors='darkgreen', pad=10)
        ax2.set_xticks(np.pi/180. * np.linspace(180, -180, 8, endpoint=False))
        ax2.set_thetalim(-np.pi, np.pi)
        ax2.set_theta_offset(np.deg2rad(long_offset - E_long))
        gridlines = ax2.xaxis.get_gridlines()
        for xax in gridlines:
            xax.set_color('darkgreen')

        return ax2


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
