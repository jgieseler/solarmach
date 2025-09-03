import dateutil.parser  # type: ignore
import glob
import os
import pickle

import astropy.constants as aconst
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sunkit_magex.pfss as pfsspy
import sunpy.map
import threadpoolctl
from astropy.coordinates import SkyCoord
from sunpy.net import Fido
from sunpy.net import attrs as a


# @st_cache_decorator
def get_pfss_hmimap(filepath, email, carrington_rot, date, rss=2.5, nrho=35):
    """
    Downloading hmi map or calculating the PFSS solution

    params
    -------
    filepath: str
            Path to the hmimap, if exists.
    email: str
            The email address of a registered user
    carrington_rot: int
            The Carrington rotation corresponding to the hmi map
    date: str
            The date of the map. Format = 'YYYY/MM/DD'
    rss: float (default = 2.5)
            The height of the potential field source surface for the solution.
    nrho: int (default = 35)
            The resolution of the PFSS-solved field line objects

    returns
    -------
    output: hmi_synoptic_map object
            The PFSS-solution
    """

    time = a.Time(date, date)
    pfname =  f"PFSS_output_{str(time.start.datetime.date())}_CR{str(carrington_rot)}_SS{str(rss)}_nrho{str(nrho)}.p"

    # Check if PFSS file already exists locally:
    print(f"Searching for PFSS file from {filepath}")
    try:
        with open(f"{filepath}/{pfname}", 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            output = u.load()
        print("Found pickled PFSS file!")

    # If not, then download MHI mag, calc. PFSS, and save as picle file for next time
    except FileNotFoundError:
        print("PFSS file not found.\nDownloading...")
        series = a.jsoc.Series('hmi.synoptic_mr_polfil_720s')
        crot = a.jsoc.PrimeKey('CAR_ROT', carrington_rot)
        result = Fido.search(time, series, crot, a.jsoc.Notify(email))
        files = Fido.fetch(result)
        hmi_map = sunpy.map.Map(files[0])
        # pfsspy.utils.fix_hmi_meta(hmi_map)

        print('Data shape: ', hmi_map.data.shape)

        hmi_map = hmi_map.resample([360, 180]*u.pix)
        print('New shape: ', hmi_map.data.shape)

        pfss_input = pfsspy.Input(hmi_map, nrho, rss)
        output = pfsspy.pfss(pfss_input)
        with open(pfname, 'wb') as f:
            pickle.dump(output, f)

    return output


def multicolorline(x, y, cvals, ax, vmin=-90, vmax=90):
    """
    Constructs a line object, with each segment of the line color coded
    Original example from: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    params
    -------
    x, y: float
    cvals: str
    ax: Figure.Axes object
    vmin, vmax: int (default = -90, 90)

    returns
    -------
    line: LineCollection object
    """

    import cmasher as cmr
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(vmin, vmax)

    cmrmap = cmr.redshift

    # sample the colormaps that you want to use. Use 90 from each so there is one
    # color for each degree
    colors_pos = cmrmap(np.linspace(0.0, 0.30, 45))
    colors_neg = cmrmap(np.linspace(0.70, 1.0, 45))

    # combine them and build a new colormap
    colors = np.vstack((colors_pos, colors_neg))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    # establish the linecollection object
    lc = LineCollection(segments, cmap=mymap, norm=norm)

    # set the values used for colormapping
    lc.set_array(cvals)

    # set the width of line
    lc.set_linewidth(3)

    # this we want to return
    line = ax.add_collection(lc)

    return line


def get_field_line_coords(longitude, latitude, hmimap, seedheight):
    """
    Returns triplets of open magnetic field line coordinates, and the field line object itself

    params
    -------
    longitude: int/float
            longitude of the seeding point for the FieldLine tracing
    latitude: int/float
            latitude of the seeding point for the FieldLine tracing
    hmimap: hmi_synoptic_map object
            hmimap
    seedheight: float
            Heliocentric height of the seeding point

    returns
    -------
    coordlist: list[list[float,float,float]]
            The list of lists  of all coordinate triplets that correspond to the FieldLine objects traced
    flinelist: list[FieldLine]
            List of all FieldLine objects traced
    """

    # The amount of coordinate triplets we are going to trace
    try:
        coord_triplets = len(latitude)
    except TypeError:
        coord_triplets = 1
        latitude = [latitude]
        longitude = [longitude]

    # The loop in which we trace the field lines and collect them to the coordlist
    coordlist = []
    flinelist = []
    for i in range(coord_triplets):

        # Inits for finding another seed point if we hit null or closed line
        turn = 'lon'
        sign_switch = 1

        # Steps to the next corner, steps taken
        corner_tracker = [1, 0]

        init_lon, init_lat = longitude[i], latitude[i]

        # Keep tracing the field line until a valid one is found
        while (True):

            # Trace a field line downward from the point lon,lat on the pfss
            fline = trace_field_line(longitude[i], latitude[i], hmimap, seedheight=seedheight)

            radius0 = fline.coords.radius[0].value
            radius9 = fline.coords.radius[-1].value
            bool_key = (radius0==radius9)

            # If fline is not a valid field line, then spiral out from init_point and try again
            # Also check if this is a null line (all coordinates identical)
            # Also check if polarity is 0, meaning that the field line is NOT open
            if ((len(fline.coords) < 10) or bool_key or fline.polarity==0):  # fline.polarity==0

                longitude[i], latitude[i], sign_switch, corner_tracker, turn = spiral_out(longitude[i], latitude[i], sign_switch, corner_tracker, turn)

            # If there was nothing wrong, break the loop and proceed with the traced field line
            else:
                break

            # Check that we are not too far from the original coordinate
            if (corner_tracker[0] >= 10):
                print(f"no open field line found in the vicinity of {np.rad2deg(init_lon)}°, {np.rad2deg(init_lat)}°")
                break

        # Get the field line coordinate values in the correct order
        # Start on the photopshere, end at the pfss
        fl_r, fl_lon, fl_lat = get_coord_values(fline)

        # Fill in the lists
        triplet = [fl_r, fl_lon, fl_lat]
        coordlist.append(triplet)
        flinelist.append(fline)

    return coordlist, flinelist


def vary_flines(lon, lat, hmimap, n_varies, seedheight):
    """
    Finds a set of sub-pfss fieldlines connected to or very near a single footpoint on the pfss.

    lon: longitude of the footpoint [rad]
    lat: latitude of the footpoint [rad]

    n_varies:   tuple that holds the amount of circles and the number of dummy flines per circle
                if type(n_varies)=int, consider that as the amount of circles, and set the
                amount of dummy flines per circle to 16

    params
    -------
    lon: int/float
            The longitude of the footpoint in radians
    lat: int/float
            The latitude of the footpoint in radians
    hmimap: hmi_synoptic_map object
            The pfss-solution used to calculate the field lines
    n_varies: list[int,int] or int
            A list that holds the amount of circles and the number of dummy flines per circle
            if type(n_varies)=int, consider that as the amount of circles, and set the
            amount of dummy flines per circle to 16
    seedheight: float
            Heliocentric height of the tracing starting point

    returns
    -------
    coordlist: list[float,float,float]
            List of coordinate triplets of the original field lines (lon,lat,height)
    flinelist: list[FieldLine-object]
            List of Fieldline objects of the original field lines
    varycoords: list[float,float,float]
            List of coordinate triplets of the varied field lines
    varyflines: list[FieldLine-object]
            List of Fieldline objects of the varied field lines
    """

    # Field lines per n_circles (circle)
    if isinstance(n_varies, (list, tuple)):
        print(f"n_varies: {n_varies}")
        n_circles = n_varies[0]
        n_flines = n_varies[1]
    else:
        n_circles = n_varies
        n_flines = 16

    # First produce new points around the given lonlat_pair
    lons, lats= np.array([lon]), np.array([lat])
    increments = np.array([0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29])
    for circle in range(n_circles):

        newlons, newlats = circle_around(lon, lat, n_flines, r=increments[circle])
        lons, lats = np.append(lons, newlons), np.append(lats, newlats)

    pointlist = np.array([lons, lats])

    # Trace fieldlines from all of these points
    varycoords, varyflines = get_field_line_coords(pointlist[0], pointlist[1], hmimap, seedheight)

    # Because the original fieldlines and the varied ones are all in the same arrays,
    # Extract the varied ones to their own arrays
    coordlist, flinelist = [], []

    # Total amount of flines = 1 + (circles) * (fieldlines_per_circle)
    total_per_fp = n_flines*n_circles+1
    erased_indices = []
    for i in range(len(varycoords)):
        # n_flines*n_circles = the amount of extra field lines between each "original" field line
        if i%(total_per_fp)==0:
            erased_indices.append(i)
            # pop(i) removes the ith element from the list and returns it
            # -> we append it to the list of original footpoint fieldlines
            coordlist.append(varycoords[i])  # .pop(i)
            flinelist.append(varyflines[i])

    # Really ugly quick fix to erase values from varycoords and varyflines
    for increment, index in enumerate(erased_indices):
        varycoords.pop(index-increment)
        varyflines.pop(index-increment)

    return coordlist, flinelist, varycoords, varyflines


def trace_field_line(lon0, lat0, hmimap, seedheight, rad=True):
    """
    Traces a single open magnetic field line at coordinates (lon0,lat0) on the pfss down
    to the photosphere

    Parameters:
    -----------
    lon0, lat0: float
            Longitude and latitude of the seedpoint
    hmimap: hmimap-object
            hmimap
    seedheight: float
            The height at which field line tracing is started (in solar radii)
    rad: bool, (default True)
            Wether or not input coordinates are in radians. If False, consider them degrees

    Returns:
    --------
    field_lines: FieldLine or list[FieldLine]
            A FieldLine object, or a list of them, if input coordinates were a list

    """
    # from pfsspy import tracing

    # if lat0 and lon0 are given in deg for some reason, transform them to rad
    if not rad:
        lat0 = np.deg2rad(lat0)
        lon0 = np.deg2rad(lon0)

    # Start tracing from a given height
    height = seedheight*aconst.R_sun
    tracer = pfsspy.tracing.PythonTracer()

    # Add unit to longitude and latitude, so that SkyCoord understands them
    lon, lat = lon0*u.rad, lat0*u.rad

    # Seed the starting coordinate at the desired coordinates
    seed = SkyCoord(lon, lat, height, frame=hmimap.coordinate_frame)

    # Trace the field line from the seed point given the hmi map
    field_lines = tracer.trace(seed, hmimap)

    # Field_lines could be list of len=1, because there's only one seed point given to the tracer
    if len(field_lines) == 1:
        return field_lines[0]
    else:
        return field_lines


def spiral_out(lon, lat, sign_switch, corner_tracker, turn):
    """
    Moves the seeding point in an outward spiral.

    Parameters
    ---------
    lon, lat: float
        the carrington coordinates on a surface of a sphere (sun or pfss)
    sign_switch: int
        -1 or 1, dictates the direction in which lon or lat is incremented
    corner_tracker: tuple
        first entry is steps_unti_corner, int that tells how many steps to the next corner of a spiral
        the second entry is steps taken on a given spiral turn
    turn: str
        indicates which is to be incremented, lon or lat

    returns
    -----------
    lon, lat: float
            new coordinate pair
    """

    # In radians, 1 rad \approx 57.3 deg
    step = 0.005

    # Keeps track of how many steps until it's time to turn
    steps_until_corner, steps_moved = corner_tracker[0], corner_tracker[1]

    if turn=='lon':

        lon = lon + step*sign_switch
        lat = lat

        steps_moved += 1

        # We have arrived in a corner, time to move in lat direction
        if steps_until_corner == steps_moved:
            steps_moved = 0
            turn = 'lat'

        return lon, lat, sign_switch, [steps_until_corner, steps_moved], turn

    if turn=='lat':

        lon = lon
        lat = lat + step*sign_switch

        steps_moved += 1

        # Hit a corner; start moving in the lon direction
        if steps_until_corner == steps_moved:
            steps_moved = 0
            steps_until_corner += 1
            turn = 'lon'
            sign_switch = sign_switch*(-1)

        return lon, lat, sign_switch, [steps_until_corner, steps_moved], turn


def get_coord_values(field_line):
    """
    Gets the coordinate values from FieldLine object and makes sure that they are in the right order.

    params
    -------
    field_line: FieldLine object

    returns
    -------
    fl_r: list[float]
        The list of heliocentric distances of each segment of the field line
    fl_lon: list[float]
        The list of longitudes of each field line segment
    fl_lat: list[float]
        The list of latitudes of each field line segment
    """

    # first check that the field_line object is oriented correctly (start on photosphere and end at pfss)
    fl_coordinates = field_line.coords
    fl_coordinates = check_field_line_alignment(fl_coordinates)

    fl_r = fl_coordinates.radius.value / aconst.R_sun.value
    fl_lon = fl_coordinates.lon.value
    fl_lat = fl_coordinates.lat.value

    return fl_r, fl_lon, fl_lat


def circle_around(x, y, n, r=0.1):
    """
    Produces new points around a (x,y) point in a circle.
    At the moment does not work perfectly in the immediate vicinity of either pole.

    params
    -------
    x,y: int/float
        Coordinates of the original point
    n: int
        The amount of new points around the origin
    r: int/float (default = 0.1)
        The radius of the circle at which new points are placed

    returns
    -------
    pointlist: list[float]
            List of new points (tuples) around the original point in a circle, placed at equal intervals
    """

    origin = (x, y)

    x_coords = np.array([])
    y_coords = np.array([])
    for i in range(0, n):

        theta = (2*i*np.pi)/n
        newx = origin[0] + r*np.cos(theta)
        newy = origin[1] + r*np.sin(theta)

        if newx >= 2*np.pi:
            newx = newx - 2*np.pi

        if newy > np.pi/2:
            overflow = newy - np.pi/2
            newy = newy - 2*overflow

        if newy < -np.pi/2:
            overflow = newy + np.pi/2
            newy = newy + 2*overflow

        x_coords = np.append(x_coords, newx)
        y_coords = np.append(y_coords, newy)

    pointlist = np.array([x_coords, y_coords])

    return pointlist


def check_field_line_alignment(coordinates):
    """
    Checks that a field line object is oriented such that it starts from
    the photpshere and ends at the pfss. If that is not the case, then
    flips the field line coordinates over and returns the flipped object.
    """

    fl_r = coordinates.radius.value

    if fl_r[0] > fl_r[-1]:
        coordinates = np.flip(coordinates)

    return coordinates


def spheric2cartesian(r, theta, phi):
    """
    Does a coordinate transformation from spherical to cartesian.
    For Stonyhurst heliographic coordinates, this means converting to HEEQ
    (Heliocentric Earth Equatorial), following Eq. (2) of Thompson (2006),
    DOI: 10.1051/0004-6361:20054262

    r : the distance to the origin
    theta : the elevation angle (goes from -pi to pi)
    phi : the azimuth angle (goes from 0 to 2pi)
    """

    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.cos(theta)
    z = r * np.sin(theta)

    return x, y, z


def sphere(radius, clr, dist=0):
    """
    Constructs a sphere with a given radius and color.

    params
    ---------
    radius : float, int
            The radius of the sphere

    clr : str
            The color code for the sphere

    dist : float, int
            The displacement of the sphere, if not centered at origin
    """

    import plotly.graph_objects as go

    # Set up 100 points. First, do angles
    phi = np.linspace(0, 2*np.pi, 100)  # phi, the azimuthal angle goes from 0 to 2pi
    theta = np.linspace(0, np.pi, 100)  # theta, the elevation angle goes from 0 to pi

    # Set up coordinates for points on the sphere
    x0 = dist + radius * np.outer(np.cos(phi), np.sin(theta))
    y0 = radius * np.outer(np.sin(phi), np.sin(theta))
    z0 = radius * np.outer(np.ones(100), np.cos(theta))

    # Set up trace (the object that is then plottable)
    trace= go.Surface(x=x0, y=y0, z=z0, colorscale=[[0, clr], [1, clr]], showscale=False, name="Sun")

    return trace


def calculate_pfss_solution(gong_map, rss, coord_sys, nrho=35):
    """
    Calculates a Potential Field Source Surface (PFSS) solution based on a GONG map and parameters.

    Parameters:
    -----------
    gong_map : {sunpy.map.Map}
        GONG map in Carrington or Stonyhurst coordinates, obtained with get_gong_map()
    rss : {float}
        source surface height in solar radii
    coord_sys: {str}
        cordinate system used: either 'car' or 'Carrington', or 'sto' or 'Stonyhurst'
    nrho : {float/int, optional}
        rho = ln(r) -> nrho is the amount of points in this logarithmic range. Default is 35.

    Returns:
    ----------
    pfss_solution : {pfsspy.Output}
        The PFSS solution that can be used to plot magnetic field lines under the source surface
    """
    # GONG map is in Carrington coordinates
    if gong_map.coordinate_system.axis1=='CRLN-CEA':
        if coord_sys.lower().startswith('sto'):
            # Convert GONG map from default Carrington to Stonyhurst coordinate system
            new_map_header = sunpy.map.header_helper.make_heliographic_header(date=gong_map.date, observer_coordinate=gong_map.observer_coordinate, shape=gong_map.data.shape, frame='stonyhurst', projection_code='CEA')
        if coord_sys.lower().startswith('car'):
            # Convert GONG map from default Carrington to Carrington coordinate system.
            # This sounds useless, but is needed to rebuild the meta data of the GONG map when using sunpy>5.1.0
            # cf. https://github.com/sunpy/sunpy/issues/7313
            new_map_header = sunpy.map.header_helper.make_heliographic_header(date=gong_map.date, observer_coordinate=gong_map.observer_coordinate, shape=gong_map.data.shape, frame='carrington', projection_code='CEA')
        gong_map = gong_map.reproject_to(new_map_header)
    # GONG map is in Stonyhurst coordinates
    elif gong_map.coordinate_system.axis1=='HGLN-CEA':
        if coord_sys.lower().startswith('car'):
            # Convert GONG map from Stonyhurst to Carrington coordinate system.
            # This shouldn't be necessary, as Carrington is the default for GONG maps, but better be sure.
            new_map_header = sunpy.map.header_helper.make_heliographic_header(date=gong_map.date, observer_coordinate=gong_map.observer_coordinate, shape=gong_map.data.shape, frame='carrington', projection_code='CEA')
            gong_map = gong_map.reproject_to(new_map_header)

    # The pfss input object, assembled from a gong map, resolution (nrho) and source surface height (rss)
    pfss_in = pfsspy.Input(gong_map, nrho, rss)

    # This is the pfss solution, calculated from the input object
    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
        pfss_solution = pfsspy.pfss(pfss_in)

    return pfss_solution


def load_gong_map(filepath=None):
    """
    https://gong.nso.edu/data/magmap/
    https://docs.sunpy.org/en/v4.0.6/generated/api/sunpy.net.dataretriever.GONGClient.html
    https://gong2.nso.edu/oQR/zqs/202104/mrzqs210413/

    """

    if not filepath:
        filepath = os.getcwd()

    # Load a GONG (Global Oscillation Network Group) synoptic magnetic map
    if isinstance(filepath, str):

        # Acquire a GONG map from which the pfss solution is calculated from
        gong_map = sunpy.map.Map(filepath)

    return gong_map


def download_gong_map(timestr, filepath=None):
    """
    Gets the download link for a GONG synoptic map
    """

    import pandas as pd
    from sunpy.net import Fido, attrs

    if filepath is None:
        filepath = os.getcwd()

    desired_time = pd.to_datetime(timestr, yearfirst=True)
    desired_time_plus_hour = desired_time + pd.Timedelta(hours=1)

    result = Fido.search(attrs.Time(desired_time, desired_time_plus_hour), attrs.Instrument('GONG'))

    file = Fido.fetch(result, path=filepath)

    print(f"Downloaded GONG map to {file.data[0]}")
    return file.data[0]


def construct_gongmap_filename(timestr, directory):
    """
    Constructs a default filepath
    """

    dtime = dateutil.parser.parse(timestr)

    # If directory is None, (equivalent to not directory in logic), then use the current directory as a base
    if not directory:
        directory = os.getcwd()

    carrington_rot = sunpy.coordinates.sun.carrington_rotation_number(t=timestr).astype(int)  # dynamic CR from date
    filename = f"mrzqs{dtime.strftime('%y%m%dt%H')}*c{carrington_rot}_*.fits.gz"
    filepath = f"{directory}{os.sep}{filename}"
    filepaths = glob.glob(filepath)

    # If exactly one match, then that most probably is the right file
    if len(filepaths) == 1:
        print(f"Automatic file search based on given time found {filepaths[0]}")
        return filepaths[0]
    else:
        print(f"Automatic file search based on given time failed in directory {directory}")
        return directory


def get_gong_map(time: str, filepath: str = None, autodownload=True):
    """
    A wrapper for functions load_gong_map() and download_gong_map().
    Returns a gong map if one is found or autodownload is True. If no map found and
    autodownload is False, then return None.

    Parameters:
    -----------
    timestr : {str}
        A pandas-compatible timestring, e.g., '2010-11-29 12:45'
    filepath : {str}, optional, default=None
        The path to the gong map file with the name of the file, e.g., 'use/xyz/gong_maps/mrzqs211009t0814c2249_105.fits.gz'
        If no filepath provided, use the current directory.
    autodownload : {bool}, optional, default=True
        If file is not found, download it automatically.
    """

    # Try to construct a filename from current working directory and given datetime
    if not filepath or filepath[-8:] not in (".fits.gz"):
        filepath = construct_gongmap_filename(timestr=time, directory=filepath)

    try:
        gong_map = load_gong_map(filepath=filepath)
    except Exception:
        if autodownload:
            new_filepath = download_gong_map(time, filepath=filepath)
            gong_map = load_gong_map(filepath=new_filepath)
        else:
            return None

    return gong_map
