solarmach
=========

|pypi Version| |conda version| |license| |python version|

.. |pypi Version| image:: https://img.shields.io/pypi/v/solarmach?style=flat&logo=pypi
   :target: https://pypi.org/project/solarmach/
.. |conda version| image:: https://img.shields.io/conda/vn/conda-forge/solarmach?style=flat&logo=anaconda
   :target: https://anaconda.org/conda-forge/solarmach/
.. |license| image:: https://img.shields.io/conda/l/conda-forge/solarmach?style=flat
   :target: https://github.com/jgieseler/solarmach/blob/main/LICENSE.rst
.. |python version| image:: https://img.shields.io/pypi/pyversions/solarmach?style=flat&logo=python

The Solar MAgnetic Connection Haus (Solar-MACH) tool is a multi-spacecraft longitudinal configuration plotter. This is the repository of the pip/conda package of Solar-MACH, called **solarmach**. For the corresponding streamlit repository, which is used for `solar-mach.github.io <https://solar-mach.github.io>`_, see `github.com/jgieseler/Solar-MACH <https://github.com/jgieseler/Solar-MACH>`_.

Installation
------------

solarmach requires python >= 3.6.

It can be installed either from `PyPI <https://pypi.org/project/solarmach/>`_ using:

.. code:: bash

    pip install solarmach
    
or from `conda <https://anaconda.org/conda-forge/solarmach/>`_ using:

.. code:: bash

    conda install -c conda-forge solarmach

Usage
-----

.. code:: python

   from solarmach import SolarMACH, print_body_list
   
   # optional: get list of available bodies/spacecraft
   print(print_body_list().index)

   # necessary:
   body_list = ['STEREO-A', 'Earth', 'BepiColombo', 'PSP', 'Solar Orbiter', 'Mars']
   vsw_list = [400, 400, 400, 400, 400, 400, 400]
   date = '2021-10-28 15:15:00'
   
   # optional:
   reference_long = 273                             # Carrington longitude of reference (None to omit)
   reference_lat = 0                                # Carrington latitude of reference (None to omit)
   plot_spirals = True                              # plot Parker spirals for each body
   plot_sun_body_line = True                        # plot straight line between Sun and body
   show_earth_centered_coord = False                # display Earth-aligned coordinate system
   reference_vsw = 400                              # define solar wind speed at reference
   transparent = False                              # make output figure background transparent
   numbered_markers = True                          # plot each body with a numbered marker
   filename = 'Solar-MACH_'+date.replace(' ', '_')  # define filename of output figure
   
   # optional
   # if input coordinates for reference are Stonyhurst, convert them to Carrington for further use
   import astropy.units as u
   from astropy.coordinates import SkyCoord
   from sunpy.coordinates import frames
   reference_long = 2                               # Stonyhurst longitude of reference (None to omit)
   reference_lat = 26                               # Stonyhurst latitude of reference (None to omit)
   coord = SkyCoord(reference_long*u.deg, reference_lat*u.deg, frame=frames.HeliographicStonyhurst, obstime=date)
   coord = coord.transform_to(frames.HeliographicCarrington(observer='Sun'))
   reference_long = coord.lon.value                 # Carrington longitude of reference
   reference_lat = coord.lat.value                  # Carrington latitude of reference
     
   # initialize
   sm = SolarMACH(date, body_list, vsw_list, reference_long, reference_lat)
   
   # make plot
   sm.plot(
      plot_spirals=plot_spirals,                            
      plot_sun_body_line=plot_sun_body_line,                
      show_earth_centered_coord=show_earth_centered_coord,  
      reference_vsw=reference_vsw,                          
      transparent=transparent,
      numbered_markers=numbered_markers,
      outfile=filename+'.png'
   )
   
   # obtain data as Pandas DataFrame
   display(sm.coord_table)

.. image:: https://github.com/jgieseler/solarmach/raw/main/examples/solarmach.png
  :alt: Example output figure
  
Example Notebook
----------------

**solarmach** can easily be run in a Jupyter Notebook. 

- `Download example notebook <https://github.com/jgieseler/solarmach/raw/main/examples/example.ipynb>`_

- Try it online: |binder|
  
.. |binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/jgieseler/solarmach/main?labpath=examples%2Fexample.ipynb
 
Acknowledgements
----------------
 
The Solar-MACH tool was originally developed at Kiel University, Germany and further discussed within the `ESA Heliophysics Archives USer (HAUS) <https://www.cosmos.esa.int/web/esdc/archives-user-groups/heliophysics>`_ group (`original code <https://github.com/esdc-esac-esa-int/Solar-MACH>`_).

Powered by: |matplotlib| |sunpy|

.. |matplotlib| image:: https://matplotlib.org/stable/_static/logo2_compressed.svg
   :height: 25px
   :target: https://matplotlib.org
.. |sunpy| image:: https://raw.githubusercontent.com/sunpy/sunpy-logo/master/generated/sunpy_logo_landscape.svg
   :height: 30px
   :target: https://sunpy.org
