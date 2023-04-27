solarmach
=========

|pypi Version| |conda version| |license| |python version| |pytest| |zenodo|

.. |pypi Version| image:: https://img.shields.io/pypi/v/solarmach?style=flat&logo=pypi
   :target: https://pypi.org/project/solarmach/
.. |conda version| image:: https://img.shields.io/conda/vn/conda-forge/solarmach?style=flat&logo=anaconda
   :target: https://anaconda.org/conda-forge/solarmach/
.. |license| image:: https://img.shields.io/conda/l/conda-forge/solarmach?style=flat
   :target: https://github.com/jgieseler/solarmach/blob/main/LICENSE.rst
.. |python version| image:: https://img.shields.io/pypi/pyversions/solarmach?style=flat&logo=python
.. |pytest| image:: https://github.com/jgieseler/solarmach/workflows/pytest/badge.svg
.. |zenodo| image:: https://zenodo.org/badge/469735286.svg
   :target: https://zenodo.org/badge/latestdoi/469735286



The Solar MAgnetic Connection Haus (Solar-MACH) tool is a multi-spacecraft longitudinal configuration plotter. This is the repository of the pip/conda package of Solar-MACH, called **solarmach**. For the corresponding streamlit repository, which is used for `solar-mach.github.io <https://solar-mach.github.io>`_, see `github.com/jgieseler/Solar-MACH <https://github.com/jgieseler/Solar-MACH>`_.

Installation
------------

solarmach requires python >= 3.7.

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

   # necessary options
   body_list = ['STEREO-A', 'Earth', 'BepiColombo', 'PSP', 'Solar Orbiter', 'Mars']
   vsw_list = [400, 400, 400, 400, 400, 400, 400]   # position-sensitive solar wind speed per body in body_list
   date = '2021-10-28 15:15:00'

   # optional parameters
   coord_sys = 'Carrington'                         # 'Carrington' (default) or 'Stonyhurst'
   reference_long = 273                             # longitude of reference (None to omit)
   reference_lat = 0                                # latitude of reference (None to omit)
   plot_spirals = True                              # plot Parker spirals for each body
   plot_sun_body_line = True                        # plot straight line between Sun and body
   long_offset = 270                                # longitudinal offset for polar plot; defines where Earth's longitude is (by default 270, i.e., at "6 o'clock")
   reference_vsw = 400                              # define solar wind speed at reference
   return_plot_object = False                        # figure and axis object of matplotib are returned, allowing further adjustments to the figure
   transparent = False                              # make output figure background transparent
   numbered_markers = True                          # plot each body with a numbered marker
   filename = 'Solar-MACH_'+date.replace(' ', '_')  # define filename of output figure

   # initialize
   sm = SolarMACH(date, body_list, vsw_list, reference_long, reference_lat, coord_sys)

   # make plot
   sm.plot(
      plot_spirals=plot_spirals,
      plot_sun_body_line=plot_sun_body_line,
      reference_vsw=reference_vsw,
      transparent=transparent,
      numbered_markers=numbered_markers,
      long_offset=long_offset,
      return_plot_object=return_plot_object,
      outfile=filename+'.png'
   )
   
   # obtain data as Pandas DataFrame
   display(sm.coord_table)

.. image:: https://github.com/jgieseler/solarmach/raw/main/examples/solarmach.png
  :alt: Example output figure
  
See `example notebook <https://github.com/jgieseler/solarmach/blob/main/examples/example.ipynb>`_ for all options!
  
Example Notebook
----------------

**solarmach** can easily be run in a Jupyter Notebook. 

- `Download example notebook <https://github.com/jgieseler/solarmach/raw/main/examples/example.ipynb>`_

- Try it online: |binder|
  
.. |binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/jgieseler/solarmach/main?labpath=examples%2Fexample.ipynb
 
Citation
--------

Please cite the following paper if you use **solarmach** in your publication:

Gieseler, J., Dresing, N., Palmroos, C., von Forstner, J.L.F., Price, D.J., Vainio, R. et al. (2022).
Solar-MACH: An open-source tool to analyze solar magnetic connection configurations. *Front. Astronomy Space Sci.* 9. `doi:10.3389/fspas.2022.1058810 <https://doi.org/10.3389/fspas.2022.1058810>`_ 
 
Acknowledgements
----------------
 
The Solar-MACH tool was originally developed at Kiel University, Germany and further discussed within the `ESA Heliophysics Archives USer (HAUS) <https://www.cosmos.esa.int/web/esdc/archives-user-groups/heliophysics>`_ group.

Powered by: |matplotlib| |sunpy|

.. |matplotlib| image:: https://matplotlib.org/stable/_static/logo2_compressed.svg
   :height: 25px
   :target: https://matplotlib.org
.. |sunpy| image:: https://raw.githubusercontent.com/sunpy/sunpy-logo/master/generated/sunpy_logo_landscape.svg
   :height: 30px
   :target: https://sunpy.org
