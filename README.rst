solarmach
=========

|pypi Version| |conda version| |python version| |pytest| |codecov| |docs| |repostatus| |license| |zenodo|

.. |pypi Version| image:: https://img.shields.io/pypi/v/solarmach?style=flat&logo=pypi
   :target: https://pypi.org/project/solarmach/
.. |conda version| image:: https://img.shields.io/conda/vn/conda-forge/solarmach?style=flat&logo=anaconda
   :target: https://anaconda.org/conda-forge/solarmach/
.. |python version| image:: https://img.shields.io/pypi/pyversions/solarmach?style=flat&logo=python
.. |pytest| image:: https://github.com/jgieseler/solarmach/actions/workflows/pytest.yml/badge.svg?branch=main
   :target: https://github.com/jgieseler/solarmach/actions/workflows/pytest.yml
.. |codecov| image:: https://codecov.io/gh/jgieseler/solarmach/branch/main/graph/badge.svg?token=CT2P8AQU3B
   :target: https://codecov.io/gh/jgieseler/solarmach
.. |docs| image:: https://readthedocs.org/projects/solarmach/badge/?version=latest
   :target: https://solarmach.readthedocs.io/en/latest/?badge=latest
.. |repostatus| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active
.. |license| image:: https://img.shields.io/conda/l/conda-forge/solarmach?style=flat
   :target: https://github.com/jgieseler/solarmach/blob/main/LICENSE.rst
.. |zenodo| image:: https://zenodo.org/badge/469735286.svg
   :target: https://zenodo.org/badge/latestdoi/469735286



The Solar MAgnetic Connection Haus (Solar-MACH) tool is a multi-spacecraft longitudinal configuration plotter. This is the repository of the pip/conda package of Solar-MACH, called **solarmach**. For the corresponding streamlit repository, which is used for `solar-mach.github.io <https://solar-mach.github.io>`_, see `github.com/jgieseler/Solar-MACH <https://github.com/jgieseler/Solar-MACH>`_.

Installation
------------

solarmach requires python >= 3.10.

It can be installed either from `PyPI <https://pypi.org/project/solarmach/>`_ using:

.. code:: bash

    pip install solarmach
    
or from `conda <https://anaconda.org/conda-forge/solarmach/>`_ using:

.. code:: bash

    conda install -c conda-forge solarmach

Basic usage
-----------

.. code:: python

   from solarmach import SolarMACH, print_body_list

   # optional: get list of available bodies/spacecraft
   print(print_body_list().index)

   # necessary options
   body_list = ['STEREO-A', 'Earth', 'BepiColombo', 'PSP', 'Solar Orbiter', 'Mars']
   date = '2021-10-28 15:15:00'

   # Previously you needed to define position-sensitive solar wind speed per
   # body in body_list, e.g., vsw_list = [400, 400, 400, 400, 400, 400, 400]
   # Now you can skip this parameter or provide an empty list. Then solarmach
   # will try to automatically obtain measured solar wind speeds from each
   # spacecraft
   vsw_list = []

   # optional parameters
   coord_sys = 'Carrington'                         # 'Carrington' (default) or 'Stonyhurst'
   reference_long = 273                             # longitude of reference (None to omit)
   reference_lat = 0                                # latitude of reference (None to omit)
   plot_spirals = True                              # plot Parker spirals for each body
   plot_sun_body_line = True                        # plot straight line between Sun and body
   long_offset = 270                                # longitudinal offset for polar plot; defines where Earth's longitude is (by default 270, i.e., at "6 o'clock")
   reference_vsw = 400                              # define solar wind speed at reference
   return_plot_object = False                       # figure and axis object of matplotib are returned, allowing further adjustments to the figure
   transparent = False                              # make output figure background transparent
   markers = 'numbers'                              # use 'numbers' or 'letters' for the body markers (use False for colored squares)
   filename = 'solarmach.png'                       # define filename of output figure. can be .png or .pdf
 
   # initialize
   sm = SolarMACH(date, body_list, vsw_list, reference_long, reference_lat, coord_sys)

   # make plot
   sm.plot(
      plot_spirals=plot_spirals,
      plot_sun_body_line=plot_sun_body_line,
      reference_vsw=reference_vsw,
      transparent=transparent,
      markers=markers,
      long_offset=long_offset,
      return_plot_object=return_plot_object,
      outfile=filename
   )
   
   # obtain data as Pandas DataFrame
   display(sm.coord_table)

.. image:: https://github.com/jgieseler/solarmach/raw/main/examples/solarmach.png
  :alt: Example output figure
  

Documentation
-------------
Full documentation for the package can be found at https://solarmach.readthedocs.io

  
Example Notebooks
-----------------

**solarmach** can easily be run in a Jupyter Notebook. 

- `Show example notebook <https://nbviewer.org/github/jgieseler/solarmach/blob/main/examples/example.ipynb>`_ |nbviewer1|
  
 
.. |nbviewer1| image:: https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg
 :target: https://nbviewer.org/github/jgieseler/solarmach/blob/main/examples/example.ipynb
 

Contributing
------------

Contributions to this package are very much welcome and encouraged! Contributions can take the form of `issues <https://github.com/jgieseler/solarmach/issues>`_ to report bugs and request new features or `pull requests <https://github.com/jgieseler/solarmach/pulls>`_ to submit new code. 

Please make contributions specific to the streamlit web-version that is used for `solar-mach.github.io <https://solar-mach.github.io>`_ in the corresponding repository at `github.com/jgieseler/Solar-MACH <https://github.com/jgieseler/Solar-MACH/>`__.


Citation
--------

Please cite the following paper if you use **solarmach** in your publication:

Gieseler, J., Dresing, N., Palmroos, C., von Forstner, J.L.F., Price, D.J., Vainio, R. et al. (2022).
Solar-MACH: An open-source tool to analyze solar magnetic connection configurations. *Front. Astronomy Space Sci.* 9. `doi:10.3389/fspas.2022.1058810 <https://doi.org/10.3389/fspas.2022.1058810>`_ 

Acknowledgements
----------------

The Solar-MACH tool was originally developed at Kiel University, Germany and further discussed within the `ESA Heliophysics Archives USer (HAUS) <https://www.cosmos.esa.int/web/esdc/archives-user-groups/heliophysics>`_ group.

This project has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No 101134999 (SOLER) and from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101004159 (SERPENTINE).

Powered by: |matplotlib| |sunpy| |speasy| |plotly|

.. |matplotlib| image:: https://matplotlib.org/stable/_static/logo_dark.svg
   :height: 25px
   :target: https://matplotlib.org
.. |sunpy| image:: https://raw.githubusercontent.com/sunpy/sunpy-logo/master/generated/sunpy_logo_landscape.svg
   :height: 30px
   :target: https://sunpy.org
.. |speasy| image:: https://raw.githubusercontent.com/SciQLop/speasy/main/logo/logo_speasy.svg
   :height: 30px
   :target: https://pypi.org/project/speasy/
.. |plotly| image:: https://avatars.githubusercontent.com/u/5997976?s=64&v=4
   :height: 30px
   :target: https://github.com/plotly/plotly.py
