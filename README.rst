solarmach
=========

The Solar MAgnetic Connection Haus (Solar-MACH) tool is a multi-spacecraft longitudinal configuration plotter. This is the repository of the pip package of Solar-MACH, called **solarmach**. For the corresponding streamlit repository, which is used for https://solar-mach.github.io, visit https://github.com/jgieseler/Solar-MACH

Installation
------------

solarmach can be installed from this repository using pip:

.. code:: bash

    pip install git+https://github.com/jgieseler/solarmach

Usage
-----

.. code:: python

   import datetime as dt
   from solarmach import SolarMACH

   # necessary:
   body_list = ['STEREO-A', 'STEREO-B', 'Earth', 'MPO', 'PSP', 'Solar Orbiter', 'Mars']
   vsw_list = [300, 400, 500, 600, 700, 800, 900, 200]
   reference_long = 0
   reference_lat = 0
   date = '2020-05-01 13:00:00'
   
   # optional:
   plot_spirals = True
   plot_sun_body_line = True
   show_earth_centered_coord = True
   reference_vsw = 400
   transparent = False
   numbered_markers = True
     
   sm = SolarMACH(date, body_list, vsw_list, reference_long, reference_lat)
   
   sm.plot(
      plot_spirals=plot_spirals,                            # plot Parker spirals for each body
      plot_sun_body_line=plot_sun_body_line,                # plot straight line between Sun and body
      show_earth_centered_coord=show_earth_centered_coord,  # display Earth-aligned coordinate system
      reference_vsw=reference_vsw,                          # define solar wind speed at reference
      transparent=transparent,
      numbered_markers=numbered_markers,
      # outfile=filename+'.png'                               # output file (optional)
   )
   
   plt.show()
   
   display(sm.coord_table)


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://solar-mach.github.io)
