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

   from solarmach import SolarMACH

   # necessary:
   body_list = ['STEREO-A', 'Earth', 'BepiColombo', 'PSP', 'Solar Orbiter', 'Mars']
   vsw_list = [400, 400, 400, 400, 400, 400, 400]
   reference_long = 273
   reference_lat = 0
   date = '2021-10-28 15:15:00'
   
   # optional:
   plot_spirals = True                              # plot Parker spirals for each body
   plot_sun_body_line = True                        # plot straight line between Sun and body
   show_earth_centered_coord = False                # display Earth-aligned coordinate system
   reference_vsw = 400                              # define solar wind speed at reference
   transparent = False                              # make figure background transparent
   numbered_markers = True                          # plot each body with a numbered marker
   filename = 'Solar-MACH_'+date.replace(' ', '_')  # define filename of output figure
     
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
   
   # print out table of results
   display(sm.coord_table)

.. image:: https://github.com/jgieseler/solarmach/raw/main/examples/Solar-MACH_2021-10-28_15:15:00.png
  :alt: Example output figure
  
Example Notebook
----------------

**solarmach** can easily be run in a Jupyter Notebook. 

- `Download example notebook <https://github.com/jgieseler/solarmach/raw/main/examples/example.ipynb>`_

- Try it online: |binder|
  
.. |binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/jgieseler/solarmach/main?labpath=examples%2Fexample.ipynb
