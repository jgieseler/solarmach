Basic usage
-----------

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

.. image:: ../examples/solarmach.png
  :alt: Example output figure
  
See example notebooks in next section for all options!