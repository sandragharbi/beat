.. short_installation:

*******************************
Short Installation instructions
*******************************

Using system packages
---------------------
First install pyrocko following this webpage:

 - `pyrocko <http://pyrocko.org/>`__

Then install the following packages, pymc3 version3.2 and pyproj::

    sudo pip install pymc3==3.2
    sudo pip install pyproj

    cd ~/src  # or whereever you keep the packages
    git clone https://github.com/hvasbath/beat
    cd beat
    sudo python setup.py install

Using anaconda2
---------------

Greens Function calculations
----------------------------

To calculate the Greens Functions we rely on modeling codes written by
`Rongjiang Wang <http://www.gfz-potsdam.de/en/section/physics-of-earthquakes-and-volcanoes/staff/profil/rongjiang-wang/>`__.
If you plan to use the GreensFunction calculation framework,
these codes are required and need to be compiled manually.
The original codes are packaged for windows and can be found 
`here <http://www.gfz-potsdam.de/en/section/physics-of-earthquakes-and-volcanoes/data-products-services/downloads-software/>`__.

For Unix systems the codes had to be repackaged.

The packages below are also github repositories and you may want to use "git clone" to download:

    git clone <url>

This also enables easy updating for potential future changes.

For configuration and compilation please follow the descriptions provided in each repository respectively.

Seismic synthetics
""""""""""""""""""
* `QSEIS <https://github.com/pyrocko/fomosto-qseis>`__
* `QSSP <https://github.com/pyrocko/fomosto-qssp>`__


Geodetic synthetics
"""""""""""""""""""
* `PSGRN_PSCMP <https://github.com/pyrocko/fomosto-psgrn-pscmp>`__

