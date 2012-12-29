.. _api_reference:

API reference
=============

This is a complete api reference to the openpiv python module.

The ``openpiv.preprocess`` module
----------------------------------
.. automodule:: openpiv.preprocess

.. currentmodule:: openpiv.preprocess

.. autosummary:: 
    :toctree: generated/
    
    dynamic_masking


The ``openpiv.tools`` module
----------------------------
.. automodule:: openpiv.tools

.. currentmodule:: openpiv.tools

.. autosummary:: 
    :toctree: generated/
    
    imread
    save
    display
    display_vector_field
    Multiprocesser

The ``openpiv.pyprocess`` module
--------------------------------
.. automodule:: openpiv.pyprocess

.. currentmodule:: openpiv.pyprocess

.. autosummary:: 
    :toctree: generated/
    
    normalize_intensity
    correlate_windows
    get_coordinates
    get_field_shape
    moving_window_array
    find_first_peak
    find_second_peak
    find_subpixel_peak_position
    piv


The ``openpiv.process`` module
--------------------------------
.. automodule:: openpiv.process

.. currentmodule:: openpiv.process

.. autosummary:: 
    :toctree: generated/
    
    extended_search_area_piv
    CorrelationFunction
    get_coordinates
    get_field_shape
    correlate_windows
    normalize_intensity
    
The ``openpiv.lib`` module
--------------------------------
.. automodule:: openpiv.lib

.. currentmodule:: openpiv.lib

.. autosummary:: 
    :toctree: generated/
    
    sincinterp    
    replace_nans
    
    
The ``openpiv.filters`` module  
------------------------------

.. automodule:: openpiv.filters

.. currentmodule:: openpiv.filters

.. autosummary:: 
    :toctree: generated/
    
    gaussian
    _gaussian_kernel
    replace_outliers
    
    
The ``openpiv.validation`` module
---------------------------------

.. automodule:: openpiv.validation

.. currentmodule:: openpiv.validation

.. autosummary:: 
    :toctree: generated/
    
    global_val
    sig2noise_val
    global_std
    local_median_val

The ``openpiv.scaling`` module
------------------------------

.. automodule:: openpiv.scaling

.. currentmodule:: openpiv.scaling

.. autosummary:: 
    :toctree: generated/
    
    uniform
