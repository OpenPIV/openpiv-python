Information for developers and contributors
===========================================

OpenPiv need developers to improve further. Your support, code and contribution is very welcome and 
we are grateful you can provide some. Please send us an email to openpiv-develop@googlegroups.com
to get started, or for any kind of information.

We use `git <http://git-scm.com/>`_ for development version control, and we have a main repository on `github <https://github.com/>`_.


Development workflow
--------------------
This is absolutely not a comprehensive guide of git development, and it is only an indication of our workflow.

1) Download and install git. Instruction can be found `here <http://help.github.com/>`_.
2) Set up a github account.
3) Clone OpenPiv repository using::

    git clone http://github.com/openpiv/openpiv-python.git
    
4) create a branch `new_feature` where you implement your new feature.
5) Fix, change, implement, document code, ...
6) From time to time fetch and merge your master branch with that of the main repository.
7) Be sure that everything is ok and works in your branch.
8) Merge your master branch with your `new_feature` branch.
9) Be sure that everything is now ok and works in you master branch.
10) Send a `pull request <http://help.github.com/pull-requests/>`_.

11) Create another branch for a new feature.

Which language can I use?
-------------------------
As a general rule, we use Python where it does not make any difference with code speed. In those situations where Python speed is
the bottleneck, we have some possibilities, depending on your skills and background. If something has to be written from scratch
use the first language from the following which you are confortable with: Cython, C, C++, FORTRAN. If you have existing, debugged, tested code that
you would like to share, then no problem. We accept it, whichever language may be written in!

Things OpenPIV currently needs, (in order of importance)
--------------------------------------------------------
* Good documentation (in progress)
* The implementation of advanced processing algorithms (in progress)
* Flow field filtering and validation functions
* Cython wrappers for C/C++ codes.
* A good graphical user interface (in progress)


How to test all the notebooks::
-----------------------------

    conda create -n openpiv
    conda activate openpiv
    conda install -c conda-forge openpiv
    conda install ipykernel  
    python -m ipykernel install --user --name openpiv --display-name="openpiv"  
    jupyter nbconvert --to html --ExecutePreprocessor.kernel_name=openpiv --execute *.ipynb  
    

Then open the `openpiv/examples/notebooks` and check the HTML files. If one of those will fail, the error message will be in the command shell

If you need to install cv2::
--------------------------

    conda install -c conda-forge opencv


Developing with or without Rust
--------------------------------

OpenPIV supports dual computational backends: **pure Python / SciPy** (default) and **compiled Rust** (via ``openpiv_rust``).

Option A: Developing in Pure Python (without Rust)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You do not need a Rust compiler installed. Simply install OpenPIV in editable mode:

.. code-block:: bash

    poetry install
    # or: pip install -e .

All core functionality, single-pass, and multi-pass deformation will use the optimized
``scipy.fft`` backend automatically.

To run the test suite:

.. code-block:: bash

    pytest openpiv/test

Option B: Developing with the Rust Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have the Rust toolchain (``cargo``) installed:

1. Install ``maturin``:

   .. code-block:: bash

       pip install maturin

2. Compile and link ``openpiv_rust`` directly into your Python environment:

   .. code-block:: bash

       cd crates/openpiv_rust
       maturin develop --release
       cd ../..

3. Verify that the Rust backend is active:

   .. code-block:: bash

       python -c "import openpiv_rust; print('Rust backend ready!')"

Running the Performance Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenPIV includes a dedicated benchmark suite comparing cross-correlation, subpixel
peak finding, and end-to-end multi-pass Windef across backends:

.. code-block:: bash

    python benchmarks/run_benchmarks.py



