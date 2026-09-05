.. _installation_instruction:

========================
Installation instruction
========================

.. _dependencies:

Dependencies
============

OpenPIV would not have been possible if other great open source projects did not
exist. We make extensive use of code and tools that other people have created, so 
you should install them before you can use OpenPIV.

The dependencies are:


* `Python <http://python.org/>`_
* `Scipy <http://numpy.scipy.org/>`_
* `Numpy <http://www.scipy.org/>`_
* `scikit-image <http://scikit-image.org/>`_   


Installation
============

Recommended: Use `uv` (fastest)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`uv <https://github.com/astral-sh/uv>`_ is a fast Python package installer and resolver written in Rust.
It provides faster and more reliable package installation compared to traditional tools.

To install OpenPIV with uv::

    pip install uv
    uv pip install openpiv

Or use `pip` (standard)
^^^^^^^^^^^^^^^^^^^^^^^^

::

    pip install openpiv

.. warning::
    **Conda packages are no longer actively maintained.** The conda-forge package may be outdated.
    
    If you previously installed OpenPIV via conda, you can migrate to pip or uv::
    
        # Remove the conda package
        conda remove openpiv
        
        # Install with pip or uv
        pip install openpiv
        # or
        uv pip install openpiv
    
.. note::
    **Precompiled Binary Wheels with Rust Acceleration**:
    Standard installation via ``pip install openpiv`` or ``uv pip install openpiv`` automatically
    downloads pre-compiled binary wheels with the multithreaded Rust acceleration engine enabled.
    No Rust compiler or C++ build tools are required on user machines.

Get OpenPIV source code!
========================

To develop or build OpenPIV from source, clone the repository with git::

    git clone https://github.com/OpenPIV/openpiv-python.git
    cd openpiv-python

Building the Rust Acceleration Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compile the high-performance parallel Rust extension locally, you need the Rust toolchain (``cargo`` and ``rustc``) and `maturin <https://github.com/PyO3/maturin>`_::

    pip install maturin
    maturin develop --release -m crates/openpiv_rust/Cargo.toml
    pip install -e .

Once built, OpenPIV will automatically detect the compiled extension and enable accelerated execution.    

Experience problems?
====================
If you encountered some issues, found difficult to install OpenPIV following these instructions
please register and write on our Google groups forum https://groups.google.com/g/openpiv-users , so that we can help you and 
improve this page!





