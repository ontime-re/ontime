.. _user_guide:

User Guide
==========


This guide gives you a topic-by-topic introduction to onTime, showcasing
concrete use cases through runnable notebooks.


.. note::

    This guide is a work in progress. If you have any questions or suggestions, please feel free to contact us.


The user guide is organized into three sections:

1. **Core**: the fundamental building blocks of the library.
2. **Module**: higher-level features built on top of the core, such as benchmarking and ML preprocessing.
3. **Context**: applied, real-world scenarios showing onTime in action.


Core
----

The core notebooks introduce onTime's foundational objects — time series,
detectors, generators, models, plots and processors — and how they fit
together. Start here if you're new to the library.

.. toctree::
    :maxdepth: 2

    0_core/0.1_time-series
    0_core/0.1.1_time-series_data-loading
    0_core/0.2_detectors
    0_core/0.3_generators
    0_core/0.4_models
    0_core/0.5_plots
    0_core/0.6_processors
    0_core/0.7_custom-class
    0_core/1-models/1.0-autoencoder

Module
------

The module notebooks cover features built on top of the core: data handling
and datasets, anomaly frequency analysis, preprocessing for PyTorch and
TensorFlow, and model benchmarking.

.. toctree::
    :maxdepth: 2

    1_module/0-data/1.0-data-dataset
    1_module/0.6-anomaly-frequency
    1_module/1-processing/1.0-preprocessing-common
    1_module/1-processing/pytorch/1.0_pytorch-dataset
    1_module/1-processing/tensorflow/1.0_tensorflow-dataset
    1_module/3-benchmarking/3.0-benchmarking

Context
-------

The context notebooks show onTime applied to real-world, domain-specific
scenarios rather than isolated features.

.. toctree::
    :maxdepth: 1

    2_context/2.0-context-common
