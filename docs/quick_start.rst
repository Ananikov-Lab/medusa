===========
Quick start
===========

Reading mass-spectra
====================
Currently we support only mzXML-file format. Vendor formats can be converted using `ProteoWizard`_. The spectra can be converted under non-Windows operating system using Wine. A good option is to use a `Docker-container`_ with everything already installed. Large-scale mass-spectra analysis requires usage of cron or Apache Airflow to manage pipelines.

Once you have a converted spectrum, you can use MEDUSA to analyze it:

.. code-block:: python

    from mass_automation.experiment import Experiment

    exp = Experiment('spectrum.mzXML', n_scans=128, n_points=6)


If the file is obtained from TOF-spectrometer or recorded during LC-MS experiment, it will contain multiple spectra. ``Experiment`` object contains multiple ``Spectrum`` objects:

.. code-block:: python

    spectrum = exp[0]

If you want to get array of masses or intensities, use methods:

.. code-block:: python

    masses = spectrum.masses
    ints = spectrum.ints

Chromatogram can be plotted with ``get_chromatogram`` method. Chromatogram is a list of total intensities of spectra, which are contained in ``Experiment``.


.. warning::
    Memory leaks are possible here. For operations with many files it is highly recommended to write a script which takes filename as an argement and saves the intermediate results.

Mass-spectra data preprocessing
===============================
Once you need to combine several spectra in ``Experiment``, you can use ``summarize`` method:

.. code-block:: python

    spectrum = exp.summarize(4, 8)

``Spectrum`` objects with indexes from 4 to 7 inclusively will be summarized.

Once you need to encode ``Spectrum`` object in vector, ``vectorize`` method can help:

.. code-block:: python

    spec_vector = spectrum.vectorize(min_mass = 150,
                                     max_mass = 1000,
                                     method = np.mean)

It is required for PCA plots and cluster maps, implemented in MEDUSA.

Deisotoping
===========
All deisotoping-related code is located at ``mass_automation.deisotoping`` module.
First, you have to create Python dictionary with structure and pickle it:

.. code-block:: python

    model = {'min_distance' : 0.021,
             'delta' : 0.007,
             'model' : ML model object}

    with open(path_to_the_saving_pickle, 'wb') as f:
        pickle.dump(model, f)

You may change hyperparameters and model object if necessary. if you use ``LinearDeisotoper``, you don't have to add model object in dictionary.

After that, you should write path to the pickle file, you created before, in ``load`` method of ``MlDeisotoper`` or ``LinearDeisotoper`` class.

.. code-block:: python

    deisotoper = MlDeisotoper().load(path)
    predictions = deisotoper(spectrum)

``predictions`` is a numpy array with labels, where -1 is a noise label. Labels are integer numbers beginning from 0.

Deisotoping can be visualised with ``plot_spectrum`` function (see "Data visualization")

Formula analysis
================
In the paper, we describe both models for classification and regression on elements. Tne related code is located in ``mass_automation.formula`` module.

One of the main objects here is ``Formula`` object.

.. code-block:: python

    from mass_automation.formula import Formula

    formula = Formula('C2H5OHNa', charge='1')


It can be used to calculate isotopic distributions and vectors. It has a couple of useful methods:

Formula name in ``str`` type:

.. code-block:: python

    formula_name = formula.str_formula

Formula name in ``dict`` type:

.. code-block:: python

    formula_dictionary = formula.dict_formula

Isotopic distribution (Arrays of masses and intensities of isotopologues). You can also delete aggregated isotopic variants with ``del_isotopologues``:

.. code-block:: python

    masses, ints = formula.isodistribution()
    masses, ints = del_isotopologues(masses, ints)

Compound presence verification
==============================

Once you need to check if your substance is in the spectrum or not, you may use ``check_presence``.
It returns cosine distance, which is correlated with our algorithm's confidence.

.. code-block:: python

    cosine_distance = check_presence(spectrum, formula)

It also supplied with visualization function ``plot_compare`` (see "Data visualization")

Data visualization
==================
* Spectra plotting

Once you need to look at spectra, ``plot_spectrum`` is solving your problem!

.. code-block:: python

    plot_spectrum(spectrum,
                  drawtype="plot",
                  x_left=800,
                  x_right=1200,
                  y_max=-1)

y_max = -1 means that maximal value on intensity scale will be the maximal intensity in interval from x_left to x_right.

If you want to look at the result of deisotoping, do this:

.. code-block:: python

    plot_spectrum(spectrum,
                  labels=predictions)

``predictions`` is a numpy array with labels, where -1 is a noise label.

* Compound presence verification visualization

Once you need to check presence of compound in spectra with your eyes automatically use ``plot_compare`` function

.. code-block:: python

    plot_compare(spectrum,
                 Formula('C2H6OH'),
                 cal_error=0.006,
                 dist_error=0.003)

You can change parameters to optimize algorithm.

* PCA-maps

PCA-map is useful tool to provide clustering of complex mixtures.
Pipeline for creation:

1. Create Excel or Pandas DataFrame. An example of typical dataframe is here:

2. Than you have to vectorize all your spectra. It can be provided via ``vectorize`` method (see "Reading mass-spectra").

3. Set parameter ``required_keys`` as list of names in spectra vectors dictionary you have to image on PCA-map.

4. Than you have to set ``colormapper`` to define legend on PCA-map.

.. code-block:: python

    colormapper = {
        'class1': 'red',
        'class2': 'green',
        'class3': 'blue',
        'class4': 'yellow'
    }

5. Add ``class_decoder`` as a sequence of object classes in the same order as objects' spectra
in spectra dictionary (It can be difficult to understand, so watch example here).

6. Add ``name_decoder`` if you want your spectra be annotated on a map

7. Last step. Use ``plot_pca``. You can change PCA on t-SNE if it is necessary.

.. code-block:: python

    plot_pca(spec_vecs,
             required_keys,
             class_decoder,
             name_decoder,
             colormapper,
             dim_red='TSNE',
             IsNameDecoder=True)

result:

* Cluster maps
* Element highlighting


.. _ProteoWizard: https://proteowizard.sourceforge.io
.. _Docker-container: https://hub.docker.com/r/chambm/pwiz-skyline-i-agree-to-the-vendor-licenses
