.. pyclstm documentation master file, created by
   sphinx-quickstart on Thu Oct  6 12:57:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyclstm
=======

Installation
------------

**Requirements:**
    - A recent version of Eigen (>= 3.3) with development headers
    - A C++ compiler (g++ is recommended)
    - Cython

**Installation:**

.. code::

    $ pip install git+https://github.com/jbaiter/clstm.git@cython


Example Usage
-------------

**Training:**

Refer to ``run_uw3_500.py`` in the root directory for a more comprehensive
example.

.. code::

    import pyclstm
    ocr = pyclstm.ClstmOcr()
    ocr.prepare_training(
        graphemes=graphemes,  # A list of characters the engine is supposed to recognize
    )

    # line_img can be an image loaded with PIL/Pillow or a numpy array
    for line_img, ground_truth in training_data:
        ocr.train(line_img, ground_truth)
    ocr.save("my_model.clstm")


**Recognition:**

.. code::

    import pyclstm
    ocr = pyclstm.ClstmOcr()
    ocr.load("my_model.clstm")
    text = ocr.recognize(line_img)


API Reference
-------------

.. automodule:: pyclstm
    :members:
    :undoc-members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

