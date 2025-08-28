StomataPy Documentation
=======================

StomataPy is a collection of resources to encourage community contributions for generalized stomatal segmentation.

1. We released `generalized stomatal segmentation models <https://huggingface.co/aliasz/StomataPy400K-Models>`_ trained on `400K diverse annotations <https://huggingface.co/datasets/aliasz/StomataPy400K>`_ 
2. Community can correct minor model prediction errors with our annotation tool `ISAT-SAM <https://github.com/yatengLG/ISAT_with_segment_anything>`_ 
3. The corrected annotations iteratively improve models, benefiting the entire community 

Features
--------

* Instance segmentation on stomatal complex; and pavment cells (require high quality images)
* Sementic segmentation of sub stomatal classes (stoma, outer ledge, pore)
* Derive cell area, length, width, rotation
* Customizable for other target such as trichome
* Batch analyze images and save results as Excel/JSON files 

Quick Start
-----------

1. **Installation**: Clone the repository and install dependencies
2. **Models**: Download pre-trained models from our Hugging Face repository
3. **Annotation**: Use ISAT-SAM to create or correct annotations
4. **Community**: Contribute your annotations to improve models for everyone

Resources
---------

* `Hugging Face Models <https://huggingface.co/aliasz/StomataPy400K-Models>`_
* `Dataset <https://huggingface.co/datasets/aliasz/StomataPy400K>`_
* `ISAT-SAM Tool <https://github.com/yatengLG/ISAT_with_segment_anything>`_

.. note::
   This project is under active development. Feedback is welcome!

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   installation



