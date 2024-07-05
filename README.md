## Mapping semantic spaces through PLSR for translation esploring intelligibility effects

[Montecchiari_Project_Report.pdf](https://github.com/memonji/gender-biases-exploration/files/14270670/Montecchiari_Project_Report.pdf)

**Author:** Emma Angela Montecchiari

**Course:** Università di Trento - Machine Learning for NLP 2022/23

**Date:** July 5, 2024

## Project Proposal 
###  Abstract

This study explores the efficacy of semantic spaces in facilitating word translation tasks among closely related languages—Catalan, Italian, and English. Leveraging Partial Least Squares Regression (PLSR), the research investigates how linguistic intelligibility influences model performance. Results indicate that languages with closer lexical proximity exhibit higher translation accuracy.

## Contents

In this repository, you will find:

- **main.py:** User interactive script to train, test and evaluate the PLSR model for translation task.
  The stored outputs (nns with optimal parameters) are in in *./results_bestncomps/manual_selection/**.
- **data_handling.py:** User interactive script to (A) extract key words from the semantic spaces and store them; (B) plot 2D and 3D representation of the spaces.
  The stored material is in *./pairs/* and *./spaces/figures/* folders.
- **cosine_similarity.py:** Script to compute cosine similarity distances between outputted nns and gold standard translations and store them.
  The stored material is in *./results_bestncomps/cosine_similarity/* with the best performing parameters outputs.

**Pre-trained semantic spaces:** Downloaded pre-trained Catalan, English and Italian semantic spaces. Stored in *./spaces/*

### Requirements
The code has been implemented over a Python 3.11.3 version and with a Conda (23.5.2) environment.

Required packages:
- scikit-learn 1.2.2
- numpy 1.24.3
- docopt 0.6.2
- matplotlib 3.7.1
