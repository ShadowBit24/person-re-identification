# Person Re-Identification System

M.Sc. project, Lakehead University.

## Overview

A computer-vision system that re-identifies a person across different camera views — matching images of the same individual despite changes in viewpoint, lighting, and pose. This implementation uses a spatial-temporal person re-identification approach: image features are combined with spatial-temporal information (camera and timestamp distributions) to improve matching, with re-ranking applied to the results.

## What I did

- Set up and adapted the training/evaluation pipeline in PyTorch
- Trained and evaluated models on the Market-1501 and DukeMTMC-reID benchmark datasets
- Ran spatial-temporal model generation and re-ranking evaluation, producing result files and distribution plots
- Documented the system, methodology, and results in the project report (Spatial-Temporal_Person_Re-identification_report.pdf, in this repo)

## Repository contents

- model.py / model_chs.py — network definitions
- train_market.py / train_duke.py / train_market_chsv.py — training entry points
- test_st_market.py / test_st_duke.py — feature extraction / testing
- gen_st_model_market.py / gen_st_model_duke.py — spatial-temporal model generation
- evaluate_st.py / evaluate_rerank_market.py / evaluate_rerank_duke.py / gen_rerank_all_scores_mat.py — evaluation and re-ranking
- prepare.py — dataset preparation
- random_erasing.py — data augmentation
- re_ranking.py — re-ranking implementation
- plot_st_distribution.py — spatial-temporal distribution plots

## Basis & credit

The code builds on the established spatial-temporal person re-identification research codebase and method (Wang et al., "Spatial-Temporal Person Re-Identification," AAAI 2019). It is shared here as my M.Sc. project work — training, evaluation, and adaptation — not as an original-from-scratch implementation.

Note: the datasets (Market-1501, DukeMTMC-reID) are not included; they are available from their original sources for research use.

## Skills demonstrated

Python · PyTorch · deep learning · computer vision · model evaluation & testing methodology
