# Notebooks

Executed Google Colab notebooks from the project period (M.Sc. Computer Science, Lakehead University), recovered from Google Drive (Colab Notebooks folder) on 2026-10-23 and preserved as run, including cell outputs. File names were made descriptive on import; the original Drive names and dates are recorded here for provenance.

| File | Original Drive name | Last modified (Drive) | Contents |
|---|---|---|---|
| 01_market1501_training_resnet50_pcb.ipynb | Untitled4.ipynb | 2023-11-25 | Extracts Market-1501, runs prepare.py, then trains with train_market.py (PCB, ResNet50, random erasing 0.5, train_all) |
| 02_pretrained_model_run.ipynb | Spatial-Temporal-Re-identification-pretrained_model_run | 2023-11-27 | Stages the project files from Drive, runs a training session (60-epoch schedule) and evaluates with the pretrained checkpoint |
| 03_market1501_dataset_preparation.ipynb | Untitled5.ipynb | 2024-02-07 | Market-1501 archive extraction and PyTorch dataset preparation |
| 04_training_resnet101_variant.ipynb | Untitled6.ipynb | 2024-02-23 | Extracts the ST-ReID codebase, runs prepare.py, then trains with train_market_res101_a.py (PCB, ResNet101 variant) |

## Note on outputs

Outputs are preserved exactly as executed in Colab. They were not edited, re-run, or cleaned up for this repository. The evaluation printout that recurs across these notebooks is the output obtained with the authors' released pretrained checkpoint, which is why it is identical in each notebook that evaluates it. The method's published figures remain attributed to Wang et al. (see "Results" and "Basis & credit" in the main README) and are not re-claimed here. The training logs in the notebooks are from the sessions shown.
