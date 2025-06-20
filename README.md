# MGCL-ST Experiment README

This repository contains the code and instructions for conducting experiments on spatial transcriptomics data using the MGCL-ST framework. The experiments integrate data enhancement (via the MGCL component) and super-resolution gene expression imputation (via the ST component). The workflows are implemented in Python and executed through Jupyter Notebook files, covering three 10X dataset slices (151673, 151674, 151675) and the CSCC dataset slice. The experiments include data preprocessing, enhancement, super-resolution imputation, and TLS (Tumor-Lymphocyte Score) identification for the CSCC dataset. This README was last updated at 11:04 AM CST on Wednesday, June 18, 2025.

## Directory Structure

- **Data**: Contains raw and processed spatial transcriptomics data.
- **Fig**: Stores generated figures or visualizations.
- **MGCL**: Contains scripts and data for spatial transcriptomics data enhancement.
- **Result**: Stores experimental results and model outputs.
- **ST**: Contains scripts and data for super-resolution gene expression imputation.
- **Notebooks**:
  - 151673.ipynb: Processes the 151673 10X dataset slice for data enhancement and super-resolution imputation.
  - 151674.ipynb: Processes the 151674 10X dataset slice for data enhancement and super-resolution imputation.
  - 151675.ipynb: Processes the 151675 10X dataset slice for data enhancement and super-resolution imputation.
  - CSCC.ipynb: Processes the CSCC dataset slice for data enhancement, super-resolution imputation, and TLS identification.
- **Other Files**:
  - environment.yml: Configuration file for the Conda environment.
  - MGCL-ST_618.pdf: Essay.

## Environment Setup

1. Activate the Conda environment using the provided configuration:
   ```bash
   conda env create -f environment.yml
   conda activate dsq_env
   ```
2. Ensure all dependencies (e.g., PyTorch, anndata, scipy) are installed. If requirement.txt is used, install additional packages:
   ```
   pip install -r requirement.txt
   ```

## Experiments

### 10X Dataset Slices (151673, 151674, 151675)
1. Running 151673 Experiment:
   - Open 151673.ipynb in Jupyter Notebook.
   - Run all cells to load data, perform data enhancement (MGCL), and apply super-resolution imputation (ST). The notebook imports necessary modules from the MGCL and ST folders.
2. Running 151674 Experiment:
   - Open 151674.ipynb in Jupyter Notebook.
   - Run all cells to process the 151674 slice with data enhancement and super-resolution imputation.
3. Running 151675 Experiment:
   - Open 151675.ipynb in Jupyter Notebook.
   - Run all cells to process the 151675 slice with data enhancement and super-resolution imputation.

### CSCC Dataset
1. Running CSCC Experiment:
   - Open CSCC.ipynb in Jupyter Notebook.
   - Run all cells to load the CSCC dataset, perform data enhancement, apply super-resolution imputation, and compute TLS identification. The notebook integrates modules from MGCL and ST folders.

## Notes
- Ensure the dsq_env environment is activated before running the notebooks.
- The .ipynb files handle data loading, preprocessing, enhancement, imputation, and TLS analysis in a streamlined manner.
- Results and intermediate outputs are saved in the Result folder.
- If CUDA-related issues occur (e.g., driver mismatch), run on CPU by setting device='cpu' in the model initialization or consult troubleshooting steps.

## Contact
For any issues or questions, please contact Siqi Ding at d15995291636@163.com.