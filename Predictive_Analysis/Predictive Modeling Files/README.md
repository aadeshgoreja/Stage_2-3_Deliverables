# Customer churn prediction — Colab + Keras ANN + train-only K-Means

An executable project using **the actual 6,741-row CSV files supplied by the user**, which originate from the [Stage_2_Deliverables GitHub repository](https://github.com/aadeshgoreja/Stage_2_Deliverables). Includes a complete ANN training pipeline, two classical ML comparison models, graphs, saved fitted models, test prediction CSVs, and a Google Colab notebook.

## Start in Google Colab (recommended)

1. **Upload** `Customer_Churn_ANN_Project.zip` to the Colab **Files** panel. This places the archive at `/content/Customer_Churn_ANN_Project.zip`.
2. Open `notebooks/Customer_Churn_ANN_Colab.ipynb` in Colab (File → Upload notebook). You can upload the notebook directly from inside the extracted ZIP on your computer.
3. Choose **Runtime → Run all**. The first cell automatically extracts `/content/Customer_Churn_ANN_Project.zip` into `/content/Customer_Churn_ANN_Project/` if not already extracted, then verifies both CSVs.
4. Training writes an entirely new results folder at `/content/Customer_Churn_ANN_Project/results/`. In Colab, saved pictures display inline and fresh model scores are printed. At the end, the notebook prepares `/content/Customer_Churn_ANN_Project_Colab_Results.zip` containing the rerun outputs for you to download.

**No separate CSV upload, Git clone, or manual path editing is required.** The first code cell offers an upload prompt only if the ZIP is missing. The ZIP already contains the datasets and pretrained artifacts, and Run all reproduces them on Colab.

### Expected Colab paths

```text
/content/Customer_Churn_ANN_Project/data/cleaned_full_dataset.csv
/content/Customer_Churn_ANN_Project/data/clustered_customer_data.csv
/content/Customer_Churn_ANN_Project/src/churn_pipeline.py
/content/Customer_Churn_ANN_Project/src/predict.py
/content/Customer_Churn_ANN_Project/results/metrics.json
/content/Customer_Churn_ANN_Project/results/figures/
/content/Customer_Churn_ANN_Project/results/models/customer_churn_ann.keras
```

## Run the Python source directly

Inside Colab or a Python terminal:

```bash
cd /content/Customer_Churn_ANN_Project
python src/churn_pipeline.py --data /content/Customer_Churn_ANN_Project/data/cleaned_full_dataset.csv --clustered /content/Customer_Churn_ANN_Project/data/clustered_customer_data.csv --output /content/Customer_Churn_ANN_Project/results --epochs 45
python src/predict.py --input /content/Customer_Churn_ANN_Project/results/example_new_customers.csv --output /content/Customer_Churn_ANN_Project/results/sample_predictions.csv
python -m unittest discover -s tests -v
```

For local use on Linux/macOS/Windows, paths are resolved relative to the project by default: `python src/churn_pipeline.py`. Install dependencies via `pip install -r requirements.txt` and ensure **either** `tensorflow` or `torch` is installed for Keras 3. On Colab, TensorFlow is generally preinstalled. In this archive the trained model was fitted using Keras 3 with the **Torch backend** because TensorFlow is not present in the build environment.

## Contents

| Folder/file | Description |
|---|---|
| `data/cleaned_full_dataset.csv` | Original 6,741-row cleaned source (12 predictors + Churn) |
| `data/clustered_customer_data.csv` | Original 6,741-row source with four published clusters and PCA columns |
| `src/churn_pipeline.py` | Full data checks, splits, new clustering, ANN + benchmark training, plotting, evaluation, and export |
| `src/predict.py` | Load fitted models and transform new *already-encoded* source-format rows for inference |
| `notebooks/Customer_Churn_ANN_Colab.ipynb` | Colab-ready notebook with real execution, plots and results |
| `results/models/` | Fitted preprocessor, `.keras` ANN and `.joblib` reference models |
| `results/figures/` | PNGs including EDA, clusters, ANN learning curves, 3 confusion matrices, ROC, PR, comparison and importance |
| `results/metrics.json` | Train/validation/test details; .5 and optimised-threshold metrics; data provenance |
| `results/model_scores.csv` | Test score comparison |
| `results/held_out_test_predictions.csv` | All 1,349 independent test labels/predictions/probabilities |
| `results/example_new_customers.csv` | Ten source-format numeric test rows for demonstrating inference |
| `tests/test_project.py` | Automated tests for dataset, transformations, output files, models and predictions |
| `docs/METHODOLOGY_AND_DATA_LIMITATIONS.md` | Source traceability, design decisions, leakage caveats and interpretation |

## Real initial results (saved in ZIP)

- Rows: **6,741**; original features **12**; new model features **16** (12 + 4 train-only K-Means indicators).
- Splits: **4,313 train / 1,079 validation / 1,349 held-out test**.
- Churn-class test recalls at **validation-optimised F2 cutoffs**: ANN **0.919**, logistic regression **0.927**, random forest **0.902**. High recall trades off overall accuracy and precision.
- Test PR-AUC / Average Precision: ANN **0.587**, logistic regression **0.596**, random forest **0.542**.
- Validation PR-AUC selected **ANN** on this seed; it is NOT the highest test-PR-AUC, demonstrating why selection must not use held-out test results.
- ANN early-stopped after **21 epochs** during the reference run (45-epoch maximum).

Re-running on Colab/TensorFlow may produce slightly different floating-point outcomes due to backend/platform implementation; compare the regenerated scores with `results/metrics.json`.

## DATA CAUTION — input is NOT raw customer records

The source GitHub `tenure` and `MonthlyCharges` columns are **already standardised**, with no upstream raw-to-standardised transformer provided. New prediction rows must use the same encoded and standardised representation. Published `Cluster`/`PCA1`/`PCA2` were calculated on all rows and are excluded from actual model inputs to avoid using precomputed global clustering. We fit new training-only clusters. See `docs/METHODOLOGY_AND_DATA_LIMITATIONS.md` for further limitations.

## License/provenance

Source GitHub: https://github.com/aadeshgoreja/Stage_2_Deliverables — original authorship and any upstream licensing remain with the original repository. The repository does not include a license file in the inspected tree; re-use terms should be verified with its owner before redistribution. This archive is supplied for the user's study and project work with source attribution.
