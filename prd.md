Below is a **Product Requirements Document (PRD)** for what you need to build (a reproducible Python pipeline + report outputs) based on the project brief. 

---

## PRD: “Explicit Song Tracks – KDD Classification Pipeline & Impact Analysis”

### 1) Background & Purpose

This project’s goal is to **run the first iteration of the KDD process** (data extraction → profiling → preparation → modeling → evaluation → critical analysis) on the Kaggle “explicit-song-tracks” dataset, with emphasis on **measuring the impact of choices** at each step on classification performance. 

### 2) Users & Primary Use Case

* **Primary user:** you (student/data scientist) producing a technical report for a supervisor-style audience.
* **Use case:** run a controlled experiment pipeline to create multiple prepared datasets, compare them using Naïve Bayes at each prep step, then train multiple classifiers on the selected prepared dataset and summarize findings.

### 3) Scope

#### In scope

* Python project using **pandas, matplotlib, scikit-learn**. 
* Read Kaggle source files, join into a single numeric feature table (n rows × m cols). 
* Perform **profiling** on dimensionality, distribution, sparsity. 
* Perform **data preparation in the prescribed order** (as per Figure 1 described in the PDF):

  1. Missing values handling (imputation alternatives)
  2. Scaling alternatives
  3. Balancing alternatives (applied only on training split; requires partitioning before balancing) 
* At each preparation step: apply **two alternative techniques**, train **Naïve Bayes** on resulting datasets, compare performance, choose the best (or keep previous if no improvement/residual). 
* Modeling: train and tune **Naïve Bayes, Logistic Regression, KNN, Decision Tree, Random Forest** on the final selected prepared dataset. 
* Evaluation: metrics + charts + parameter impact, best model per method, variable relevance when possible. 
* Critical cross-model analysis (largest grading weight). 

#### Out of scope

* Full multi-iteration KDD loops (only first iteration). 
* Complex processing of non-numeric features beyond the optional ones mentioned (album_type, release_date, artist_genres). 
* AutoML frameworks (discouraged). 

### 4) Assumptions & Constraints

* Dataset is from the specified Kaggle competition. 
* **Non-numeric variables must be discarded** unless optionally processed; if processed, code must be included and described. 
* sklearn cannot train on missing values; must ensure final dataset has none. 
* Scaling “shouldn’t change NB results” theoretically; if applied, justify. 
* Balancing must be applied **only to training data**, so split must happen before balancing. 
* Deliverable required by class is a report, but you’re asking to build “whatever is needed”: this PRD defines the pipeline + artifacts that make the report easy to write.

### 5) Success Criteria

* End-to-end pipeline runs from raw Kaggle files to:

  * a single joined numeric dataset
  * profiling outputs
  * prepared dataset candidates per step with NB comparisons
  * final trained/tuned models for each method
  * evaluation visuals + tables
  * serialized results to reproduce report claims
* Reproducible: single command/notebook run with fixed random seeds.

---

## 6) Functional Requirements

### FR1 — Data Extraction & Integration

**Goal:** produce a single pandas DataFrame `X` and target vector `y`.

**Requirements**

* Load all provided Kaggle source files.
* Join/merge into one table by keys as needed.
* Drop non-numeric columns by default.
* Optional: implement feature engineering for:

  * `album_type`
  * `release_date`
  * `artist_genres`
    If implemented: keep it clearly modular and document it. 

**Outputs**

* `data/raw_joined.csv` (optional)
* `data/processed/base_numeric.parquet` (recommended)
* Data dictionary summary (column names, types, null counts)

**Acceptance criteria**

* Joined dataset has shape (n, m) with only numeric columns in `X`.
* `y` extracted correctly; no row misalignment after joins.

---

### FR2 — Data Profiling (Dimensionality, Distribution, Sparsity)

**Dimensionality**

* Report: #rows, #features, target class counts, feature type summary.

**Distribution**

* For numeric features: summary stats, skewness indicators, hist/boxplots for selected top features (or auto-select by variance/skew).
* Identify outliers, heavy tails, near-constant features.

**Sparsity**

* Missing values per feature, % missing, visualize missingness (bar chart; optional heatmap).
* Zero-inflation (fraction of zeros per feature) if applicable.

**Outputs**

* `reports/profiling/` charts (png)
* `artifacts/profiling_summary.json` (stats)

**Acceptance criteria**

* Profiling section can be written directly from saved outputs and includes the three required perspectives. 

---

### FR3 — Data Preparation Methodology (Sequential, NB-gated)

The pipeline must implement the **stepwise process** described on page 3–4: for each step, generate two alternative datasets, train NB, compare, choose best, proceed. 

#### FR3.1 — Partitioning

* Split into train/test (or train/validation/test).
* Use stratification on target.
* Store split indices to guarantee comparability across experiments.

**Acceptance criteria**

* All comparisons use identical split.

#### FR3.2 — Missing Values (2 alternatives)

Apply only if missingness exists; must still mention if skipped. 

Two techniques (example set; you can pick equivalent):

* A1: SimpleImputer(strategy="median")
* A2: SimpleImputer(strategy="mean") (or KNNImputer if allowed/desired)

Train NB on each and compare to “previous dataset” baseline.

#### FR3.3 — Scaling (2 alternatives)

Apply only if justified; note that NB theoretically shouldn’t change. 
Two techniques:

* B1: StandardScaler
* B2: MinMaxScaler (or RobustScaler)

Evaluate via NB; choose best (or keep previous).

#### FR3.4 — Balancing (2 alternatives, train-only)

Must be applied **only on training data**, after splitting. 
Two techniques (depending on allowed libs; if avoiding imbalanced-learn, use sklearn-friendly options):

* C1: class_weight where applicable (for later models) — for NB you may instead use:
* C1: Random over-sampling implemented manually on train set (simple resample)
* C2: Random under-sampling implemented manually

Evaluate via NB on balanced-train → evaluate on untouched validation/test.

**Outputs for FR3**

* A results table per step: dataset variant → NB metrics
* Selected “winner” dataset object and its transformation pipeline

**Acceptance criteria**

* Clear NB comparison at each step and a final chosen prepared dataset to be used in Modeling. 

---

### FR4 — Modeling (Multiple Methods + Parameter Impact)

Train all models using the **same final prepared dataset** from FR3 (as required). 

Models required:

1. Naïve Bayes (baseline + maybe variants if appropriate)
2. Logistic Regression
3. KNN
4. Decision Tree
5. Random Forest 

**Parameter impact requirement**
For each model, define a small but meaningful grid of hyperparameters and run controlled comparisons (GridSearchCV or manual loops allowed; no AutoML). 

Minimum suggested parameter sets:

* Logistic Regression: C, penalty (if solver supports), class_weight (if used)
* KNN: k, distance metric (if feasible), weights
* Decision Tree: max_depth, min_samples_leaf, criterion
* Random Forest: n_estimators, max_depth, max_features, min_samples_leaf

**Outputs**

* `artifacts/models/` fitted estimators (joblib)
* `artifacts/model_results.csv` with:

  * model family
  * params
  * metrics
  * train time (optional)
* Parameter impact plots (line plots or small multiples)

**Acceptance criteria**

* For each technique: you can identify and justify the best model and show how parameters affected performance. 

---

### FR5 — Evaluation & Reporting Assets

Evaluation must include:

* Confidence measures (e.g., accuracy, precision, recall, F1, ROC-AUC when applicable)
* Evaluation charts (confusion matrix, ROC curve if binary, PR curve optional)
* Comparison across models
* Relevant variables “when possible”:

  * Logistic Regression coefficients
  * Tree/Forest feature importance
  * (NB: limited interpretability but can show per-class likelihood insights if desired) 

**Outputs**

* `reports/figures/` charts
* `artifacts/final_summary.md` (optional auto-generated narrative bullets)
* `artifacts/feature_importance_{model}.csv`

**Acceptance criteria**

* Evaluation section supports the three required analysis levels:

  1. parameter impact
  2. best model per technique + performance
  3. relevant variables when possible 

---

### FR6 — Critical Analysis Support (Cross-model Comparison)

Provide structured comparisons:

* When each model wins/loses and why given dataset properties (e.g., scaling sensitivity, class imbalance, non-linearity)
* Error analysis: which class is harder, typical confusion patterns
* Stability: variance across folds/splits
* Tradeoffs: interpretability vs performance vs compute

**Acceptance criteria**

* Produces enough evidence (tables + plots) to write the critical analysis (30% weight). 

---

## 7) Non-Functional Requirements

* Reproducibility: fixed `random_state` everywhere.
* Modularity: each KDD stage can be run independently.
* Traceability: every dataset variant and model run has an ID and stored parameters.
* Runtime: should finish on a typical laptop (keep grids modest).

---

## 8) Deliverables

Even though the course asks only for a report, the “thing to develop” is:

1. **Python codebase / notebook pipeline**
2. **Artifacts folder** with:

   * processed datasets
   * transformation configs
   * model results tables
   * trained models
   * charts
3. **Report-ready outputs** matching template needs (you’ll paste results/figures in)

---

## 9) Proposed Project Structure (Implementation-Oriented)

* `src/`

  * `01_extract.py` (load + join + numeric filter)
  * `02_profile.py` (dimensionality/distribution/sparsity)
  * `03_prepare.py` (NB-gated stepwise prep)
  * `04_model.py` (train/tune all methods)
  * `05_evaluate.py` (charts + comparisons + feature importance)
  * `utils/` (config, metrics, plotting, io)
* `configs/`

  * `experiment.yaml` (splits, grids, chosen techniques)
* `data/` raw/processed
* `artifacts/` results/models/summaries
* `reports/figures/`

---

## 10) Milestones

1. Extraction + joined numeric dataset
2. Profiling outputs
3. Preparation step runner (NB comparisons + selection)
4. Modeling runner (grids + best models)
5. Evaluation charts + feature relevance exports
6. Critical-analysis comparison pack (final tables/plots)

---

If you want, I can also generate a **starter repository skeleton** (folders + config + stub scripts) and a **results schema** (CSV columns for every experiment run) so you can start coding immediately.
