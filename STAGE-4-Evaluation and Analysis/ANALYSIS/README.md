# Stage 4: Statistical Analysis & Paper-Facing Report Generation

> **Location:** [`STAGE-4-Evaluation and Analysis/ANALYSIS`](file:///E:/SAE-STATSTEER/STAGE-4-Evaluation%20and%20Analysis/ANALYSIS)  
> **Submodule:** Analysis & Reporting

---

## 🎯 1. Objective of this Folder

The objective of **`ANALYSIS`** is to aggregate raw LLM judge evaluations from Stage 4 Evaluation and convert them into statistical metrics, paper-facing comparison tables, and diagnostic reports.

Key analytical metrics computed include:
* **Primary Delta ($\Delta_p$):** Mean score change in primary domain attribute ($1\text{--}10$ scale).
* **Quality Delta ($\Delta_q$):** Mean score change across Relevance, Richness, and Coherence.
* **Raw Success Rate (SR):** Percentage of prompts where steered primary score exceeds baseline.
* **Clean Success Rate (Clean SR):** Percentage of prompts meeting $\Delta_p > 0$ with Relevance $\ge 7$, Richness $\ge 4$, and Coherence $\ge 4$.
* **Holistic Correlations:** Spearman rank correlations testing depth vs. steering strength effects.

---

## 🚀 2. How to Run the Scripts & Execution Order

Execution flows from per-model domain analysis to cross-model holistic reporting:

### Step 1: Run Per-Model Domain Analysis
Execute domain analysis for each target model directory:

```bash
# For Gemma 2 2B:
cd CODE_FOR_ANALYSIS_GEMMA_2_2B
python analyze_domains.py
python analyze_moral.py
python generate_combined_report.py
cd ..

# Repeat for Gemma 2 9B and Gemma 3 4B in their respective directories.
```

### Step 2: Generate All-Models Consolidated Paper Report
From `ANALYSIS/`, compile cross-model summary tables into `ALL_MODELS_ALL_DOMAINS_REPORT.md`:

```bash
python generate_all_models_report.py
```

### Step 3: Run Holistic Meta-Statistical Analysis
From `ANALYSIS/`, execute statistical modeling and correlation tests:

```bash
python STATISTICAL_ANALYSIS/holistic_statistics.py
```

---

## 📥 3. Required Inputs for Each Script

### 1. Per-Model Analysis Scripts (`analyze_domains.py`, `analyze_moral.py`, `generate_combined_report.py`)
* **Inputs:** Evaluated JSON result files exported by `evaluator.py` (e.g. `*_evaluated.json` or `results/` folder inside each model directory).
* **Outputs:** `COMBINED_DOMAIN_REPORT_g2_2B.md`, `COMBINED_DOMAIN_REPORT_g2_9B.md`, `COMBINED_DOMAIN_REPORT_g3_4b.md`.

### 2. [`generate_all_models_report.py`](file:///E:/SAE-STATSTEER/STAGE-4-Evaluation%20and%20Analysis/ANALYSIS/generate_all_models_report.py)
* **Inputs:** Individual model report folders (`CODE_FOR_ANALYSIS_GEMMA_2_2B`, `CODE_FOR_ANALYSIS_GEMMA_2_9B`, `CODE_FOR_ANALYSIS_GEMMA_3_4B`) containing parsed CSV metrics.
* **Outputs:** `ALL_MODELS_ALL_DOMAINS_REPORT.md` (unified summary table).

### 3. [`STATISTICAL_ANALYSIS/holistic_statistics.py`](file:///E:/SAE-STATSTEER/STAGE-4-Evaluation%20and%20Analysis/ANALYSIS/STATISTICAL_ANALYSIS/holistic_statistics.py)
* **Inputs:** All evaluated domain metrics across models and layers.
* **Outputs:** `STATISTICAL_ANALYSIS/HOLISTIC_STATISTICAL_REPORT.md` and statistical diagnostic plots in `STATISTICAL_ANALYSIS/outputs/`.
