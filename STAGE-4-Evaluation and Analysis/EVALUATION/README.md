# Stage 4: Automated Multi-Judge LLM Evaluation

> **Location:** [`STAGE-4-Evaluation and Analysis/EVALUATION`](file:///E:/SAE-STATSTEER/STAGE-4-Evaluation%20and%20Analysis/EVALUATION)  
> **Submodule:** LLM-as-a-Judge Evaluation Harness

---

## 🎯 1. Objective of this Folder

The objective of **`EVALUATION`** is to perform automated, multi-judge LLM evaluation of steered completions generated in Stage 3.

Using Gemini 2.5 Flash (`gemini-2.5-flash` via Google GenAI SDK / Vertex AI) with randomized A/B prompt pairing, `evaluator.py` scores generated responses across four quantitative metrics ($1\text{--}10$ scale):
1. **Primary Score:** Domain target score (Logical validity, Moral permissibility, Rightward political stance, Positive sentiment).
2. **Relevance Score:** Adherence to original prompt context.
3. **Richness Score:** Detail and depth of text generation.
4. **Coherence Score:** Grammatical fluency and absence of repetitive artifacts.

For LOGIC prompts, objective ground-truth answer keys from `logic-prompt-keys.json` are injected into the evaluation prompt to ensure precise scoring.

---

## 🚀 2. How to Run the Scripts & Execution Order

Execution consists of running `evaluator.py` on Stage 3 output result files:

### Setup Environment
Ensure your Google GenAI / Vertex AI API credentials are configured:
```bash
export GEMINI_API_KEY="your_api_key_here"
# OR set GCP project credentials for Vertex AI:
# gcloud auth application-default login
```

### Run Evaluation
Copy or link Stage 3 output files (`*_results.json` or `*_checkpoint.json`) into the working directory (or specify via `CONFIG["results_dir"]`), then run:

```bash
python evaluator.py
```

---

## 📥 3. Required Inputs for Each Script

### Script: [`evaluator.py`](file:///E:/SAE-STATSTEER/STAGE-4-Evaluation%20and%20Analysis/EVALUATION/evaluator.py)
* **Required Files:**
  1. `*_results.json` / `*_checkpoint.json`: Output files from Stage 3 containing prompt texts, baseline responses ($\alpha=0$), and steered responses ($\alpha > 0$).
  2. `logic-prompt-keys.json`: Ground-truth logic reference keys for LogicBench prompts.
* **Environment Dependencies:**
  * `google-genai` SDK (`from google import genai`)
  * `pydantic` for structured response validation (`TextEvaluation` schema)
  * Valid Gemini API key (`GEMINI_API_KEY`) or Google Cloud Project ID (`thesis-2-areeb`).
* **Outputs Generated:**
  * Evaluated JSON files containing pairwise scores, metric breakdowns, winner flags ("A", "B", "Same"), and judge justifications saved in `results/`.