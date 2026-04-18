You are a Python + ML + stocks expert. Refactor, fix, and validate this financial analysis project end-to-end.
## 1. Patch Integration (MANDATORY)
* Find all `.patch/.diff` or patch-like files
* Merge their changes into the main code (focus: `App_v2.py`)
* Resolve conflicts, remove duplicates/incomplete logic
* Delete patch files ONLY after successful integration
## 2. Data (ECB + Fed)
* Collect last 10 years of **official speech transcripts** for ecb and fed too
* Store in transcripts folders like there are some already, put all transcripts in one main folder with subfolders.
## 3. Cleanup
* Keep only necessary files for `App_v2.py`
* Remove unused/experimental code
## 4. Run & Fix Pipeline
Steps:
1. Ingestion
2. Preprocessing
3. NLP features
4. Model (train/infer)
5. App
* Fix all errors with clear corrections
## 5. Market Regime Prediction
* Debug + stabilize outputs
* Improve features (e.g., returns, volatility)
* Redesign if needed
## 6. Stock Impact Fix
Error: “Selected topic not found in merged dataset.”
* Fix merge + topic alignment
* Add fallback (closest topic / available data)
## 7. App Validation
* `App_v2.py` must fully work:
  * Speech analysis
  * Topic extraction
  * Market impact
  * Regime prediction
## OUTPUT
* Clean structure
* Working `App_v2.py` + only necessary files
* Summary of fixes (patches + bugs)
* Run instructions
## RULES
* Integrate patches (do not ignore)
* No unused code
* Keep it simple, production-ready
Start by analyzing patches + project, then proceed stepwise.