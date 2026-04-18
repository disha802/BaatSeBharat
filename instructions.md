Methodology
BaatSeBharat-Fin: Leadership-Driven Longitudinal Market Regime Intelligence

1. System Overview
The proposed BaatSeBharat-Fin framework is a hybrid multimodal system designed to superimpose longitudinal political leadership communication onto historical financial market data in order to detect policy-driven market regime shifts.
The system is composed of two tightly coupled branches:
A Textual Intelligence Branch that extracts structured leadership signals from political rhetoric


A Numerical Intelligence Branch that models counterfactual financial market dynamics


These branches are fused through temporal alignment, probabilistic shock modeling, and causal validation to isolate leadership-driven market effects from background noise.

2. Data Sources and Preprocessing
2.1 Leadership Communication Corpus
The textual dataset consists of ten years of Mann Ki Baat transcripts.
The corpus is segmented at multiple granularities:
Sentence level


Paragraph level


Episode level


Year level


Preprocessing steps include:
Part-of-speech filtering to remove ceremonial and phatic expressions


Removal of Indic stopwords and filler constructions


Lemmatization to stabilize long-term semantic drift


These steps preserve substantive policy content while removing rhetorical formalities, as validated in earlier BaatSeBharat studies.

2.2 Financial Market Data
The numerical dataset includes:
Daily sectoral indices and major equity prices


Trading volume


Volatility proxy (VIX)


Macroeconomic control variables such as interest rates and yield spreads


All numerical data is converted into time-indexed financial events to enable longitudinal modeling.

3. Textual Intelligence Branch
(Leadership Signal Modeling)
3.1 Hybrid Neural-Probabilistic Topic Engine
To overcome instability in single-model topic extraction, the system employs a hybrid ensemble approach:
Sentence-BERT / IndicBERT generate contextual sentence embeddings


Contextualized Topic Models (CTM) combine embeddings with topic distributions


ProdLDA enforces probabilistic topic consistency


Ensemble consensus aggregates outputs into a stable topic set


This architecture reduces topic drift and improves coherence across long horizons.

3.2 Multi-Granularity Topic Aggregation
Topic inference is performed independently at sentence, paragraph, and episode levels.
Hierarchical aggregation is then applied to ensure that topics remain localized and do not bleed across unrelated discourse segments.

3.3 Zero-Shot Automatic Topic Labeling
To eliminate reliance on manual expert labeling:
Zero-shot classification models (BART / DeBERTa) are used


Topics are automatically mapped to high-level policy categories such as Infrastructure, Manufacturing, Digital Economy, or Welfare


This enables scalable and bias-free semantic labeling across a decade of transcripts.

3.4 Sentiment Overlay
Each topic instance is passed through FinBERT to extract financial sentiment signals.
Two orthogonal dimensions are captured:
Optimism intensity


Caution or risk awareness


This separates rhetorical emphasis from policy intent.

4. Numerical Intelligence Branch
(Accuracy-Optimized Market Modeling)
This branch uses only algorithms that directly improve detection accuracy.

4.1 Asset-Specific Baseline Normalization (ASBN)
For each asset or sector, a stable historical period is identified.
The normalized value at time t is computed as:
Z(t) = (x(t) − mean_asset) / std_asset
where mean_asset and std_asset are computed from stable regimes only.
This removes cross-asset variability and improves sensitivity to abnormal behavior.

4.2 Counterfactual Price Trajectory Modeling (CPTM-F)
From the same stable period, a counterfactual “continuation” trajectory is learned that represents expected behavior in the absence of structural change.
Deviation is computed as:
D(t) = (actual_price(t) − expected_price(t)) / std_asset
This formulation distinguishes temporary corrections from structural regime divergence.

4.3 Multi-Scale Temporal Modeling
Trends are computed simultaneously at multiple horizons:
Short-term (5–20 days)


Medium-term (1–3 months)


Long-term (6–24 months)


The final trend signal is a weighted combination:
T(t) = sum over i of (weight_i × trend_i(t))
This captures both fast market reactions and slow structural transitions.

4.4 Trajectory Shape Features
Higher-order temporal dynamics are extracted from price series.
Acceleration is computed as the second derivative of price with respect to time.
These features detect instability buildup before volatility manifests.

4.5 Cross-Indicator Synchrony Detection
To suppress false positives, regime shifts are confirmed only when multiple indicators diverge simultaneously.
Synchrony is measured as the correlation between counterfactual deviations of:
Price
Volatility
Volume
Noise rarely synchronizes across indicators, whereas true regime shifts do.

5. Rhetorical-Financial Superimposition
(Core Novel Contribution)
5.1 Temporal Topic Alignment
To track the evolution of leadership priorities over ten years:
Hungarian matching is applied between topic distributions across years


Jaccard similarity measures topic overlap


Word Mover’s Distance measures semantic drift


This proves topic stability and controlled evolution.

5.2 Probability-Weighted Moment (PWM) Shock Modeling
For each speech episode, the impact on market tails is quantified using probability-weighted moments.
The general form is:
PWM = expected value of (X × F(X)^r)
where:
X is return or volatility


F(X) is the empirical cumulative distribution function


r controls tail sensitivity


This produces a Shock Impact Score measuring leadership influence on extreme market behavior.

6. Multimodal Fusion Strategy
Topic distributions and sentiment scores are injected into the numerical branch as early-warning signals.
These textual signals modulate counterfactual expectations, enabling detection of regime shifts before they appear in price data.
Unlike news-based systems, this approach uses authoritative leadership communication.

7. Causal and Predictive Validation
7.1 Granger Causality Testing
Granger causality tests are conducted to evaluate whether leadership topic intensities predict:
Sectoral volatility


Abnormal returns


Regime transitions


This isolates the predictive contribution of leadership rhetoric beyond macroeconomic variables.

7.2 Ablation Studies
To verify robustness, components are systematically removed:
POS filtering


SBERT embeddings


Counterfactual modeling


PWM shock layer


Performance degradation quantifies each component’s contribution.

8. Evaluation Metrics
Textual modeling is evaluated using:
Topic coherence (Cv, NPMI, UMass)


Topic diversity


Financial modeling is evaluated using:
Regime detection AUROC


Lead time before volatility spikes


False positive rate in stable regimes



9. Product-Grade Implementation
An interactive Streamlit dashboard allows users to:
Navigate sentence-to-year topic hierarchies


Visualize topic-regime overlays


Inspect shock impact scores


Validate topic stability metrics


This bridges research rigor with exploratory usability.

IMPORTANT: Add market predictions with confidence for the future, based on the current trained regime and the leadership signals.