# Gemma 3 4B Combined Steering Evaluation Report

This report combines the completed analyses for:

- MORAL
- LOGIC
- POLITIC
- SENTIMENT

It is intended as a paper-facing synthesis: not every per-domain table is repeated, but the core cross-domain findings, useful caveats, and recommended claims are collected in one place.

## Critical Interpretation Rule

The primary score has different meanings across domains.

| Domain | Primary Score Meaning | How To Interpret Positive Delta |
|---|---|---|
| MORAL | Ethical/safety alignment | Higher is more ethical/safe, so positive delta is an improvement. |
| LOGIC | Correctness against ground truth | Higher is more correct, so positive delta is an improvement. |
| POLITIC | Political polarity, 1 left to 10 right | Positive delta means more right-leaning, not automatically better. |
| SENTIMENT | Sentiment polarity, 1 negative to 10 positive | Positive delta means more positive, not automatically better unless positivity is the target. |

This distinction is essential for paper writing. MORAL and LOGIC can be described as improvement-oriented domains. POLITIC and SENTIMENT should be described as **directional control** domains.

## Executive Summary

| Domain | Primary Meaning | Configs | Positive Configs | Mean Delta | Best Raw | Best Quality-Positive | Best Clean-Positive | Most Robust Layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MORAL | more ethical/safe | 32 | 15/32 | -0.012 | L9 a=0.7 (+0.380) | L9 a=0.3 (+0.260, q=+0.533) | L9 a=0.7 (10.0%) | L22 (75.0% positive) |
| LOGIC | more correct / closer to ground truth | 32 | 22/32 | +0.140 | L17 a=1 (+0.750) | L22 a=0.5 (+0.460, q=+0.535) | L17 a=1 (26.0%) | L22 (100.0% positive) |
| POLITIC | more right-leaning/conservative | 32 | 14/32 | -0.051 | L22 a=2 (+0.796) | L22 a=2 (+0.796, q=+0.537) | L22 a=2 (34.7%) | L22 (62.5% positive) |
| SENTIMENT | more positive sentiment | 32 | 14/32 | -0.026 | L22 a=1 (+0.540) | L9 a=1.5 (+0.120, q=+0.473) | L22 a=0.3 (10.0%) | L9 (62.5% positive) |

## Main Cross-Domain Findings

### 1. LOGIC Is The Strongest Improvement-Oriented Domain

LOGIC has the clearest broad improvement pattern:

```text
Positive configs: 22/32
Best raw setting: L17 alpha 1, delta +0.750
Most robust layer: L22
```

Layer `22` is especially strong for LOGIC. It is positive in most alpha settings and layer `17` has the best raw correctness shift. This suggests that the effective LOGIC intervention region for Gemma 3 4B is concentrated around these evaluated layers rather than uniformly distributed across the model.

Paper-facing interpretation:

> LOGIC steering shows the strongest evidence of broad improvement, with layer 17 producing the best raw correctness shift and layer 22 showing the most robust positive behavior across steering strengths.

### 2. MORAL Has A Strong Peak But Also Quality Constraints

MORAL has a meaningful best raw result:

```text
Best raw setting: L9 alpha 0.7, delta +0.380
Best practical setting: L9 alpha 0.3
Most robust layer: L22
```

The key MORAL finding is the separation between peak and practical performance:

- Layer 9 alpha 0.7 gives the strongest ethical-alignment shift.
- Layer 9 alpha 0.3 gives the best practical balance of positive moral shift and quality.
- Clean success remains low, so many moral gains are not fully usable generations.

Paper-facing interpretation:

> MORAL steering works in selected regimes, but the strongest raw alignment shift is not the same as the best practical setting once output quality is considered.

### 3. POLITIC Shows A Concentrated Rightward Shift

POLITIC should not be written as "better" or "worse." The primary score measures political polarity:

```text
1 = left/progressive
5 = center
10 = right/conservative
```

The result is directional:

```text
Best rightward shift: L22 alpha 2, delta +0.796
Positive configs: 14/32
Best clean-positive setting: L22 alpha 2
```

The strongest rightward effect appears at layer `22`, while the most robust rightward layer is `22`. This means POLITIC steering is localized, but for Gemma 3 4B the relevant layers differ from the smaller-model setting.

Paper-facing interpretation:

> POLITIC steering exhibits localized directional control: layer 22 produces the clearest rightward polarity shift, while robustness is concentrated around layer 22.

### 4. SENTIMENT Shows A Narrow Positive-Sentiment Effect

SENTIMENT also needs directional framing:

```text
Positive delta = more positive sentiment
```

The strongest result is:

```text
L22 alpha 1, delta +0.540
```

14/32 configurations are positive. The best result is concentrated at layer 22 alpha 1, while the most robust layer is 9.

Paper-facing interpretation:

> SENTIMENT steering produces a positive shift in selected regimes, especially layer 22 alpha 1, with broader robustness concentrated at layer 9.

## Top Configurations By Domain

| Domain | Rank | Layer | Alpha | Delta | Steered Avg | Baseline Avg | Primary Success | Judge Win | Clean Success | Quality Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MORAL | 1 | 9 | 0.7 | +0.380 | 4.79 | 4.41 | 37.0% | 53.0% | 10.0% | +0.530 |
| MORAL | 2 | 29 | 0.7 | +0.360 | 4.95 | 4.59 | 33.0% | 45.0% | 9.0% | +0.307 |
| MORAL | 3 | 29 | 0.3 | +0.350 | 4.55 | 4.20 | 28.0% | 38.0% | 7.0% | +0.177 |
| LOGIC | 1 | 17 | 1 | +0.750 | 4.44 | 3.69 | 31.0% | 40.0% | 26.0% | +0.065 |
| LOGIC | 2 | 22 | 1.5 | +0.650 | 4.17 | 3.52 | 27.0% | 32.0% | 19.0% | +0.282 |
| LOGIC | 3 | 9 | 0.7 | +0.560 | 4.11 | 3.55 | 25.0% | 28.0% | 19.0% | +0.177 |
| POLITIC | 1 | 22 | 2 | +0.796 | 5.52 | 4.72 | 45.9% | 54.1% | 34.7% | +0.537 |
| POLITIC | 2 | 17 | 2 | +0.633 | 5.68 | 5.05 | 43.9% | 42.9% | 27.6% | -0.048 |
| POLITIC | 3 | 9 | 0.2 | +0.490 | 5.50 | 5.01 | 48.0% | 54.1% | 33.7% | +0.323 |
| SENTIMENT | 1 | 22 | 1 | +0.540 | 4.39 | 3.85 | 33.0% | 44.0% | 10.0% | +0.360 |
| SENTIMENT | 2 | 29 | 1.5 | +0.350 | 3.97 | 3.62 | 29.0% | 39.0% | 5.0% | -0.207 |
| SENTIMENT | 3 | 9 | 0.2 | +0.320 | 4.37 | 4.05 | 29.0% | 35.0% | 8.0% | +0.127 |

## Layer Robustness Across Domains

| Domain | Layer | Positive Configs | Mean Delta | Best Alpha | Best Delta | Volatility SD | Mean Quality | Mean Clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MORAL | 9 | 5/8 | +0.076 | 0.7 | +0.380 | 0.216 | +0.237 | 5.0% |
| MORAL | 17 | 0/8 | -0.330 | 0.3 | -0.170 | 0.247 | -0.091 | 4.9% |
| MORAL | 22 | 6/8 | +0.130 | 0.3 | +0.320 | 0.190 | +0.153 | 5.9% |
| MORAL | 29 | 4/8 | +0.076 | 0.7 | +0.360 | 0.194 | -0.048 | 5.4% |
| LOGIC | 9 | 5/8 | +0.128 | 0.7 | +0.560 | 0.256 | +0.077 | 16.6% |
| LOGIC | 17 | 6/8 | +0.200 | 1 | +0.750 | 0.299 | +0.033 | 18.6% |
| LOGIC | 22 | 8/8 | +0.324 | 1.5 | +0.650 | 0.207 | +0.154 | 15.6% |
| LOGIC | 29 | 3/8 | -0.092 | 0.7 | +0.460 | 0.387 | -0.203 | 16.0% |
| POLITIC | 9 | 3/8 | -0.214 | 0.2 | +0.490 | 0.463 | -0.132 | 25.3% |
| POLITIC | 17 | 4/8 | +0.028 | 2 | +0.633 | 0.344 | -0.024 | 25.9% |
| POLITIC | 22 | 5/8 | +0.177 | 2 | +0.796 | 0.401 | +0.197 | 27.4% |
| POLITIC | 29 | 2/8 | -0.194 | 0.5 | +0.184 | 0.268 | -0.153 | 24.4% |
| SENTIMENT | 9 | 5/8 | +0.050 | 0.2 | +0.320 | 0.185 | +0.146 | 5.2% |
| SENTIMENT | 17 | 1/8 | -0.189 | 0.7 | +0.020 | 0.161 | -0.361 | 4.6% |
| SENTIMENT | 22 | 3/8 | -0.006 | 1 | +0.540 | 0.247 | +0.131 | 6.0% |
| SENTIMENT | 29 | 5/8 | +0.040 | 1.5 | +0.350 | 0.197 | -0.235 | 5.1% |

## Cross-Domain Pattern

The most useful cross-domain insight is:

```text
Different domains prefer different layers.
```

Observed best raw settings:

- MORAL: layer 9, alpha 0.7
- LOGIC: layer 17, alpha 1
- POLITIC: layer 22, alpha 2
- SENTIMENT: layer 22, alpha 1

This suggests that steering behavior is not universal across attributes. Instead, layer choice depends on the target domain:

- LOGIC's strongest raw shift is at layer 17.
- MORAL's strongest raw shift is at layer 9.
- POLITIC's strongest rightward shift is at layer 22.
- SENTIMENT's strongest positive shift is at layer 22.

This is a strong paper insight:

> Activation steering is domain-localized: the most effective intervention layer varies by behavioral attribute.

## Alpha Behavior

Across domains, alpha response is not monotonic. Larger alpha does not always produce stronger or better steering.

Examples:

- MORAL: alpha 0.7 is best at layer 9, while alpha 0.3 at layer 9 is better for quality-positive behavior.
- LOGIC: layer 17 alpha 1 is best raw, while layer 22 alpha 0.5 is best quality-positive.
- POLITIC: layer 22 alpha 2 gives the strongest rightward shift, while layer 22 alpha 2 gives the best quality-positive setting.
- SENTIMENT: layer 22 alpha 1 gives the strongest positive shift.

Paper-facing claim:

> Steering strength interacts nonlinearly with layer and domain; increasing alpha is not a reliable way to monotonically increase useful steering.

## Quality And Clean Success

Clean success is stricter than raw directional success:

```text
delta_primary > 0
AND steered_relevance >= 7
AND steered_richness >= 4
AND steered_coherence >= 4
```

Best clean-positive settings:

- MORAL: L9 alpha 0.7, 10.0%
- LOGIC: L17 alpha 1, 26.0%
- POLITIC: L22 alpha 2, 34.7%
- SENTIMENT: L22 alpha 0.3, 10.0%

Clean success is generally much lower than raw directional shift. This is important because it shows that steering can move the target attribute without always producing high-quality usable text.

Paper-facing claim:

> Clean-success analysis reveals a gap between attribute movement and usable generation quality, making quality-preserving evaluation necessary for activation steering.

## Failure Signals

| Domain | Top Failure Signals |
| --- | --- |
| MORAL | repetitive (55.7%), repetition (50.6%), coherence (46.4%), incoherent (17.1%), unethical (8.6%) |
| LOGIC | incorrect (54.7%), repetitive (29.2%), incoherent (17.3%), contradicts (12.2%), irrelevant (9.1%) |
| POLITIC | repetitive (49.1%), coherence (33.4%), repetition (22.3%), incoherent (11.8%), off-topic (3.5%) |
| SENTIMENT | repetition (53.4%), repetitive (48.6%), coherence (44.4%), off-topic (7.1%), incoherent (5.9%) |

The repeated appearance of repetition, coherence, and incoherence signals means that the dominant failure mode across domains is often generation quality, not only failure to shift the target attribute.

This helps justify why the paper should include both target-shift metrics and quality-preservation metrics.

## Recommended Main Paper Figures

Use a small number of high-value figures in the main paper:

1. **Four-domain primary delta heatmap grid**
   - MORAL, LOGIC, POLITIC, SENTIMENT
   - Shows domain-specific layer-alpha behavior.

2. **Pareto plot: target shift vs quality delta**
   - Either one combined figure or one per domain in appendix.
   - Shows best practical settings.

3. **Transition matrices**
   - Main text: MORAL and LOGIC.
   - Appendix: POLITIC and SENTIMENT.

4. **Layer robustness table**
   - Compact table showing best layer, positive config rate, and volatility.

## Recommended Main Paper Table

A compact table should include:

| Domain | Best Raw | Best Practical | Robust Layer | Main Interpretation |
|---|---|---|---|---|
| MORAL | L9 a=0.7 | L9 a=0.3 | L22 | Ethical shift exists, but quality matters. |
| LOGIC | L17 a=1 | L22 a=0.5 | L22 | Strongest improvement-oriented domain. |
| POLITIC | L22 a=2 | L22 a=2 | L22 | Localized rightward polarity control. |
| SENTIMENT | L22 a=1 | L9 a=1.5 | L9 | Positive sentiment shift in selected regimes. |

## Strongest Overall Paper Claim

The strongest combined claim is:

> Activation steering produces measurable but domain-specific behavioral shifts. LOGIC and MORAL show improvement-oriented effects, while POLITIC and SENTIMENT demonstrate directional control over polarity dimensions. The optimal layer and alpha vary substantially by domain, and the largest raw shift is not always the best practical setting once output quality, clean success, and regression behavior are considered.

## Claims To Avoid

Avoid these claims:

- "Steering improves all domains."
- "Higher alpha always improves steering."
- "Political steering is better when the score is higher."
- "Sentiment steering is better when the score is higher" unless the target is explicitly positive sentiment.
- "Judge win ratio alone proves success."

Use these instead:

- "Steering induces domain-specific directional shifts."
- "Some domains show improvement-oriented gains."
- "Layer-alpha interaction is central."
- "Quality-preserving and clean-success metrics are necessary."
- "POLITIC and SENTIMENT require directional interpretation."

## Report Links

- [MORAL report](outputs_moral/MORAL_SUMMARY_REPORT.md)
- [LOGIC report](outputs_logic/LOGIC_SUMMARY_REPORT.md)
- [POLITIC report](outputs_politic/POLITIC_SUMMARY_REPORT.md)
- [SENTIMENT report](outputs_sentiment/SENTIMENT_SUMMARY_REPORT.md)
