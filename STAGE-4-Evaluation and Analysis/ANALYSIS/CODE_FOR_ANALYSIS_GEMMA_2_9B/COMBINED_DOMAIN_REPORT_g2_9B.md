# Gemma 2 9B Combined Steering Evaluation Report

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
| MORAL | more ethical/safe | 32 | 20/32 | +0.050 | L38 a=0.5 (+0.700) | L38 a=0.2 (+0.370, q=+0.527) | L38 a=0.1 (17.0%) | L38 (87.5% positive) |
| LOGIC | more correct / closer to ground truth | 32 | 25/32 | +0.281 | L19 a=0.1 (+1.160) | L19 a=0.5 (+0.570, q=+1.070) | L26 a=1 (26.0%) | L19 (100.0% positive) |
| POLITIC | more right-leaning/conservative | 32 | 15/32 | -0.011 | L38 a=1.5 (+0.612) | L26 a=0.7 (+0.459, q=+0.398) | L19 a=0.5 (30.6%) | L31 (62.5% positive) |
| SENTIMENT | more positive sentiment | 32 | 22/32 | +0.166 | L26 a=0.3 (+0.640) | L31 a=0.7 (+0.250, q=+0.407) | L31 a=0.1 (17.0%) | L31 (87.5% positive) |

## Main Cross-Domain Findings

### 1. LOGIC Is The Strongest Improvement-Oriented Domain

LOGIC has the clearest broad improvement pattern:

```text
Positive configs: 25/32
Best raw setting: L19 alpha 0.1, delta +1.160
Most robust layer: L19
```

Layer `19` is especially strong for LOGIC. It is positive in most alpha settings and layer `19` has the best raw correctness shift. This suggests that the effective LOGIC intervention region for Gemma 2 9B is concentrated around these evaluated layers rather than uniformly distributed across the model.

Paper-facing interpretation:

> LOGIC steering shows the strongest evidence of broad improvement, with layer 19 producing the best raw correctness shift and layer 19 showing the most robust positive behavior across steering strengths.

### 2. MORAL Has A Strong Peak But Also Quality Constraints

MORAL has a meaningful best raw result:

```text
Best raw setting: L38 alpha 0.5, delta +0.700
Best practical setting: L38 alpha 0.2
Most robust layer: L38
```

The key MORAL finding is the separation between peak and practical performance:

- Layer 38 alpha 0.5 gives the strongest ethical-alignment shift.
- Layer 38 alpha 0.2 gives the best practical balance of positive moral shift and quality.
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
Best rightward shift: L38 alpha 1.5, delta +0.612
Positive configs: 15/32
Best clean-positive setting: L19 alpha 0.5
```

The strongest rightward effect appears at layer `38`, while the most robust rightward layer is `31`. This means POLITIC steering is localized, but for Gemma 2 9B the relevant layers differ from the smaller-model setting.

Paper-facing interpretation:

> POLITIC steering exhibits localized directional control: layer 38 produces the clearest rightward polarity shift, while robustness is concentrated around layer 31.

### 4. SENTIMENT Shows A Narrow Positive-Sentiment Effect

SENTIMENT also needs directional framing:

```text
Positive delta = more positive sentiment
```

The strongest result is:

```text
L26 alpha 0.3, delta +0.640
```

22/32 configurations are positive. The best result is concentrated at layer 26 alpha 0.3, while the most robust layer is 31.

Paper-facing interpretation:

> SENTIMENT steering produces a positive shift in selected regimes, especially layer 26 alpha 0.3, with broader robustness concentrated at layer 31.

## Top Configurations By Domain

| Domain | Rank | Layer | Alpha | Delta | Steered Avg | Baseline Avg | Primary Success | Judge Win | Clean Success | Quality Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MORAL | 1 | 38 | 0.5 | +0.700 | 5.19 | 4.49 | 45.0% | 50.0% | 12.0% | +0.040 |
| MORAL | 2 | 38 | 1.5 | +0.600 | 5.00 | 4.40 | 42.0% | 51.0% | 13.0% | +0.160 |
| MORAL | 3 | 38 | 0.1 | +0.580 | 4.88 | 4.30 | 41.0% | 49.0% | 17.0% | +0.410 |
| LOGIC | 1 | 19 | 0.1 | +1.160 | 5.09 | 3.93 | 33.0% | 37.0% | 24.0% | +0.618 |
| LOGIC | 2 | 31 | 1 | +0.950 | 4.86 | 3.91 | 36.0% | 42.0% | 22.0% | +0.193 |
| LOGIC | 3 | 19 | 1.5 | +0.760 | 4.71 | 3.95 | 29.0% | 36.0% | 17.0% | +0.065 |
| POLITIC | 1 | 38 | 1.5 | +0.612 | 5.37 | 4.76 | 34.7% | 42.9% | 25.5% | +0.252 |
| POLITIC | 2 | 26 | 0.7 | +0.459 | 5.24 | 4.79 | 33.7% | 36.7% | 19.4% | +0.398 |
| POLITIC | 3 | 26 | 0.3 | +0.388 | 5.38 | 4.99 | 34.7% | 38.8% | 21.4% | +0.235 |
| SENTIMENT | 1 | 26 | 0.3 | +0.640 | 4.67 | 4.03 | 33.0% | 38.0% | 12.0% | +0.110 |
| SENTIMENT | 2 | 19 | 0.5 | +0.500 | 4.74 | 4.24 | 38.0% | 46.0% | 13.0% | +0.317 |
| SENTIMENT | 3 | 26 | 1.5 | +0.480 | 4.62 | 4.14 | 34.0% | 43.0% | 10.0% | +0.007 |

## Layer Robustness Across Domains

| Domain | Layer | Positive Configs | Mean Delta | Best Alpha | Best Delta | Volatility SD | Mean Quality | Mean Clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MORAL | 19 | 6/8 | +0.045 | 2 | +0.200 | 0.118 | -0.292 | 11.2% |
| MORAL | 26 | 4/8 | -0.029 | 1.5 | +0.250 | 0.169 | -0.098 | 12.5% |
| MORAL | 31 | 3/8 | -0.181 | 0.7 | +0.220 | 0.294 | -0.042 | 8.9% |
| MORAL | 38 | 7/8 | +0.365 | 0.5 | +0.700 | 0.277 | +0.188 | 12.4% |
| LOGIC | 19 | 8/8 | +0.559 | 0.1 | +1.160 | 0.341 | +0.428 | 20.2% |
| LOGIC | 26 | 4/8 | -0.026 | 0.7 | +0.520 | 0.441 | -0.053 | 16.8% |
| LOGIC | 31 | 7/8 | +0.461 | 1 | +0.950 | 0.323 | +0.274 | 20.4% |
| LOGIC | 38 | 6/8 | +0.130 | 1 | +0.460 | 0.302 | -0.052 | 16.7% |
| POLITIC | 19 | 3/8 | -0.070 | 0.7 | +0.357 | 0.302 | -0.287 | 21.4% |
| POLITIC | 26 | 4/8 | +0.027 | 0.7 | +0.459 | 0.319 | +0.096 | 18.0% |
| POLITIC | 31 | 5/8 | +0.000 | 0.2 | +0.367 | 0.275 | -0.016 | 22.8% |
| POLITIC | 38 | 3/8 | -0.001 | 1.5 | +0.612 | 0.344 | -0.121 | 19.3% |
| SENTIMENT | 19 | 6/8 | +0.256 | 0.5 | +0.500 | 0.211 | +0.082 | 9.9% |
| SENTIMENT | 26 | 6/8 | +0.295 | 0.3 | +0.640 | 0.261 | -0.076 | 10.0% |
| SENTIMENT | 31 | 7/8 | +0.193 | 1 | +0.440 | 0.221 | +0.111 | 10.4% |
| SENTIMENT | 38 | 3/8 | -0.081 | 0.3 | +0.190 | 0.150 | +0.063 | 10.5% |

## Cross-Domain Pattern

The most useful cross-domain insight is:

```text
Different domains prefer different layers.
```

Observed best raw settings:

- MORAL: layer 38, alpha 0.5
- LOGIC: layer 19, alpha 0.1
- POLITIC: layer 38, alpha 1.5
- SENTIMENT: layer 26, alpha 0.3

This suggests that steering behavior is not universal across attributes. Instead, layer choice depends on the target domain:

- LOGIC's strongest raw shift is at layer 19.
- MORAL's strongest raw shift is at layer 38.
- POLITIC's strongest rightward shift is at layer 38.
- SENTIMENT's strongest positive shift is at layer 26.

This is a strong paper insight:

> Activation steering is domain-localized: the most effective intervention layer varies by behavioral attribute.

## Alpha Behavior

Across domains, alpha response is not monotonic. Larger alpha does not always produce stronger or better steering.

Examples:

- MORAL: alpha 0.5 is best at layer 38, while alpha 0.2 at layer 38 is better for quality-positive behavior.
- LOGIC: layer 19 alpha 0.1 is best raw, while layer 19 alpha 0.5 is best quality-positive.
- POLITIC: layer 38 alpha 1.5 gives the strongest rightward shift, while layer 26 alpha 0.7 gives the best quality-positive setting.
- SENTIMENT: layer 26 alpha 0.3 gives the strongest positive shift.

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

- MORAL: L38 alpha 0.1, 17.0%
- LOGIC: L26 alpha 1, 26.0%
- POLITIC: L19 alpha 0.5, 30.6%
- SENTIMENT: L31 alpha 0.1, 17.0%

Clean success is generally much lower than raw directional shift. This is important because it shows that steering can move the target attribute without always producing high-quality usable text.

Paper-facing claim:

> Clean-success analysis reveals a gap between attribute movement and usable generation quality, making quality-preserving evaluation necessary for activation steering.

## Failure Signals

| Domain | Top Failure Signals |
| --- | --- |
| MORAL | repetitive (58.8%), repetition (50.5%), coherence (41.1%), incoherent (17.3%), unethical (9.8%) |
| LOGIC | incorrect (53.3%), repetitive (27.7%), incoherent (17.7%), contradicts (11.5%), irrelevant (9.8%) |
| POLITIC | repetitive (34.6%), incoherent (25.9%), coherence (25.4%), repetition (19.0%), irrelevant (17.5%) |
| SENTIMENT | repetitive (49.6%), repetition (48.5%), coherence (38.2%), off-topic (7.2%), incoherent (7.1%) |

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
| MORAL | L38 a=0.5 | L38 a=0.2 | L38 | Ethical shift exists, but quality matters. |
| LOGIC | L19 a=0.1 | L19 a=0.5 | L19 | Strongest improvement-oriented domain. |
| POLITIC | L38 a=1.5 | L26 a=0.7 | L31 | Localized rightward polarity control. |
| SENTIMENT | L26 a=0.3 | L31 a=0.7 | L31 | Positive sentiment shift in selected regimes. |

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
