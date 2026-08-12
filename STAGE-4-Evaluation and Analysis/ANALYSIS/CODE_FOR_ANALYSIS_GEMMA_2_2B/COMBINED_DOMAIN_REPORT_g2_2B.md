# Combined Steering Evaluation Report

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
| MORAL | more ethical/safe | 24 | 16/24 | +0.037 | L16 a=1.5 (+0.550) | L12 a=1 (+0.300, q=+0.493) | L19 a=0.1 (11.0%) | L12 (87.5% positive) |
| LOGIC | more correct / closer to ground truth | 24 | 15/24 | +0.121 | L19 a=0.5 (+0.590) | L19 a=1 (+0.370, q=+0.605) | L19 a=0.7 (17.0%) | L19 (87.5% positive) |
| POLITIC | more right-leaning/conservative | 24 | 5/24 | -0.146 | L16 a=0.5 (+0.610) | L16 a=1 (+0.470, q=+0.603) | L16 a=1 (25.0%) | L16 (50.0% positive) |
| SENTIMENT | more positive sentiment | 16 | 5/16 | -0.094 | L13 a=0.2 (+0.400) | L13 a=0.2 (+0.400, q=+0.463) | L19 a=0.2 (7.0%) | L19 (50.0% positive) |

## Main Cross-Domain Findings

### 1. LOGIC Is The Strongest Improvement-Oriented Domain

LOGIC has the clearest broad improvement pattern:

```text
Positive configs: 15/24
Best raw setting: L19 alpha 0.5, delta +0.590
Most robust layer: L19
```

Layer `19` is especially strong for LOGIC. It is positive in most alpha settings and has the best raw correctness shift. This suggests that later-layer steering is more effective for factual/logical answer correction in this setup.

Paper-facing interpretation:

> LOGIC steering shows the strongest evidence of broad improvement, with layer 19 producing the best raw correctness shift and the most robust positive behavior across steering strengths.

### 2. MORAL Has A Strong Peak But Also Quality Constraints

MORAL has a meaningful best raw result:

```text
Best raw setting: L16 alpha 1.5, delta +0.550
Best practical setting: L12 alpha 1
Most robust layer: L12
```

The key MORAL finding is the separation between peak and practical performance:

- Layer 16 alpha 1.5 gives the strongest ethical-alignment shift.
- Layer 12 alpha 1.0 gives the best practical balance of moral shift and quality.
- Clean success remains low, so many moral gains are not fully usable generations.

Paper-facing interpretation:

> MORAL steering works in selected regimes, but the strongest raw alignment shift is not the same as the best practical setting once output quality is considered.

### 3. POLITIC Shows A Concentrated Rightward Shift At Layer 16

POLITIC should not be written as "better" or "worse." The primary score measures political polarity:

```text
1 = left/progressive
5 = center
10 = right/conservative
```

The result is directional:

```text
Best rightward shift: L16 alpha 0.5, delta +0.610
Positive configs: 5/24
Best clean-positive setting: L16 alpha 1
```

The effect is concentrated in layer 16. Layer 12 and layer 19 are mostly not positive in the rightward direction.

Paper-facing interpretation:

> POLITIC steering exhibits localized directional control: layer 16 produces the clearest rightward polarity shift, while other layers show weaker or negative directional movement.

### 4. SENTIMENT Shows A Narrow Positive-Sentiment Effect

SENTIMENT also needs directional framing:

```text
Positive delta = more positive sentiment
```

The strongest result is:

```text
L13 alpha 0.2, delta +0.400
```

But only 5/16 configurations are positive. The best result is concentrated at layer 13 alpha 0.2.

Paper-facing interpretation:

> SENTIMENT steering produces a positive shift in a narrow regime, especially layer 13 alpha 0.2, but the effect is not broadly robust across layers and alphas.

## Top Configurations By Domain

| Domain | Rank | Layer | Alpha | Delta | Steered Avg | Baseline Avg | Primary Success | Judge Win | Clean Success | Quality Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MORAL | 1 | 16 | 1.5 | +0.550 | 4.56 | 4.01 | 34.0% | 45.0% | 7.0% | +0.010 |
| MORAL | 2 | 12 | 1 | +0.300 | 4.94 | 4.64 | 33.0% | 52.0% | 7.0% | +0.493 |
| MORAL | 3 | 12 | 0.7 | +0.290 | 4.77 | 4.48 | 36.0% | 45.0% | 6.0% | +0.227 |
| LOGIC | 1 | 19 | 0.5 | +0.590 | 3.88 | 3.29 | 30.0% | 31.0% | 12.0% | +0.128 |
| LOGIC | 2 | 16 | 2 | +0.560 | 4.18 | 3.62 | 30.0% | 33.0% | 13.0% | -0.005 |
| LOGIC | 3 | 19 | 2 | +0.520 | 4.26 | 3.74 | 24.0% | 34.0% | 12.0% | +0.410 |
| POLITIC | 1 | 16 | 0.5 | +0.610 | 5.42 | 4.81 | 32.0% | 39.0% | 17.0% | +0.203 |
| POLITIC | 2 | 16 | 1 | +0.470 | 5.11 | 4.64 | 40.0% | 45.0% | 25.0% | +0.603 |
| POLITIC | 3 | 16 | 0.7 | +0.270 | 5.07 | 4.80 | 30.0% | 38.0% | 14.0% | -0.027 |
| SENTIMENT | 1 | 13 | 0.2 | +0.400 | 4.66 | 4.26 | 25.0% | 49.0% | 4.0% | +0.463 |
| SENTIMENT | 2 | 19 | 0.2 | +0.280 | 4.60 | 4.32 | 31.0% | 27.0% | 7.0% | -0.377 |
| SENTIMENT | 3 | 19 | 2 | +0.110 | 4.25 | 4.14 | 26.0% | 44.0% | 6.0% | +0.133 |

## Layer Robustness Across Domains

| Domain | Layer | Positive Configs | Mean Delta | Best Alpha | Best Delta | Volatility SD | Mean Quality | Mean Clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MORAL | 12 | 7/8 | +0.136 | 1 | +0.300 | 0.191 | +0.271 | 5.0% |
| MORAL | 16 | 5/8 | +0.005 | 1.5 | +0.550 | 0.284 | -0.189 | 5.5% |
| MORAL | 19 | 4/8 | -0.031 | 0.7 | +0.170 | 0.180 | +0.002 | 7.2% |
| LOGIC | 12 | 2/8 | -0.184 | 1.5 | +0.400 | 0.416 | +0.125 | 11.5% |
| LOGIC | 16 | 6/8 | +0.229 | 2 | +0.560 | 0.264 | +0.082 | 10.8% |
| LOGIC | 19 | 7/8 | +0.318 | 0.5 | +0.590 | 0.196 | +0.359 | 10.8% |
| POLITIC | 12 | 1/8 | -0.291 | 2 | +0.080 | 0.325 | +0.475 | 9.1% |
| POLITIC | 16 | 4/8 | +0.031 | 0.5 | +0.610 | 0.393 | +0.188 | 16.9% |
| POLITIC | 19 | 0/8 | -0.180 | 0.2 | +0.000 | 0.158 | -0.282 | 15.1% |
| SENTIMENT | 13 | 1/8 | -0.091 | 0.2 | +0.400 | 0.233 | -0.022 | 4.8% |
| SENTIMENT | 19 | 4/8 | -0.098 | 0.2 | +0.280 | 0.278 | -0.120 | 6.9% |

## Cross-Domain Pattern

The most useful cross-domain insight is:

```text
Different domains prefer different layers.
```

Observed best raw settings:

- MORAL: layer 16, alpha 1.5
- LOGIC: layer 19, alpha 0.5
- POLITIC: layer 16, alpha 0.5
- SENTIMENT: layer 13, alpha 0.2

This suggests that steering behavior is not universal across attributes. Instead, layer choice depends on the target domain:

- LOGIC benefits from later-layer intervention.
- MORAL and POLITIC show strong mid-layer effects.
- SENTIMENT has its strongest result at an earlier/mid layer.

This is a strong paper insight:

> Activation steering is domain-localized: the most effective intervention layer varies by behavioral attribute.

## Alpha Behavior

Across domains, alpha response is not monotonic. Larger alpha does not always produce stronger or better steering.

Examples:

- MORAL: alpha 1.5 is best at layer 16, but not robust across all layers.
- LOGIC: layer 19 alpha 0.5 is best raw, while alpha 1.0 is best quality-positive.
- POLITIC: layer 16 alpha 0.5 gives the strongest rightward shift, but alpha 1.0 gives better quality and clean success.
- SENTIMENT: the best positive shift is alpha 0.2, not a large alpha.

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

- MORAL: L19 alpha 0.1, 11.0%
- LOGIC: L19 alpha 0.7, 17.0%
- POLITIC: L16 alpha 1, 25.0%
- SENTIMENT: L19 alpha 0.2, 7.0%

Clean success is generally much lower than raw directional shift. This is important because it shows that steering can move the target attribute without always producing high-quality usable text.

Paper-facing claim:

> Clean-success analysis reveals a gap between attribute movement and usable generation quality, making quality-preserving evaluation necessary for activation steering.

## Failure Signals

| Domain | Top Failure Signals |
| --- | --- |
| MORAL | repetition (57.6%), repetitive (55.9%), coherence (44.7%), incoherent (18.2%), unethical (7.0%) |
| LOGIC | incorrect (51.8%), repetitive (32.7%), incoherent (19.3%), contradicts (14.2%), irrelevant (12.9%) |
| POLITIC | repetitive (39.2%), incoherent (36.5%), coherence (28.2%), repetition (24.1%), irrelevant (23.0%) |
| SENTIMENT | repetition (54.8%), repetitive (46.3%), coherence (41.2%), off-topic (8.8%), incoherent (5.8%) |

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
| MORAL | L16 a=1.5 | L12 a=1.0 | L12 | Ethical shift exists, but quality matters. |
| LOGIC | L19 a=0.5 | L19 a=1.0 | L19 | Strongest improvement-oriented domain. |
| POLITIC | L16 a=0.5 | L16 a=1.0 | L16 | Localized rightward polarity control. |
| SENTIMENT | L13 a=0.2 | L13 a=0.2 | L19 by clean success / L13 by peak | Narrow positive sentiment shift. |

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
