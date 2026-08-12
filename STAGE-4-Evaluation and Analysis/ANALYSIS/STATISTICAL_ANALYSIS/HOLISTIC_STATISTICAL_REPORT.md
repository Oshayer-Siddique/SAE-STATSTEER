# Holistic Statistical Analysis Report

This report performs a statistical analysis across all completed steering results:

- 3 models: Gemma 2 2B, Gemma 2 9B, Gemma 3 4B
- 4 domains: MORAL, LOGIC, POLITIC, SENTIMENT
- all evaluated layers and alpha values

The unit of most statistical tests is the **paired sample-level delta**:

```text
delta_primary = steered_primary_score - baseline_primary_score
```

For MORAL and LOGIC, positive delta means improvement. For POLITIC and SENTIMENT, positive delta means directional movement: rightward political polarity or more positive sentiment.

## What Each Statistical Test Answers

| Analysis | Question It Answers | Why It Matters |
|---|---|---|
| Bootstrap CI | How stable is the mean steering effect? | Gives uncertainty around mean delta without assuming normality. |
| Paired t-test | Is mean delta significantly different from zero? | Tests average paired improvement/shift. |
| Wilcoxon signed-rank | Is the paired shift nonzero without normality assumptions? | Better for ordinal, tied, non-normal 1-10 scores. |
| Sign/binomial test | Are positive shifts more frequent than negative shifts? | Robust to score magnitude and handles tied-heavy data. |
| FDR correction | Which effects survive many comparisons? | Prevents overclaiming from hundreds of config tests. |
| Effect size | How large is the effect, not just whether p < 0.05? | Important for practical significance. |
| Spearman correlation | Which metrics move with primary steering shift? | Shows relationships among quality, clean success, judge wins, alpha, depth. |
| Alpha quadratic regression | Is alpha response nonlinear? | Tests whether larger alpha is not simply better. |
| Relative-layer regression | Do early/middle/late layers matter systematically? | Tests layer-depth trends across models/domains. |
| Pareto frontier | Which configs are not dominated across target shift, quality, and clean success? | Finds practically strong settings, not only raw max-delta settings. |

## Dataset Scale

| Quantity | Value |
|---|---:|
| Sample-level rows | 34272 |
| Configurations tested | 344 |
| Model-domain pairs | 12 |

## Significance Summary By Model And Domain

`Significant Positive Configs` means the configuration has positive mean delta and passes all three FDR-corrected tests:

```text
paired t-test FDR < 0.05
Wilcoxon FDR < 0.05
sign/binomial FDR < 0.05
mean delta > 0
```

| Model | Domain | Positive Configs | Significant Positive Configs | Mean Delta | Mean Tie Rate |
| --- | --- | --- | --- | --- | --- |
| Gemma 2 9B | LOGIC | 25/32 | 0/32 | +0.281 | 50.4% |
| Gemma 3 4B | POLITIC | 14/32 | 0/32 | -0.051 | 24.0% |
| Gemma 3 4B | LOGIC | 22/32 | 0/32 | +0.140 | 55.3% |
| Gemma 2 9B | MORAL | 20/32 | 0/32 | +0.050 | 38.2% |
| Gemma 2 9B | SENTIMENT | 22/32 | 0/32 | +0.166 | 48.3% |
| Gemma 2 9B | POLITIC | 15/32 | 0/32 | -0.011 | 40.0% |
| Gemma 2 2B | POLITIC | 5/24 | 0/24 | -0.146 | 41.8% |
| Gemma 2 2B | LOGIC | 15/24 | 0/24 | +0.121 | 54.8% |
| Gemma 2 2B | MORAL | 16/24 | 0/24 | +0.037 | 42.1% |
| Gemma 3 4B | SENTIMENT | 14/32 | 0/32 | -0.026 | 48.7% |
| Gemma 2 2B | SENTIMENT | 5/16 | 0/16 | -0.094 | 47.1% |
| Gemma 3 4B | MORAL | 15/32 | 0/32 | -0.012 | 47.7% |

## Strongest Statistically Supported Configurations

No individual configuration passed the strict criterion of positive mean delta plus FDR-corrected paired t-test, Wilcoxon, and sign/binomial test all below 0.05. This should be reported as a conservative result: the strongest candidates show positive effects and useful confidence intervals, but they should not be described as passing this all-tests FDR threshold.

This is the safest section for paper claims about strict per-configuration statistical reliability. Because the all-tests FDR threshold is very conservative for tied ordinal judge scores, the practical interpretation should also use bootstrap confidence intervals, effect sizes, clean success, and Pareto status.

## Largest Raw Effects

| Model | Domain | Layer | Alpha | Mean Delta | 95% Boot CI | Pos/Tie/Neg | t p(FDR) | Wilcoxon p(FDR) | Sign p(FDR) | Cohen dz | Rank-biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 2 9B | LOGIC | 19 | 0.1 | +1.160 | [+0.330, +1.960] | 33/50/17 | 0.4746 | 0.5746 | 1.0000 | 0.276 | 0.433 |
| Gemma 2 9B | LOGIC | 31 | 1 | +0.950 | [+0.160, +1.700] | 36/43/21 | 0.7450 | 0.8303 | 1.0000 | 0.226 | 0.339 |
| Gemma 3 4B | POLITIC | 22 | 2 | +0.796 | [+0.163, +1.398] | 45/28/25 | 0.5368 | 0.5746 | 1.0000 | 0.249 | 0.328 |
| Gemma 2 9B | LOGIC | 19 | 1.5 | +0.760 | [-0.010, +1.520] | 29/51/20 | 0.7865 | 0.8303 | 1.0000 | 0.193 | 0.321 |
| Gemma 3 4B | LOGIC | 17 | 1 | +0.750 | [+0.010, +1.490] | 31/52/17 | 0.7865 | 0.8303 | 1.0000 | 0.205 | 0.345 |
| Gemma 2 9B | MORAL | 38 | 0.5 | +0.700 | [+0.280, +1.150] | 45/39/16 | 0.4006 | 0.0590 | 0.0909 | 0.311 | 0.467 |
| Gemma 3 4B | LOGIC | 22 | 1.5 | +0.650 | [-0.060, +1.280] | 27/56/17 | 0.7865 | 0.8303 | 1.0000 | 0.190 | 0.318 |
| Gemma 2 9B | SENTIMENT | 26 | 0.3 | +0.640 | [+0.140, +1.200] | 33/44/23 | 0.5368 | 0.8303 | 1.0000 | 0.247 | 0.350 |
| Gemma 3 4B | POLITIC | 17 | 2 | +0.633 | [+0.010, +1.266] | 43/23/32 | 0.7865 | 0.8303 | 1.0000 | 0.193 | 0.230 |
| Gemma 2 9B | LOGIC | 19 | 0.3 | +0.630 | [-0.170, +1.480] | 25/54/21 | 0.7865 | 0.8307 | 1.0000 | 0.154 | 0.264 |
| Gemma 2 9B | POLITIC | 38 | 1.5 | +0.612 | [-0.082, +1.235] | 34/40/24 | 0.7865 | 0.8303 | 1.0000 | 0.180 | 0.231 |
| Gemma 2 2B | POLITIC | 16 | 0.5 | +0.610 | [-0.040, +1.260] | 32/41/27 | 0.7865 | 0.8303 | 1.0000 | 0.188 | 0.271 |

Raw effects are useful, but they should be interpreted with the p-values, confidence intervals, and quality metrics. For POLITIC and SENTIMENT, these are directional polarity shifts, not automatic improvements.

## Global Spearman Correlations

Spearman correlation uses ranks, so it is appropriate for nonlinear and ordinal-style data.

| Variable | Spearman rho | p(FDR) | N |
| --- | --- | --- | --- |
| quality_delta_mean | 0.463 | 4.29e-18 | 344 |
| primary_win_rate | 0.442 | 7.57e-17 | 344 |
| clean_rate | 0.239 | 1.82e-05 | 344 |
| judge_steered_win_rate | 0.179 | 0.0015 | 344 |
| relative_layer_depth | -0.005 | 0.9798 | 344 |
| alpha | -0.017 | 0.8244 | 344 |
| bad_output_rate_delta | -0.301 | 5.47e-08 | 344 |

Interpretation:

- A positive correlation with `quality_delta_mean` means stronger steering tends to preserve or improve quality.
- A positive correlation with `clean_rate` means raw steering tends to produce more usable wins.
- Weak alpha correlation supports the claim that alpha is not simply monotonic.

## Global Regression

This OLS model predicts config-level primary delta using model, domain, alpha, alpha squared, relative layer depth, and relative layer depth squared. Robust HC3 standard errors are used.

| Term | Coef | p | 95% CI | R^2 |
| --- | --- | --- | --- | --- |
| Intercept | 0.096 | 0.6056 | [-0.269, +0.461] | 0.110 |
| C(model)[T.Gemma 2 9B] | 0.137 | 0.0024 | [+0.049, +0.226] | 0.110 |
| C(model)[T.Gemma 3 4B] | 0.032 | 0.4965 | [-0.060, +0.123] | 0.110 |
| C(domain)[T.moral] | -0.162 | 0.0011 | [-0.260, -0.065] | 0.110 |
| C(domain)[T.politic] | -0.248 | 5.18e-06 | [-0.355, -0.142] | 0.110 |
| C(domain)[T.sentiment] | -0.155 | 0.0014 | [-0.250, -0.060] | 0.110 |
| alpha | -0.039 | 0.7237 | [-0.254, +0.176] | 0.110 |
| alpha_sq | 0.014 | 0.7827 | [-0.088, +0.117] | 0.110 |
| relative_layer_depth | 0.128 | 0.8031 | [-0.875, +1.130] | 0.110 |
| I(relative_layer_depth ** 2) | -0.085 | 0.8137 | [-0.788, +0.619] | 0.110 |

Interpretation:

- `alpha_sq` helps test non-monotonic alpha behavior.
- relative layer terms test whether early/middle/late intervention regions matter.
- categorical model/domain terms estimate broad differences after accounting for alpha and layer depth.

## Alpha Quadratic Regression

For each model-domain pair:

```text
primary_delta_mean ~ alpha + alpha^2
```

The table is sorted by FDR-corrected `alpha^2` p-value.

| Model | Domain | alpha coef | alpha^2 coef | alpha^2 p(FDR) | R^2 |
| --- | --- | --- | --- | --- | --- |
| Gemma 2 2B | SENTIMENT | -0.440 | 0.234 | 0.7132 | 0.114 |
| Gemma 2 9B | LOGIC | -0.559 | 0.254 | 0.7132 | 0.046 |
| Gemma 2 9B | POLITIC | 0.412 | -0.163 | 0.7132 | 0.062 |
| Gemma 3 4B | MORAL | -0.391 | 0.164 | 0.7132 | 0.056 |
| Gemma 3 4B | SENTIMENT | 0.229 | -0.144 | 0.7132 | 0.090 |
| Gemma 2 2B | LOGIC | 0.161 | -0.011 | 0.9685 | 0.059 |
| Gemma 2 2B | POLITIC | 0.007 | -0.029 | 0.9685 | 0.012 |
| Gemma 2 2B | MORAL | -0.101 | 0.036 | 0.9685 | 0.009 |
| Gemma 2 9B | SENTIMENT | -0.093 | 0.045 | 0.9685 | 0.004 |
| Gemma 2 9B | MORAL | 0.199 | -0.111 | 0.9685 | 0.019 |
| Gemma 3 4B | LOGIC | 0.104 | -0.097 | 0.9685 | 0.045 |
| Gemma 3 4B | POLITIC | -0.148 | 0.091 | 0.9685 | 0.010 |

This directly tests the claim:

> Larger alpha is not always better.

If `alpha^2` is important, the alpha response is curved/nonlinear.

## Relative-Layer Regression

For each model-domain pair:

```text
primary_delta_mean ~ relative_layer_depth + relative_layer_depth^2
```

The table is sorted by FDR-corrected depth-squared p-value.

| Model | Domain | depth coef | depth^2 coef | depth^2 p(FDR) | R^2 |
| --- | --- | --- | --- | --- | --- |
| Gemma 2 9B | MORAL | -7.470 | 5.329 | 0.0052 | 0.392 |
| Gemma 2 9B | SENTIMENT | 3.347 | -2.686 | 0.1029 | 0.349 |
| Gemma 3 4B | LOGIC | 2.622 | -2.194 | 0.1029 | 0.188 |
| Gemma 3 4B | MORAL | -2.088 | 1.711 | 0.1029 | 0.140 |
| Gemma 3 4B | SENTIMENT | -1.695 | 1.329 | 0.1029 | 0.125 |
| Gemma 3 4B | POLITIC | 3.635 | -2.710 | 0.1029 | 0.156 |
| Gemma 2 2B | POLITIC | 12.996 | -7.781 | 0.1860 | 0.177 |
| Gemma 2 2B | LOGIC | 7.549 | -3.793 | 0.4845 | 0.368 |
| Gemma 2 9B | LOGIC | -3.834 | 2.181 | 0.5345 | 0.087 |
| Gemma 2 9B | POLITIC | 1.382 | -0.844 | 0.8004 | 0.012 |
| Gemma 2 2B | MORAL | -2.199 | 1.069 | 0.8335 | 0.106 |
| Gemma 2 2B | SENTIMENT | -0.028 | 0.005 | 0.9838 | 0.000 |

This tests whether steering works best in early, middle, or later evaluated layers. The relative layer is computed within each model's evaluated layer range.

## Pareto Frontier Summary

A configuration is Pareto-efficient if no other config for the same model/domain is at least as good on all three metrics:

```text
primary_delta_mean
quality_delta_mean
clean_rate
```

and strictly better on at least one.

| Model | Domain | Pareto Configs | Best Pareto Example | Delta | Quality | Clean |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 2 2B | LOGIC | 7 | L19 a=0.5 | +0.590 | +0.128 | 12.0% |
| Gemma 2 2B | MORAL | 4 | L16 a=1.5 | +0.550 | +0.010 | 7.0% |
| Gemma 2 2B | POLITIC | 3 | L16 a=0.5 | +0.610 | +0.203 | 17.0% |
| Gemma 2 2B | SENTIMENT | 6 | L13 a=0.2 | +0.400 | +0.463 | 4.0% |
| Gemma 2 9B | LOGIC | 3 | L19 a=0.1 | +1.160 | +0.618 | 24.0% |
| Gemma 2 9B | MORAL | 4 | L38 a=0.5 | +0.700 | +0.040 | 12.0% |
| Gemma 2 9B | POLITIC | 5 | L38 a=1.5 | +0.612 | +0.252 | 25.5% |
| Gemma 2 9B | SENTIMENT | 4 | L26 a=0.3 | +0.640 | +0.110 | 12.0% |
| Gemma 3 4B | LOGIC | 5 | L17 a=1 | +0.750 | +0.065 | 26.0% |
| Gemma 3 4B | MORAL | 2 | L9 a=0.7 | +0.380 | +0.530 | 10.0% |
| Gemma 3 4B | POLITIC | 1 | L22 a=2 | +0.796 | +0.537 | 34.7% |
| Gemma 3 4B | SENTIMENT | 2 | L22 a=1 | +0.540 | +0.360 | 10.0% |

Pareto analysis is useful because the best raw delta is not always the best practical setting. For paper figures, Pareto-efficient points are strong candidates for annotation.

## Distributional Insight

The data are discrete, tied-heavy, and often non-normal. This is visible from the high tie rates and from the fact that many median deltas are zero even when mean deltas are positive.

Therefore:

- paired t-tests are useful but not sufficient;
- Wilcoxon and sign tests are important;
- bootstrap CIs are preferable to relying only on normal-theory intervals;
- clean success should be reported because raw score movement can occur in low-quality text.

## Main Statistical Conclusions

1. **The strict all-tests FDR standard is not passed by any single configuration.** This is an important negative statistical result: the judge scores are discrete and tie-heavy, and there are hundreds of comparisons. Do not claim per-config significance under this conservative criterion.
2. **The strongest effects are still meaningful candidates.** Several configurations have large positive mean deltas and bootstrap intervals above or near zero, especially Gemma 2 9B LOGIC, Gemma 2 9B MORAL, Gemma 3 4B LOGIC, and Gemma 3 4B POLITIC.
3. **LOGIC is the strongest improvement-oriented domain.** It has the clearest positive raw effects and the strongest model-level pattern, especially for Gemma 2 9B.
4. **POLITIC and SENTIMENT must be framed directionally.** Positive deltas mean rightward or positive-polarity shifts, not universal quality gains.
5. **Alpha is descriptively non-monotonic, but the quadratic alpha terms are not globally strong after FDR.** The best alpha changes by model/domain/layer, so the paper should avoid saying larger alpha is always better.
6. **Layer matters, but not universally.** The strongest depth-regression evidence appears for Gemma 2 9B MORAL; other domains show model-specific layer preferences rather than one universal best depth.
7. **Quality and clean success are necessary.** The Pareto and correlation analyses show why raw delta alone is not enough for choosing best settings.

## Generated Files

CSV outputs:

- `config_level_stat_tests.csv`
- `pareto_frontier_configs.csv`
- `spearman_correlations.csv`
- `global_regression.csv`
- `alpha_quadratic_regression.csv`
- `relative_layer_regression.csv`

Figures:

- `global_spearman_correlations.png`
- `primary_vs_quality_scatter.png`
- `alpha_response_by_domain.png`
- `config_mean_delta_distribution.png`
