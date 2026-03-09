# Diverse Beam Search (DBS) Benchmark

This document details the evaluation results of the Diverse Beam Search (DBS) [implementation](https://github.com/yipkingster/t5x/blob/main/t5x/dbs_metrics/TECH_BLOG.md) in T5X compared against standard Beam Search. 

The evaluation is conducted on the **WMT14 English-to-German (En-De)** dataset (`wmt14_en_de_v003`) using the `t5_base` model (checkpoint step: 999900) [^1].

## Methodology

We evaluate two key decoding metrics across both standard Beam Search and Diverse Beam Search:

1. **BLEU Score**: Measures the syntactic overlap of the best generated sequence against the ground-truth human references. Evaluates translation *quality*.
2. **Self-BLEU Score**: Measures the syntactic overlap between *all* generated sequences (beams) for the exact same input sentence against one another. Evaluates translation *diversity* (Lower Self-BLEU = Higher Diversity).

### Hyperparameters

| Parameter | Standard Beam Search | Diverse Beam Search (DBS) |
| :--- | :--- | :--- |
| `num_decodes` (total beams) | 4 | 4 |
| `num_beam_grps` | 1 | 2 |
| `diversity_strength` | N/A | 0.5 |

## Results

| Metric | Standard Beam Search | Diverse Beam Search | Absolute Difference | Percentage Difference |
| :--- | :--- | :--- | :--- | :--- |
| **BLEU** | 27.49 | 27.47 | -0.02 | **-0.07%** |
| **Self-BLEU** | 86.04 | 75.18 | **-10.86** | **-12.62%** |

## Conclusion

1. **Preserved Quality**: The Diverse Beam Search implementation produces an almost identical BLEU score compared to standard Beam Search (27.47 vs 27.49). This negligible absolute drop of `0.02` (a **0.07%** decrease) demonstrates that enforcing diversity doesn't degrade the accuracy of the primary predicted translations.
2. **Dramatically Improved Diversity**: The Self-BLEU score dropped by nearly `11.0` absolute points (86.04 down to 75.18), representing a significant **12.62%** decrease in intra-beam syntactic overlap. Standard beam search suffers from extreme intra-beam repetition (generating effectively permutations of the same sentence). By applying `diversity_strength=0.5` across `num_beam_grps=2`, DBS successfully forces the model to explore and generate meaningfully distinct alternative phrasing and vocabulary in the secondary group.

The DBS implementation provides exactly the intended behavior: robust alternative sequences without sacrificing top-1 quality!

[^1]: We specifically use the original T5 v1.0 base model over T5 v1.1. T5 v1.0 was pre-trained on a multi-task mixture that explicitly included English-to-German supervised translation pairs. Conversely, T5 v1.1 was pre-trained exclusively on an unsupervised masking objective (C4 span corruption) and lacks any zero-shot translation capabilities out-of-the-box. Evaluating T5 v1.1 without fine-tuning would result in invalid, hallucinated outputs for both BLEU and Self-BLEU metrics.
