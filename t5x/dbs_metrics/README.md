# Diverse Beam Search (DBS) Mathematical Formulation

This document outlines the mathematical foundation of our Diverse Beam Search (DBS) implementation, grounded in the original paper: [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models (Vijayakumar et al., 2016)](https://arxiv.org/abs/1610.02424).

---

## Standard Beam Search Objective

In standard auto-regressive sequence generation, given an input $X$, the goal is to find the sequence of tokens $Y = (y_1, y_2, \ldots, y_T)$ that maximizes the joint log-probability:

$$ Y^* = \arg\max_{Y \in \mathcal{Y}} \sum_{t=1}^T \log P(y_t \mid y_1, \ldots, y_{t-1}, X) $$

Because standard beam search explores paths purely based on descending probability, the resulting highest-scoring candidate sequences often suffer from a lack of diversity (e.g., differing by only a single suffix word or punctuation mark).

---

## Diverse Beam Search Objective

Diverse Beam Search addresses this lack of diversity by splitting the total number of beams ($B$) evenly into independent subsets called groups ($G$). 

If $B = 4$ and $G = 2$, we have two groups, each containing 2 beams.

DBS optimizes these groups **sequentially**. For the first group ($g=1$), the search behaves exactly like standard beam search. However, for any subsequent group $g \in \{2, \ldots, G\}$, a penalty $\Delta$ is applied at each step $t$ to discourage the selection of tokens that were already chosen by the previous groups $\{1, \ldots, g-1\}$.

The modified objective function for the $g$-th group is:

$$ Y^{[g]} = \arg\max_{Y \in \mathcal{Y}} \sum_{t=1}^T \left( \log P(y_t \mid y_1, \ldots, y_{t-1}, X) + \lambda \sum_{h=1}^{g-1} \Delta(y_t, y_t^{[h]}) \right) $$

Where:
* $Y^{[g]}$ is the final output sequence decoded for the $g$-th group.
* $y_t^{[h]}$ is the token generated at time step $t$ by previous group $h$.
* $\lambda \ge 0$ is the `diversity_strength` parameter, controlling the magnitude of the penalty.
* $\Delta$ is the diversity penalty function.

---

## The Hamming Diversity Penalty

While the original paper proposes several variations for the penalty function $\Delta$ (such as n-gram, cumulative, and bag-of-words), the most universally adopted implementation for sequential generation is the **Hamming Diversity Penalty**.

The Hamming penalty penalizes a candidate token proportional to how many previous groups selected that exact same token at the current time step $t$. It is defined mathematically using the indicator function $\mathbb{I}$:

$$ \Delta(y_t, y_t^{[h]}) = - \mathbb{I}(y_t == y_t^{[h]}) $$

When aggregated across all $g-1$ previously evaluated groups, the total penalty applied to the log-probability of candidate token $y_t$ becomes:

$$ \text{Penalty}(y_t) = -\lambda \sum_{h=1}^{g-1} \mathbb{I}(y_t == y_t^{[h]}) $$

### Mechanics in Practice
If $\lambda = 0.5$, and two preceding groups have both selected the word `"apple"` ($y_t = \text{"apple"}$) at step $t=3$:
1. The indicator function triggers twice (once for each prior group).
2. The logit score for `"apple"` in the current group $g$ is penalized by: $-(0.5) \times 2 = -1.0$.
3. This massive subtractive penalty forces the current beam search group to explore highly probable alternative tokens (e.g., `"orange"`, `"banana"`) that the preceding groups ignored.

### T5X Configuration Variables
In our T5X architecture implementation, these mathematical variables directly map to the Gin configuration:
* Total Beams ($B$) $\rightarrow$ `num_decodes`
* Total Groups ($G$) $\rightarrow$ `num_beam_grps`
* Penalty Strength ($\lambda$) $\rightarrow$ `diversity_strength`
