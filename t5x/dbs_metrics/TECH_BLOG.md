# Implementing Diverse Beam Search in T5X: Lessons in LLM Decoding Logic

Standard sequence generation models traditionally rely on **Beam Search (BS)** to find the most likely translation or text sequence. However, standard BS suffers from a critical flaw: a lack of diversity. Because the algorithm explores paths based purely on descending probability, the final candidates often end up being permutations of the exact same sentence, differing by only a single word or punctuation mark. 

So we implemented **Diverse Beam Search (DBS)**. DBS solves this by partitioning the beams into independent groups, enforcing diversity penalties sequentially so that subsequent groups are mathematically forced to explore different vocabulary compared to prior groups.

This post reviews the math, design decisions, and implementation challenges of integrating DBS natively into the Google T5X sequence decoding framework.


---

## The Mathematics of DBS

The implementation is based on the paper [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models (Vijayakumar et al., 2016)](https://arxiv.org/abs/1610.02424).

### Standard Beam Search Objective
Since seeking the best probability of a sequence over the entire space of possible sequences is computationally impossible, standard BS acts as a greedy heuristic optimization across $B$ parallel beams (the `num_decodes`). At each time step $t$, it attempts to find the set of next tokens across all beams $Y_{[t]}$ that locally maximizes the probabilities when extending the previous sequence step states:

$$ \mathbf{Y}_{[t]} = \argmax_{y_1^{[t]}, \ldots, y_B^{[t]} \in \mathcal{V} \text{ s.t. } y_i^{[t]} \neq y_j^{[t]}} \sum_{b=1}^B \log P(y_b^{[t]} \mid Y_{b, [t-1]}, X) $$

Here is the piece by piece explanation:
1. $B$ is the number of active beams.
2. The superscript $[t]$ placed in brackets (as in $y^{[t]}$ or $\mathbf{Y}_{[t]}$) denotes the **time step** index of the sequence generation process. For example, $y_b^{[t]}$ represents the single token being evaluated exactly at time step $t$ for beam $b$.
3. The "s.t." in the term $\text{s.t. } y_i^{[t]} \neq y_j^{[t]}$ means "subject to". It simply forces the algorithm to select $B$ distinct candidate tokens in each beam, avoiding duplication at a single timestep. 
4. The summation run $\sum_{b=1}^B$ means the algorithm searches for a **set** of next-tokens that maximizes the probabilities across all $B$ concurrently active beam-branches strictly localized at step $t$.
5. $\log P$ ("Log-Probability"): The natural logarithm of the probability score assigned by the neural network.  We use the sum of log-probabilities instead of product of raw probabilities to avoid numerical underflow by multiplying many small numbers together.
6. ($y_b^{[t]} \mid Y_{b, [t-1]}, X$): Given the prompt/input $X$ (e.g. the translating sentence) and the previous context $Y_{b, [t-1]}$ leading up to this point within beam $b$, we find the specific next token $y_b^{[t]}$ that maximizes that beam's conditional probability.

### Diverse Beam Search Objective
DBS partitions the total beams ($B$) into groups ($G$). It optimizes these groups sequentially at each time step. The first group ($g=1$) acts like standard BS. For any subsequent group $g$, a penalty $\Delta$ is applied to discourage the selection of tokens that were already chosen by the previous groups $\{1, \ldots, g-1\}$ at that identical timestep $t$.

$$ \mathbf{Y}_{[t]}^{[g]} = \argmax_{y_1^{[t]}, \ldots, y_{B'}^{[t]} \in \mathcal{V}} \sum_{b=1}^{B'} \left( \log P(y_b^{[t]} \mid Y_{b,[t-1]}^{[g]}, X) + \lambda \sum_{h=1}^{g-1} \Delta(y_b^{[t]}, y_{b}^{[t], [h]}) \right) $$

Here is the piece by piece explanation:
1. $\mathbf{Y}_{[t]}^{[g]}$ ("Y at step t for group g"): The set of $B'$ tokens chosen for group $g$ at time step $t$.
2. $\argmax_{y_1^{[t]}, \ldots, y_{B'}^{[t]} \in \mathcal{V}}$: The $\argmax$ function. It looks through the entire vocabulary ($\mathcal{V}$) to find the next-tokens that maximize the total score of the summation next to it.
3. $\sum_{b=1}^{B'}$: We sum the scores across all $B'$ parallel beam groups. (Where $B'$ is simply the total beams $B$ divided by the number of groups $G$).
4. $\log P(y_b^{[t]} \mid Y_{b,[t-1]}^{[g]}, X)$: The log-probability that the model's neural network assigns to the candidate token $y_b^{[t]}$, given the original prompt $X$ and the sequence selected before this timestep for this specific beam group $g$: $Y_{b,[t-1]}^{[g]}$.
5. $\lambda$: The `diversity_strength` penalty multiplier. It controls how severely we want to punish the model for duplicate tokens.
6. $\sum_{h=1}^{g-1}$: A loop that iterates through every single previous group ($h$) that has already been decided at this particular step $t$ (from group $1$ up to $g-1$).
7. $\lambda \sum \Delta(y_b^{[t]}, y_{b}^{[t], [h]})$: The **Hamming Diversity Penalty**. By multiplying the number of occurrences of this token by $\lambda$, we subtract a penalty proportional to how many times the exact same token has already been selected by older groups at this time step.

---

## Design Ideas and Architecture

When modifying a complex, highly optimized library like T5X, our primary goal was to be as minimum invasive to the existing code structure as possible.

After all, standard Beam Search is merely a special case of Diverse Beam Search where the number of groups (`num_beam_grps`) is exactly $1$. 

Instead of duplicating the decoding loops, we injected our diversity logic purely at the token selection phase. Specifically, we introduced a new function `diverse_beam_search_top_k()` that seamlessly replaces the old `top_k_two_stage` call.

---

## Implementation Challenges

Bringing the math into JAX array operations presented several tricky challenges:

### 1. Inability to Use Flattened Log-probs Initially
Standard beam search flattens the `[num_decodes, vocab_size]` logits array into a massive `[num_decodes * vocab_size]` array to run a single, global `top_k` operation. We couldn't do this because DBS requires partitioning the `num_decodes` dimension by `num_beam_grps` first to compute intra-group top-K tokens and apply penalties sequentially.

### 2. The `group_offsets` Nuance
A subtle but critical bug we overcame involved global indexing. When selecting the top indices across flattened beam dimensions, we must shift those positional indices by an offset specifically designed for each group (`group_offsets`). Without calculating and adding `group_offsets`, surviving beams selected in Group 2 would incorrectly index into the memory states of Group 1, causing silent corruption for the rest of the generation.

---

## Our Testing Strategy

Validating beam search involves mocking token transition probabilities (edge potentials). The existing T5X unit tests used a Markov chain with mock states `A` and `B`. 

We built the unit test on top of this existing design by introducing a new state `C`. State `C` was designed as a "diverse" alternative path to `A` and `B` with the same costs at 1 timestep. This allows us to reuse the existing infrastructure to test a new state transition path and a new search strategy. By introducing only at 1 timestep it also reduces the complexity and make the code easier to debug.

---

## Conclusion & Results

See our formal performance results in [BENCHMARK.md](BENCHMARK.md) where we achieved a significant **12.62%** improvement in sample diversity on the WMT14 English-to-German (En-De) dataset (wmt14_en_de_v003).
