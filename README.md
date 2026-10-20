# MMBO: Modularity Optimisation via (Signless) Total Variation

Code for **Li, Z., van Gennip, Y. & John, V. "An MBO method for modularity optimisation
based on total variation and signless total variation."** European Journal of Applied
Mathematics, 37(3), 743–825, 2026.

MMBO is a Merriman–Bence–Osher-type method for graph clustering (community detection) by
modularity optimisation. Leading eigenpairs are approximated with a Nyström extension and
QR decomposition (`Nystrom_extension_QR.py`), so the method scales to graphs with 70,000
nodes (MNIST).

## Experiments

| Dataset | Newman–Girvan null model | Erdős–Rényi null model |
|---|---|---|
| Zachary's karate club | `Example1_ZKC_NG.py` | `EX1_ZKC_ER_test.py` |
| MNIST (70k nodes, 10-NN graph) | `Example2_MNIST_NG.py` | `EX2_MNIST_ER_test.py` |
| Stochastic block model | `Example3_SBM_NG.py` | `EX3_SBM_ER_test.py` |

`MMBO_HU_Boyd.py` contains MMBO and the two earlier MBO schemes used for comparison
(Hu et al., Boyd et al.); `MNIST_ER_lou_CNM_sc.py` runs the Louvain, CNM and spectral
clustering baselines. Results are evaluated by modularity, ARI, purity, NMI and runtime.

Install the dependencies listed in `Required_packages.txt`. MNIST is downloaded by
[GraphLearning](https://github.com/jwcalder/GraphLearning).

## Authors

Implementation and experiments: Zijun Li. The research idea is due to Yves van Gennip;
the theory was developed jointly by Zijun Li and Yves van Gennip; Volker John (FU Berlin)
co-supervised the work.
