import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests

def main():

    cindex = pd.read_csv('Cindices_per_fold.csv')

    models = cindex["Model"].unique()

    pairs = list(itertools.combinations(models, 2))

    pair_names = []
    raw_pvals = []

    for m1, m2 in pairs:

        v1 = cindex.loc[cindex["Model"] == m1].sort_values("Fold")["C-index"].to_numpy()
        v2 = cindex.loc[cindex["Model"] == m2].sort_values("Fold")["C-index"].to_numpy()

        stat, p = wilcoxon(v1, v2, alternative="two-sided")

        pair_names.append(f"{m1} vs {m2}")
        raw_pvals.append(p)

    raw_pvals = np.array(raw_pvals)

    _, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

    results_df = pd.DataFrame({
        "comparison": pair_names,
        "p_value": raw_pvals,
        "p_value_corrected": pvals_corrected
    })

    results_df.to_csv("stats_comparisons.csv")

if __name__ == "__main__":
    main()