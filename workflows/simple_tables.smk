"""Simple plots and tables for illustratory purposes."""
import matplotlib
matplotlib.use('agg')

import seaborn as sns
import numpy as np
import pandas as pd


workdir: "generated/simple_tables/"

rule all:
    input: "identifiability_Bernoulli_mixture.txt"


rule identifiability_bernoulli_mixture:
    output:
        text_file = "identifiability_Bernoulli_mixture.txt",
        latex_file = "identifiability_Bernoulli_mixture.tex"
    run:
        b = np.arange(2, 10)
        k = 2 * np.ceil(np.log2(b)) + 1
        k = k.astype(int)
        
        df = pd.DataFrame({
            "Mixture components": b,
            "Required features": k,
        })

        df.to_latex(output.latex_file, index=False, index_names=False)
        df.to_csv(output.text_file, sep="\t", index=False)

