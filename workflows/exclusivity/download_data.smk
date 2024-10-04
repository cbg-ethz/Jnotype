
workdir: "generated/exclusivity/data"

rule all:
    input: ["gbm-muex-2014.csv", "gbm-muex-genesets.csv"]

rule download_mutations:
  output: "gbm-muex-2014.rda"
  shell: "wget https://github.com/cbg-ethz/TiMEx/raw/refs/heads/master/data/gbmMuex.rda -O {output}"


# Use 
# import pandas as pd
# pd.read_csv("generated/exclusivity/data/gbm-muex-2014.csv", index_col=0)
# to read the data frame
rule convert_rda_to_csv:
  input: "gbm-muex-2014.rda"
  output: "gbm-muex-2014.csv"
  shell: "Rscript -e \"load('{input}'); write.csv(gbmMuex, file='{output}', row.names=TRUE)\""


rule download_genesets:
  output: "gbm-muex-genesets-raw.txt"
  shell: "wget https://doi.org/10.1371/journal.pcbi.1003503.s009 -O {output}"


rule convert_genesets_to_csv:
  input: "gbm-muex-genesets-raw.txt"
  output: "gbm-muex-genesets.csv"
  run:
    import pandas as pd
    df = pd.read_csv(str(input), sep="\t")
    df.to_csv(str(output), index=None)

