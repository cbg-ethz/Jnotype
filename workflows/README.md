# Workflows

This directory presents workflows associated with different publications.

Note that as the package evolved, some workflows may have been deprecated.
To rerun the analysis, it's then important to rewind the changes to the right point in the `git` history.
We mark these points with tags: if the tag is not provided, the workflows for a given project are compatible with the current version of the package.

Rewinding the changes to a particular tag can be accomplished by running:

```bash
$ git checkout tags/TAG_NAME
```

where `TAG_NAME` represents the right point in time for a particular project.

Note that 


## Projects

### Bayesian modeling of mutual exclusivity
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2024.10.29.620937-b31b1b.svg)](https://doi.org/10.1101/2024.10.29.620937)

**Directory:** `exclusivity`

**Tag:** None

Bayesian workflow employed to model mutual exclusivity patterns. This project is accompanied by the manuscript [*Bayesian modeling of mutual exclusivity patterns*](https://doi.org/10.1101/2024.10.29.620937).


### Labeled Bayesian Pyramids

**Directory:** `pyramids`

**Tag:** None

In this project we looked at labeled Bayesian pyramids, which introduce fixed effects to the framework of [Bayesian pyramids](https://academic.oup.com/jrsssb/article/85/2/399/7074362) of Yuqi Gu and David Dunson.
As such, the labeled Bayesian pyramids are a subclass of dependent mixture models and can be used for exploratory data analysis of cancer genotypes. 
The manuscript will be released in Summer 2025.

### Conditional Bernoulli mixture model

**Directory:** `cbmm`

**Tag:** None

In this project we propose the conditional Bernoulli mixture model and use them to model cancer genomes.
The manuscript will be released in Autumn 2025.

### Nonparametric extensions of the parametric models 

**Directory:** `nonparametric`

**Tag:** None

In this project we discuss how to extend a given parametric model by forming a mixture of Dirichlet processes around it.
The manuscript will be released in Autumn 2025.
