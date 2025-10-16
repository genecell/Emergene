## EMERGENE

[![Python Version](https://img.shields.io/pypi/pyversions/emergene.svg)](https://pypi.org/project/emergene/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://genecell.github.io/Emergene/)


### Individual cell-based differential transcriptomic analysis across conditions

If your scRNA-seq data have two conditions, e.g., disease and control, and you'd like to know which cells changed the most in the disease condition, then Emergene is the tool for you. Similarly, if you are studying development, and want to understand the emergence of cell type diversity, Emergene can help you with that. Not only scRNA-seq data, but also the spatial transcriptomics data could be used as the input for Emergene.

### 📖 Documentation

[EMERGENE Documentation](https://genecell.github.io/Emergene/) 


### 📦 Installation

You could simply install EMERGENE via `pip` in your conda environment:
```bash
pip install emergene
```
For the development version in GitHub, you could install via:
```bash
pip install git+https://github.com/genecell/Emergene.git
```


### 🌟 Key features

- **Cell-level analysis**: Identify which individual cells change the most between conditions
- **Gene pattern detection**: Discover genes with coordinated local expression patterns and condition-specific expression patterns
- **Multi-condition support**: Compare multiple conditions simultaneously
- **Spatial compatibility**: Works with both scRNA-seq and spatial transcriptomics data
- **Local fold changes**: Quantify cell-specific expression changes relative to other conditions
- **Statistical rigor**: Built-in significance testing and background correction for each individual cell


## 💡 When to Use EMERGENE

#### Perfect for:
- 🔬 **Disease vs. Control**: Identify cells and genes most affected by disease
- 🧬 **Developmental Biology**: Track emergence of cell type diversity over time
- 🗺️ **Spatial Transcriptomics**: Discover spatially coordinated expression patterns
- 🧪 **Treatment Response**: Find cells that respond most to interventions
- 🔄 **Cell State Transitions**: Detect genes driving state changes

#### Example Questions:
- *Which cells changed the most in disease compared to healthy controls?*
- *What genes show emergent coordinated patterns in specific developmental stages?*
- *Which spatial regions show the strongest transcriptional shifts?*
- *What cell types are most responsive to treatment?*


### Citation
If EMERGENE is useful for your research,  please consider citing [Wu et al., Pyramidal neurons proportionately alter the identity and survival of specific cortical interneuron subtypes, bioRxiv (2024)](https://www.biorxiv.org/content/10.1101/2024.07.20.604399v1). 

### Contact
Min Dai
dai@broadinstitute.org
