# Attract-Repel
Nikola Mrkšić, University of Cambridge (nikola.mrksic@gmail.com)

This repository contains the code and data for the Attract-Repel method presented in [Semantic Specialisation of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints](https://arxiv.org/abs/1706.00374) (Mrkšić et al., TACL 2017).


### Available Word Vector Spaces

The bilingual word vector spaces for English + 51 languages ([link](https://drive.google.com/open?id=0B_pyA_IW4g-jQzhCekVZTFFmWmc) to list of languages with respective language codes) are available [here](https://drive.google.com/open?id=0B_pyA_IW4g-jZHlWWVBfaWRYY0E). The four-lingual EN-DE-IT-RU vector space which achieves state-of-the-art performance on Multilingual SimLex-999 can be downloaded [here](https://drive.google.com/open?id=0B_pyA_IW4g-jZzBIZXpYS1RseFk). 

The five baseline bilingual vector spaces used in the paper (Tables 7 and 8) are available [here](https://drive.google.com/open?id=0B_pyA_IW4g-jQ2lTTnVnOFBWU2s).

Hebrew and Croatian SimLex-999 datasets are available [here](https://drive.google.com/open?id=0B_pyA_IW4g-jTlJzOHlSWVZWbTQ). The two datasets are also included in this repository. 

The Italian and German Wizard-of-Oz (WOZ) dialogue state tracking datasets are available [here](https://drive.google.com/open?id=0B_pyA_IW4g-jd3BRM2JlVHF5UVE). 

The large morphologically specialised vectors (SGNS-LARGE) for English, Italian and German, presented in [Morph-fitting: Fine-Tuning Word Vector Spaces with Simple Language-Specific Rules](https://arxiv.org/abs/1706.00377), (Vulić et al., ACL 2017) are available [here](https://drive.google.com/open?id=0B_pyA_IW4g-jSW5ITXFqNFJ6LTQ). 
 

### Configuring the Tool

The Attract-Repel tool reads all the experiment config parameters from the ```experiment_parameters.cfg``` file in the root directory. An alternative config file can be provided as the first (and only) argument to ```attract-repel.py```. 

The config file specifies:
* the location of the initial word vectors (```distributional_vectors```);
* the sets of linguistic constraints to be injected into the vector space (```antonyms``` and ```synonyms```);
* whether to print SimLex scores after each epoch (```print_simlex```) and whether to log SimLex/WS-353 scores to file (```log_scores_over_time```).

The config file also specifies the hyperparameters of the attract-repel procedure (set to their default values in ```config/experiment_parameters.cfg```). 

The evaluation directory contains the SimLex-999 dataset (Hill et al., 2014), its multilingual variant (Leviant and Reichard, 2015), the SimVerb dataset (Gerz et al., 2016), and mono- and multilingual WS-353 datasets (Finkelstein et al., 2002; Leviant and Reichart, 2015). It also containts the Hebrew and Croatian SimLex-999 datasets collected in our work. 


### Running Experiments

```python code/attract-repel.py config/experiment_parameters.cfg```

Running the experiment loads the word vectors specified in the config file and fits them to the provided linguistic constraints. The procedure prints the updated word vectors to the results directory as ```results/final_vectors.txt``` (one word vector per line), alternative write path can be specified in the config file (```output_filepath```).  


### References

The TACL paper which introduces the Attract-Repel procedure, the cross-lingual vector spaces, Hebrew and Croatian SimLex-999 datasets and Italian and German Dialogue State Tracking corpora:
```
 @Article{Mrksic:2017,
  author    = {Nikola Mrk\v{s}i\'c and Ivan Vuli\'{c} and Diarmuid {\'O S\'eaghdha} and Ira Leviant and Roi Reichart and Milica Ga\v{s}i\'c and Anna Korhonen and Steve Young},
  title     = {Semantic Specialisation of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints},
  journal   = {Transactions of the Association for Computational Linguistics},
  volume    = {5}
  year      = {2017},
  pages     = {309--324},  
 }
```

The ACL paper which uses the Attract-Repel procedure and simple language-specific rules to induce high-quality vector spaces which model morphological phenomena: 
```
@inproceedings{Vulic:2017,
  author    = {Vuli\'{c}, Ivan and Mrk\v{s}i\'{c}, Nikola and Reichart, Roi and {\'O S\'eaghdha}, Diarmuid and Young, Steve and Korhonen, Anna},
  title     = {Morph-fitting: {F}ine-Tuning Word Vector Spaces with Simple Language-Specific Rules},
  booktitle = {Proceedings of ACL},
  year      = {2017},
  pages     = {56--68},
  }
``` 

If you are using PPDB 2.0 (Pavlick et al., 2015) or WordNet (Miller, 1995) constraints, please cite these papers. If you are using BabelNet constraints, please cite (Navigli and Ponzetto, 2012).
