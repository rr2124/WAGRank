# WAGRank

This is an implement of WAGRank 1 model and WAGRank 2 model proposed in the paper *WAGRank: A word ranking model based on word attention graph for keyphrase extraction*.

## Installation

1. Download "GoogleNews-vectors-negative300.bin.gz." from below link:

    https://code.google.com/archive/p/word2vec/

2. Download "stanford-corenlp-full-2018-02-27" from below link:

    https://stanfordnlp.github.io/CoreNLP/history.html

3. Launch Stanford Core NLP tagger:
   * Open a new terminal
   * Go to the stanford-core-nlp-full directory
   * Run the server `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 & `

4. Download "swisscom_ai" from below link:

    https://github.com/swisscom/ai-research-keyphrase-extraction

## Results Replication

The main results of the paper can be replicated as below:

1. Datasets
   
   Two datasets (SemEval2017 and  Inspec) used in the paper are contained in the "Data.file".

2. Hyperparameter
   
   Set hyperparameter "topn" in models to 20 on SemEval2017 dataset and 30 on Inspec dataset.

3. Run WAGRank1.py and WAGRank2.py, respectively.

## Example
This is an example of WAGRank for keyphrase extraction.

```python
import WAGRank1
import WAGRank2
# Input Text
text = 'The study outlines a trial of transient response analysis on full-scale motorway bridge structures to obtain information concerning the steel–concrete interface and is part of a larger study to assess the long-term sustained benefits offered by Impressed Current Cathodic Protection (ICCP) after the interruption of the protective current [1]. These structures had previously been protected for 5–16years by an ICCP system prior to the start of the study. The protective current was interrupted, in order to assess the long-term benefits provided by ICCP after it has been turned off. This paper develops and examines a simplified approach for the on-site use of transient response analysis and discusses the potential advantages of the technique as a tool for the assessment of the corrosion condition of steel in reinforced concrete structures.'

# Keyphrase Extraction
WAGRank1.extract_key_phrases(text, topn=20)
WAGRank2.extract_key_phrases(text, topn=20)

# Output Keyphrases
# Keyphrases extracted by WAGRank1:
['iccp system', 'iccp', 'long-term sustained benefits', 'impressed current cathodic protection', 'full-scale motorway bridge structures', 'long-term benefits', 'transient response analysis', 'on-site use', 'protective current', 'concrete structures']
# Keyphrases extracted by WAGRank2:
['iccp system', 'long-term sustained benefits', 'iccp', 'long-term benefits', 'impressed current cathodic protection', 'full-scale motorway bridge structures', 'transient response analysis', 'on-site use', 'protective current', 'corrosion condition']
```