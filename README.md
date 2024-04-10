# Topic Modelling & Sentiment Analysis on Tweets about Russian Ukraine War
This is the project for IS450 Text Mining and Natural Language Processing Group 2 where we apply text mining and NLP techniques (topic modelling, sentiment analysis) on a set of Twitter data focused on Russian Ukraine War.

## Topic Modeling Analysis

### Overview
This Jupyter notebook contains a comparative analysis of three different topic modeling techniques—Latent Semantic Analysis (LSA), Probabilistic Latent Semantic Analysis (pLSA), and Latent Dirichlet Allocation (LDA)—applied to a corpus of documents related to the Russia-Ukraine War. The goal is to identify and amplify genuine public discourse around the conflict, particularly emphasizing the human impact.

### Models Overview
- **LSA**: Utilizes mathematical techniques to deduce and articulate the semantics of words through statistical analysis of a vast text corpus.
- **pLSA**: Introduces a probabilistic framework to LSA, enabling the association of latent topics with documents in a statistical model.
- **LDA**: A more sophisticated approach that views documents as composites of topics, which in turn comprise mixtures of words.

### Notebook Contents
1. Data Preprocessing
2. Topic Modeling with LSA
3. Topic Modeling with pLSA
4. Topic Modeling with LDA
5. Model Evaluation and Comparison

### Prerequisites
- **Required Libraries**: `pandas`, `re`, `sys`, `csv`, `ast`, `wordcloud`, `gensim`, `nltk`, `numpy`, `matplotlib`, `PLSA`

## Lexicon-Based Sentiment Analysis

### Overview
This section presents an analysis of sentiments using three distinct lexicons—NLTK Vader, TextBlob, and SentiWordNet—applied to a labeled dataset titled 'uk_ru_2023_en_text_random_labeled_dataset_Labelled.csv'. This dataset comprises English text data that has been pre-labeled.

### Prerequisites
- **Required Libraries**: `pandas`, `nltk`, `sklearn`, `wordcloud`, `matplotlib`, `textblob`
