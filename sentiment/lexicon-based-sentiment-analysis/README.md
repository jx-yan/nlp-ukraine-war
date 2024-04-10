# Lexicon Based Sentiment Analysis

This repository contains code for sentiment analysis on a dataset using three different lexicons: NLTK Vader, TextBlob, and SentiWordNet. The dataset used is 'uk_ru_2023_en_text_random_labeled_dataset_Labelled.csv', which contains labelled English text data.

## Requirements

Make sure you have the following libraries installed:

- pandas
- nltk
- sklearn
- wordcloud
- matplotlib
- textblob

You can install these libraries using pip:
```
pip install pandas nltk scikit-learn wordcloud matplotlib textblob
```

Additionally, you need to download NLTK resources:
```python
import nltk
nltk.download('vader_lexicon')nltk.download('punkt')
nltk.download('stopwords')nltk.download('wordnet')
nltk.download('sentiwordnet')```