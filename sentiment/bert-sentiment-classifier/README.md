# Sentiment Analysis of Russian Ukraine War Twitter Dataset using BERT

## Project Description
- This notebook performs sentiment analysis using a fine-tuned BERT model on a Twitter dataset with Russian Ukraine War.
- Key steps include:
    1. Data loading and preprocessing
    2. Text encoding with BERT tokenizer
    3. Fine-tuning a BERT model for sentiment classification
    4. Model training and evaluation
    5. Applying the trained model to predict sentiment
    6. Calculating sentiment distribution and per-class accuracy

## Dependencies

- torch (PyTorch deep learning framework)
- pandas (Data manipulation and analysis)
- transformers (Hugging Face Transformers library)
- seaborn (Visualization)
- matplotlib (Visualization)
- sklearn (Scikit-learn for machine learning tools)
- numpy (Numerical computation)
- tqdm (Progress bars)

## Installation
> It is recommeneded to run this notebook on a GPU due to the model's high compute requirements
- Step 1: Install CUDA Toolkit, cuDNN Library, Anaconda
- Step 2: Create a Conda Environment and install dependencies:

```bash
conda create --name gpu_env python=<supported-python-version>
conda install -c anaconda tensorflow-gpu keras-gpu
python -m ipykernel install --user --name gpu_env --display-name "Python (GPU)"
```

- Step 3: Install Python dependencies in conda environment:

```bash
pip install torch pandas transformers seaborn matplotlib sklearn numpy tqdm
```
- Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

- Step 5: Verify that Jupyter Notebook is using GPU
```python
import torch
torch.cuda.is_available()
# output: True
```


## Data Preparation

- Obtain the dataset: 
    * Download 'uk_ru_2023_en_text_random_labeled_dataset_Labelled.csv'.
    * Place it in the same directory as the notebook.
- Data Format: Ensure the CSV has at least the following columns:
    * processed_text: The preprocessed text samples.
    * sentiment: The true sentiment labels.

## Usage

- Clone or download the repository
- Install dependencies (See 'Installation' section)
- Run the notebook ('bert-sentiment-classifier.ipynb')

### Example Code Snippet

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load fine-tuned model and tokenizer (assuming they're saved)
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

# Example text 
text = "This is a positive sentiment example." 

# Encode text
encoded_input = tokenizer(text, return_tensors='pt') 

# Get sentiment prediction
output = model(**encoded_input)
predicted_label = output.logits.argmax(-1).item()
```

## Results

The notebook outputs:
- Per-class accuracy
- Training and validation loss curves (if plotted in the notebook)
- A saved fine-tuned model in the './saved_model' directory

## Customization
- Dataset: Modify the code to load your specific dataset.
- Preprocessing: Adjust preprocessing steps if needed.
- Hyperparameters: Experiment with training parameters (batch size, learning rate, epochs) in the training loop.