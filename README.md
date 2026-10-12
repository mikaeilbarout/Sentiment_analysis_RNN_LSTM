# IMDB Sentiment Analysis with RNN, LSTM and GRU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaeilbarout/Sentiment_analysis_RNN_LSTM/blob/main/sentiment_analysis.ipynb)

Classifying IMDB movie reviews as **positive** or **negative** with recurrent neural networks in TensorFlow / Keras.
I trained five different models on the same data and compared them.

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle)

- 50,000 reviews, 25,000 positive and 25,000 negative
- Two columns: `review` (text) and `sentiment` (`positive` / `negative`)

The CSV file is about 66 MB, so it's **not included** in this repo. Download it from Kaggle, rename it to `IMDB-Dataset.csv` and put it next to the notebook.

## Steps

1. **Cleaning the text**: remove HTML tags (`<br />`), keep only letters, lowercase, remove English stopwords (NLTK)
2. **Train / test split**: 80% / 20%
3. **Tokenizing**: Keras `Tokenizer` with the 15,000 most common words (fitted on the training set only)
4. **Padding**: every review is cut/padded to the same length (mean + 2 × std of the review lengths)
5. **Training 5 models** (each one: `Embedding(50)` → 3 recurrent layers with 16 / 8 / 4 units + `Dropout(0.1)` → `Dense(1, sigmoid)`):

| Model | Recurrent layer |
|---|---|
| LSTM | `LSTM` |
| Simple RNN | `SimpleRNN` |
| GRU | `GRU` |
| Bidirectional GRU | `Bidirectional(GRU)` |
| Bidirectional LSTM | `Bidirectional(LSTM)` |

All models use `Adam(learning_rate=1e-3)`, `binary_crossentropy`, 5 epochs, batch size 1000 and a 30% validation split.
For each model the notebook plots training vs validation accuracy and loss, and evaluates it on the test set.

## How to run

**Google Colab** (recommended, turn on the GPU: `Runtime → Change runtime type → T4 GPU`):
click the "Open in Colab" badge and upload `IMDB-Dataset.csv` to the session (or put it in your Google Drive and change `DRIVE_PATH`).

**Locally:**

```bash
git clone https://github.com/mikaeilbarout/Sentiment_analysis_RNN_LSTM.git
cd Sentiment_analysis_RNN_LSTM
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

Put `IMDB-Dataset.csv` in the same folder before running.

## Project structure

```
Sentiment_analysis_RNN_LSTM/
├── sentiment_analysis.ipynb
├── requirements.txt
└── README.md
```
