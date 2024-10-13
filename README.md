# Hamlet Text Generation Model

## Overview

This project involves the creation of a text generation model based on William Shakespeare's play **Hamlet** using TensorFlow and NLTK. The model is designed to predict the next word in a sequence, allowing for generation of text that mimics the style of the play. It uses an LSTM-based Recurrent Neural Network (RNN) to learn and predict word sequences.

### Files:
- **hamlet.txt**: Contains the raw text of Shakespeare's *Hamlet* obtained using NLTK's Gutenberg corpus.
- **README.md**: This file provides an overview of the project.
- **train_model.py**: Python script for training the text generation model.

### Technologies Used:
- **Natural Language Toolkit (nltk)**: Used to download and process the *Hamlet* text.
- **TensorFlow/Keras**: Used to build, train, and evaluate the LSTM model for text generation.
- **Scikit-learn**: Used to split the data into training and testing sets.
- **Pandas and NumPy**: Libraries for data manipulation and matrix operations.

## Model Description

The model is built using the **LSTM (Long Short-Term Memory)** architecture, a type of recurrent neural network that is particularly effective for sequence prediction tasks like text generation. It is trained on sequences of words from the *Hamlet* text and learns to predict the next word in a sequence based on the previous words.

### Model Architecture:
1. **Embedding Layer**: This layer converts the input words into dense vectors of a fixed size, allowing the model to understand the semantic meaning of the words.
2. **LSTM Layer 1**: This LSTM layer processes sequences and returns sequences to feed into the next LSTM.
3. **Dropout Layer**: Regularizes the model by randomly setting a fraction of input units to 0, preventing overfitting.
4. **LSTM Layer 2**: The final LSTM layer processes the sequence output from the previous LSTM and passes the result to the Dense layer.
5. **Dense Layer**: This layer outputs a probability distribution over the entire vocabulary, predicting the next word in the sequence.

### Training:
The model is trained for **100 epochs** with **categorical cross-entropy** as the loss function and the **Adam optimizer**. The input sequences are padded to a fixed length to ensure all sequences are of equal size.

## Steps to Run the Project:

### 1. Install Required Libraries
Make sure the following libraries are installed:
```bash
pip install nltk tensorflow scikit-learn numpy pandas
```

### 2. Download the Dataset
The dataset is automatically downloaded from the **Gutenberg corpus** using NLTK:
```python
import nltk
nltk.download('gutenberg')
```

### 3. Tokenize and Prepare Data
The script reads the *Hamlet* text and tokenizes it, splitting it into sequences of words. Each sequence is padded to ensure they have the same length.

### 4. Train the Model
The training data is split into training and test sets using `train_test_split`. The model is then trained using `model.fit()` with 100 epochs:
```python
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=1)
```

### 5. Predicting Next Word
Once trained, the model can predict the next word in a sequence. The `predict_next_word()` function is provided to predict the next word based on a given text input.

```python
def predict_next_word(model, tokenizer, text, max_sq_len):
    # Tokenizes and pads the input, then predicts the next word
```

### 6. Example Prediction:
You can test the model's prediction capabilities using the following:
```python
predicted_word = predict_next_word(model, tokenizer, "to be or not to be", max_sq_len)
print(predicted_word)
```

## Results:
- **Training Accuracy**: ~73%
- **Validation Accuracy**: ~4.6%
- **Training Loss**: 1.15
- **Validation Loss**: 15.50

The model demonstrates reasonable training accuracy but has limited generalization capabilities, as reflected in the low validation accuracy. Further tuning or use of a larger corpus could help improve performance.

## Improvements:
- Use a larger dataset with additional Shakespeare plays or similar texts.
- Experiment with model architecture (e.g., more layers, different hyperparameters).
- Implement more sophisticated preprocessing techniques such as removing stopwords or performing lemmatization.

## Requirements:
- Python 3.x
- TensorFlow 2.x
- NLTK
- Scikit-learn
- Pandas
- NumPy
