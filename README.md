
# Hamlet Text Generation Model

## Overview

This project creates a text generation model based on William Shakespeare's play **Hamlet** using TensorFlow and NLTK. The goal is to predict the next word in a sequence using both **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) architectures. The performance of the two models is compared in terms of training accuracy, validation accuracy, and loss.

## Technologies Used

- **Natural Language Toolkit (nltk)**: To download and process the *Hamlet* text.
- **TensorFlow/Keras**: To build, train, and evaluate the LSTM and GRU models.
- **Scikit-learn**: For splitting the dataset into training and testing sets.
- **Pandas and NumPy**: Libraries used for data manipulation and matrix operations.

## Model Description

### LSTM Model Architecture:
1. **Embedding Layer**: Converts input words into dense vectors.
2. **LSTM Layer 1**: Processes sequences and returns sequences.
3. **Dropout Layer**: Regularizes the model to prevent overfitting.
4. **LSTM Layer 2**: Processes the output sequence from the previous LSTM layer.
5. **Dense Layer**: Outputs a probability distribution over the entire vocabulary, predicting the next word.

### GRU Model Architecture:
1. **Embedding Layer**: Converts input words into dense vectors.
2. **GRU Layer 1**: Processes sequences and returns sequences.
3. **Dropout Layer**: Regularizes the model to prevent overfitting.
4. **GRU Layer 2**: Processes the output sequence from the previous GRU layer.
5. **Dense Layer**: Outputs a probability distribution over the entire vocabulary, predicting the next word.

### Training Setup:
- **Epochs**: 100
- **Loss Function**: Categorical cross-entropy
- **Optimizer**: Adam
- **Test Split**: 80% training, 20% testing
- Both models were trained with the same data and under similar conditions for a fair comparison.

## Model Comparison

| Model        | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|--------------|-------------------|---------------|---------------------|-----------------|
| **LSTM**     | 73.13%             | 1.1503        | 4.62%               | 15.5095         |
| **GRU**      | 80.07%             | 0.8107        | 5.34%               | 12.1686         |

### Key Observations:
1. **Training Performance**: The GRU model achieves higher training accuracy (80.07%) compared to the LSTM model (73.13%). It also has a lower training loss (0.8107 vs. 1.1503).
   
2. **Validation Performance**: The validation accuracy of the GRU model (5.34%) is slightly better than the LSTM model (4.62%). Similarly, the GRU model exhibits a lower validation loss (12.1686 vs. 15.5095).

3. **Training Time**: The GRU model generally trains faster per epoch than the LSTM model. This can be attributed to GRU's simpler architecture, which makes it less computationally intensive.

## Conclusion

Both the LSTM and GRU models show promising results for text generation based on the *Hamlet* text. However, the **GRU model** outperforms the LSTM model in terms of accuracy and loss on both the training and validation sets. Given its faster training time and comparable or better performance, the GRU model may be more suitable for this task.

## Steps to Run the Project

### 1. Install Required Libraries
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
The script reads the *Hamlet* text, tokenizes it, and splits it into sequences of words, padded to the same length.

### 4. Train the Models
Train both the LSTM and GRU models using the preprocessed data:
```python
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=1)
```

### 5. Predicting the Next Word
Use the trained model to predict the next word based on input text:
```python
predicted_word = predict_next_word(model, tokenizer, "to be or not to be", max_sq_len)
print(predicted_word)
```

## Requirements
- Python 3.x
- TensorFlow 2.x
- NLTK
- Scikit-learn
- Pandas
- NumPy
```
