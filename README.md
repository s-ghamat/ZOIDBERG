# Zoidberg

Machine learning project for detecting pneumonia from chest X-ray images.

The project uses image preprocessing, PCA dimensionality reduction, and classical machine learning models to classify radiographs as either normal or pneumonia cases.

## Overview

Zoidberg was developed as part of an Epitech academic project.

The objective is to build a complete pneumonia detection pipeline, from data exploration and preprocessing to model training, evaluation, prediction, and testing.

## Scope

* Chest X-ray image preprocessing
* Data exploration and visualization
* Feature extraction
* PCA dimensionality reduction
* Model training and comparison
* Pneumonia prediction on new images
* Single-image and batch prediction
* Unit testing
* Final project report

## Project Structure

```text
Zoidberg/
├── data/
├── src/
│   ├── preprocessing/
│   │   └── preprocess_images.py
│   ├── visualization/
│   │   └── visualizer.py
│   ├── models/
│   │   └── predict.py
│   └── utils/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training_sklearn_with_pca.ipynb
├── models/
│   └── regression_logistic_model.pkl
├── tests/
│   ├── test_preprocessing.py
│   └── test_predict.py
├── demo.py
├── index.html
├── rapport_pneumonie.pdf
└── requirements.txt
```

## Methods

Several classical machine learning models were trained and compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine

The Logistic Regression model achieved the best performance and is saved for prediction.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Single Image Prediction

```bash
python demo.py path/to/image.jpg
```

### Batch Prediction

```bash
python demo.py --batch path/to/folder
```

### Python API

```python
from src.models.predict import predict_pneumonia

result, probability = predict_pneumonia("path/to/image.jpg")

print(f"Result: {result}")
print(f"Probability: {probability:.4f}")
```

## Notebooks

| Notebook                                             | Purpose                 |
| ---------------------------------------------------- | ----------------------- |
| `notebooks/01_exploration.ipynb`                     | Dataset exploration     |
| `notebooks/02_preprocessing.ipynb`                   | Image preprocessing     |
| `notebooks/03_model_training_sklearn_with_pca.ipynb` | Model training with PCA |

## Results

The Logistic Regression model achieved the best performance.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 95.94% |
| Precision | 97.81% |
| Recall    | 96.70% |
| F1-score  | 97.25% |

The model shows strong performance for pneumonia detection while maintaining reliable classification of normal cases.

## Tests

Run the unit tests:

```bash
python -m unittest tests/test_predict.py
python -m unittest tests/test_preprocessing.py
```

## Limitations

* The model does not distinguish between viral and bacterial pneumonia.
* X-ray image quality can affect prediction reliability.
* The project is academic and not intended for clinical diagnosis.
* Future work could include deep learning models such as CNNs.

## Report

The full project report is available at:

```text
rapport_pneumonie.pdf
```

## Author

Setayesh Ghamat
