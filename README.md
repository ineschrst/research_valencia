## Machine Learning Models for Predicting Action Potential Sequences

A machine learning pipeline to analyze ischemic cardiac signals by transforming high-dimensional Action Potential (AP) waves into a symbolic alphabet for sequence prediction.

---

## Table of Contents
- [Methodology](#methodology)
- [Data](#data--segmentation)
- [Machine Learning Models](#machine-learning-models)
- [Running the Code](#running-the-code)
- [Results & Figures](#results--figures)

---

## Methodology

The primary objective of this research internship was to evaluate the efficacy of various Machine Learning architectures in predicting cardiac Action Potential (AP) sequences. Leveraging data from ischemia simulations, the project utilizes two distinct segmentation strategies: Mode 0 (isolated pulses) and Mode 1 (pulses including diastolic intervals).To handle the high dimensionality of the raw signals (approximately 50,000 pulses), I implemented a Symbolic Alphabet approach. This involves applying Principal Component Analysis (PCA) to reduce the signals to 12 core components (sufficient to capture 99.9% of variance at Minute 0) and subsequently clustering these components using K-Means. By predicting discrete cluster labels (alphabet size $\approx$ 10) rather than raw data points, the computational complexity is significantly reduced.The predictive pipeline utilizes Random Forest, LSTM, GRU, and CNN models to forecast future 'letters' (horizon) based on past sequences (lookback). Evaluation is performed on independent test sets, measuring both classification accuracy and signal similarity—accounting for reconstruction errors introduced by PCA and centroid approximation. Results indicate that the models outperformed initial expectations in capturing the temporal evolution of the ischemic signals.

---

## Data

*Data Availability*: The simulated ischemic action potential data (`.mat` and `.npy`) used in this project was provided by CoMMLab at the Unversity of Valencia. Consequently, the raw datasets are not included in this repository.

The models were primarily trained and tested on Minute 0 data, with further evaluation across different stages of ischemic progression (Minutes 3, 5, 9, and 13).

--- 

## Machine Learning Models 

This project uses machine learning to predict AP sequences after processing the raw signals into a symbolic alphabet. Used Models:

- Random Forest (RF) – classification-based prediction
- LSTM – sequence prediction with memory
- GRU – sequence prediction with gated updates
- CNN – sequence prediction using convolutional feature extraction

Random Forest
```
RandomForestClassifier(n_estimators=200, n_jobs=-1)
```

LSTM 
```
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(self.look_back, self.nclusters)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(self.nclusters*self.hp, activation='softmax'),
    Reshape((self.hp, self.nclusters))
])
optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)       
```

GRU 
```
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(self.look_back, self.nclusters)),
    Dropout(0.2),
    GRU(32, return_sequences=False),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(self.nclusters*self.hp, activation='softmax'),
    Reshape((self.hp, self.nclusters))
])
optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
```

CNN
```
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(self.look_back, self.nclusters)),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(self.nclusters*self.hp, activation='softmax'),
    Reshape((self.hp, self.nclusters))
])
optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
```

---

## Running the Code 

All execution scripts are located in the `code/` directory and should be run from the project root.

#### 1. Configuration & Core Functions
All core logic, including data preprocessing, segmentation, and ML model definitions (CNN, LSTM, GRU, etc.), is contained in:
* `code/ap_utils.py`

#### 2. Training and Single-Step Prediction
To train a model and evaluate it on a single-step horizon:
`alpha_hp.py`: Adjust parameters like lookback or batch_size within the script. Results are logged to `results/alpha_results.txt`.

#### 3. Stepwise (Recursive) Prediction
To train a model and evaluate it on a stepwise horizon by feeding its own predictions back into itself. 
`alpha_stepwise.py`: Adjust parameters like stepsize or batch_size within the script. Results are logged to `results/stepwise_results.txt`.

--- 

## Results & Figures 
* `results/`: Contains exemplary logging files showing model performance.

* `figures/`: Contains exemplary figures including raw pulses, PCA clusters, and prediction overlays.

*  Detailed Analysis: For more information on the research results, refer to the included `presentation_Ines_Commlab.pdf`.

