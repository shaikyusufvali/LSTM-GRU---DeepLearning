# LSTM-GRU---DeepLearning
# LSTM and GRU — Deep Learning Notes

## 1. Introduction

LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are advanced types of Recurrent Neural Networks (RNNs).

They are designed to process sequential data and learn relationships between previous and current information.

Examples of sequential data:

* Text and sentences
* Speech and audio
* Time-series data
* Weather measurements
* Stock price history
* Robot sensor readings
* Human activity data

The main purpose of LSTM and GRU is to learn useful information from sequences and use that information to produce predictions.

---

## 2. What Is Sequential Data?

Sequential data is data in which the order of observations matters.

Example:

Time 1: Motor current = 1.2
Time 2: Motor current = 1.3
Time 3: Motor current = 1.5
Time 4: Motor current = 2.1
Time 5: Motor current = 2.8

These values form a sequence because each observation belongs to a particular time.

An LSTM or GRU can learn patterns from historical sensor readings and use them to predict a future value or classify a sequence.

Other examples include predicting the next word in a sentence and forecasting tomorrow's temperature from historical weather measurements.

---

## 3. Why Do We Need LSTM and GRU?

A standard RNN passes hidden-state information from one time step to the next.

However, during training on long sequences, standard RNNs can experience vanishing or exploding gradients.

Vanishing gradients can make it difficult to learn relationships involving information from much earlier time steps.

LSTM and GRU introduce gates that regulate information flow.

Their gates help the model learn which information to preserve, incorporate, or use when producing an output.

Important:

* LSTM and GRU can learn long-range dependencies.
* They do not guarantee that every important detail will be remembered.
* Their performance depends on the data, architecture, training process, and task.

---

# PART A: LSTM

## 4. What Is LSTM?

LSTM stands for Long Short-Term Memory.

It is a gated recurrent neural network architecture designed to learn dependencies across sequences.

An LSTM maintains two main states:

1. Cell State (C_t)
2. Hidden State (h_t)

### Cell State

The cell state carries information through the LSTM sequence.

It provides a pathway through which information can be retained or modified over multiple time steps.

### Hidden State

The hidden state represents information produced by the LSTM at the current time step.

It can be passed to the next time step and used by downstream layers.

---

## 5. LSTM Architecture

The main components of an LSTM cell are:

1. Forget Gate
2. Input Gate
3. Candidate Cell State
4. Cell State Update
5. Output Gate
6. Hidden State Update

The three main gates are the Forget Gate, Input Gate, and Output Gate.

The candidate cell state and state-update equations are important components, but they are not additional gates.

---

## 6. Forget Gate

### Purpose

The Forget Gate determines how much of the previous cell state should be retained.

### Formula

f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)

Where:

* f_t = forget gate output
* W_f = learnable weight matrix
* h_(t-1) = previous hidden state
* x_t = current input
* b_f = bias
* sigmoid = sigmoid activation function

### Sigmoid Function

sigmoid(x) = 1 / (1 + exp(-x))

The sigmoid function produces values between 0 and 1.

Interpretation:

* A value near 0 indicates strong suppression of the corresponding previous cell-state information.
* A value near 1 indicates strong retention.
* Intermediate values indicate partial retention.

### Numerical Example

Suppose:

Previous cell state = 0.8
Forget gate output = 0.25

Retained contribution:

0.25 × 0.8 = 0.2

The previous cell state's contribution to the updated memory is 0.2.

These values are illustrative examples, not trained model parameters.

---

## 7. Input Gate

### Purpose

The Input Gate regulates how much candidate information should be added to the cell state.

### Formula

i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)

Where:

* i_t = input gate output
* W_i = input gate weight matrix
* h_(t-1) = previous hidden state
* x_t = current input
* b_i = bias

The input gate works with the candidate cell state.

---

## 8. Candidate Cell State

### Purpose

The candidate cell state represents new information that may be incorporated into the cell state.

### Formula

C_candidate = tanh(W_C [h_(t-1), x_t] + b_C)

Where:

* C_candidate = candidate cell state
* W_C = candidate-state weight matrix
* b_C = candidate-state bias
* tanh = hyperbolic tangent activation function

The tanh function produces values between -1 and 1.

The candidate cell state can contain positive or negative values.

---

## 9. Cell State Update

### Formula

C_t = f_t × C_(t-1) + i_t × C_candidate

Where:

* C_t = new cell state
* f_t = forget gate output
* C_(t-1) = previous cell state
* i_t = input gate output
* C_candidate = candidate cell state

The equation has two parts:

Part 1:
f_t × C_(t-1)

This represents the retained contribution from previous memory.

Part 2:
i_t × C_candidate

This represents the contribution from new candidate information.

### Numerical Example

Suppose:

Previous cell state = 0.8
Forget gate output = 0.25
Input gate output = 0.7
Candidate cell state = 0.6

Step 1: Retained information

0.25 × 0.8 = 0.2

Step 2: New information

0.7 × 0.6 = 0.42

Step 3: Updated cell state

C_t = 0.2 + 0.42

C_t = 0.62

The updated cell state is 0.62.

All values in this example are illustrative.

---

## 10. Output Gate

### Purpose

The Output Gate regulates how much of the updated cell-state information contributes to the current hidden state.

### Formula

o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)

Where:

* o_t = output gate
* W_o = output gate weight matrix
* b_o = output gate bias

The output gate produces values between 0 and 1.

---

## 11. Hidden State Update

### Formula

h_t = o_t × tanh(C_t)

Where:

* h_t = new hidden state
* o_t = output gate output
* C_t = updated cell state

### Numerical Example

Suppose:

C_t = 0.62
o_t = 0.5

Then:

h_t = 0.5 × tanh(0.62)

h_t is approximately 0.2755.

The hidden state can be passed to the next time step or used by a downstream prediction layer.

The exact result depends on the numerical precision used.

---

## 12. Complete LSTM Information Flow

Current input and previous hidden state
|
v
Forget Gate
|
v
Retain previous cell-state information

Current input and previous hidden state
|
v
Input Gate
|
v
Candidate Cell State
|
v
Cell State Update
|
v
Updated C_t
|
v
Output Gate
|
v
Hidden State h_t

The LSTM passes the updated cell state and hidden state to the next time step.

---

# PART B: GRU

## 13. What Is GRU?

GRU stands for Gated Recurrent Unit.

It is a gated recurrent neural network architecture designed to process sequential data.

GRU uses a hidden state to carry information across time steps.

Unlike a standard LSTM, a GRU does not maintain a separate cell state.

GRU has two main gates:

1. Update Gate
2. Reset Gate

---

## 14. Update Gate

### Purpose

The Update Gate controls the balance between the previous hidden state and the candidate hidden state.

One common GRU convention uses the following equation:

z_t = sigmoid(W_z [x_t, h_(t-1)] + b_z)

Where:

* z_t = update gate output
* W_z = update gate weight matrix
* x_t = current input
* h_(t-1) = previous hidden state
* b_z = bias

The hidden-state update under this convention is:

h_t = (1 - z_t) × h_(t-1) + z_t × h_candidate

The update gate controls the mixture of old and candidate information.

Some implementations use the opposite update-gate convention, so the corresponding equation may look different.

### Numerical Example

Suppose:

Previous hidden state = 0.4
Update gate = 0.7
Candidate hidden state = 0.8

Step 1:

1 - 0.7 = 0.3

Step 2:

0.3 × 0.4 = 0.12

Step 3:

0.7 × 0.8 = 0.56

Step 4:

h_t = 0.12 + 0.56

h_t = 0.68

The new hidden state is 0.68.

These are illustrative values.

---

## 15. Reset Gate

### Purpose

The Reset Gate controls how much of the previous hidden state contributes to the candidate hidden-state calculation.

### Formula

r_t = sigmoid(W_r [x_t, h_(t-1)] + b_r)

Where:

* r_t = reset gate output
* W_r = reset gate weight matrix
* x_t = current input
* h_(t-1) = previous hidden state
* b_r = bias

The candidate hidden state is commonly calculated as:

h_candidate = tanh(W_h [x_t, r_t × h_(t-1)] + b_h)

The reset gate influences the previous hidden-state contribution used to calculate the candidate state.

### Numerical Example

Suppose:

Reset gate = 0.2
Previous hidden state = 0.8

The gated previous-state contribution is:

0.2 × 0.8 = 0.16

This value is part of the candidate-state calculation.

It does not mean the entire previous hidden state is directly erased.

The final candidate hidden state also depends on the input, weights, bias, and activation function.

---

## 16. Complete GRU Information Flow

Current input and previous hidden state
|
+-------+-------+
|               |
v               v
Update Gate      Reset Gate
|               |
|               v
|        Candidate Hidden State
|               |
+-------+-------+
|
v
Hidden State Update
|
v
New Hidden State

The new hidden state is passed to the next time step.

---

# PART C: LSTM vs GRU

## 17. Key Differences

| Feature             | LSTM                                   | GRU                            |
| ------------------- | -------------------------------------- | ------------------------------ |
| Full form           | Long Short-Term Memory                 | Gated Recurrent Unit           |
| Main gates          | Forget, Input, Output                  | Update, Reset                  |
| Cell state          | Separate cell state                    | No separate cell state         |
| Hidden state        | Yes                                    | Yes                            |
| Architecture        | More components                        | Simpler structure              |
| Parameters          | Often more parameters                  | Often fewer parameters         |
| Training speed      | Depends on implementation and hardware | Can be faster in some settings |
| Long-range learning | Designed to support it                 | Designed to support it         |
| Best choice         | Depends on task and data               | Depends on task and data       |

GRU is not automatically better than LSTM, and LSTM is not automatically better than GRU.

Their performance should be compared on the actual task using the same training, validation, and evaluation conditions.

---

# PART D: Real-Life Applications

## 18. Time-Series Forecasting

Examples:

* Weather forecasting
* Electricity demand forecasting
* Traffic prediction
* Sensor-value forecasting

Historical measurements form a sequence.

An LSTM or GRU can learn temporal patterns and predict a future value.

Example:

Input:
Previous 30 days of temperature measurements

Output:
Predicted next-day temperature

The model must be trained and evaluated on suitable historical data.

---

## 19. Speech Recognition

Speech consists of audio measurements that change over time.

LSTM and GRU can process sequences of audio features.

Possible workflow:

Audio
|
Audio feature extraction
|
LSTM or GRU
|
Speech-related output

Modern speech-recognition systems often use Transformer-based or other architectures instead of relying exclusively on LSTM or GRU.

---

## 20. Natural Language Processing

LSTM and GRU can process ordered sequences of words or tokens.

Applications include:

* Sentiment classification
* Sequence labelling
* Text classification
* Older next-word prediction systems
* Some language-generation tasks

Example:

Input:
"The movie was excellent."

Output:
Positive sentiment

The model learns relationships between the sequence and the target label during training.

Modern large language models commonly use Transformer architectures.

---

## 21. Robotics and Predictive Maintenance

Robot sensors generate telemetry over time.

Example:

Time 1: Motor current = 1.2
Time 2: Motor current = 1.3
Time 3: Motor current = 1.5
Time 4: Motor current = 2.1
Time 5: Motor current = 2.8

A trained LSTM or GRU may learn patterns from:

* Motor current
* Joint position
* Velocity
* Temperature
* Vibration
* Other available telemetry features

Possible outputs:

* Fault category
* Anomaly score
* Predicted sensor value
* Estimated future failure risk

Important:
Increasing current alone does not prove that a motor is faulty. Reliable fault diagnosis requires suitable training labels or anomaly-detection methods, evaluation on unseen data, and safety validation.

---

# PART E: Python Implementation

## 22. Install TensorFlow

In the VS Code terminal:

pip install tensorflow

If TensorFlow is already installed, verify it:

python -c "import tensorflow as tf; print(tf.**version**)"

---

## 23. LSTM Regression Example

This is a small architecture demonstration using randomly generated data.

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# 100 samples, 10 time steps, 3 features
X = tf.random.normal((100, 10, 3))

# One target value per sequence
y = tf.random.normal((100, 1))

model = Sequential([
    Input(shape=(10, 3)),
    LSTM(16),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

model.fit(
    X,
    y,
    epochs=2,
    batch_size=16
)

prediction = model.predict(X[:1], verbose=0)

print("Prediction:", prediction)
```

The random data is used only to demonstrate the architecture and execution flow. The resulting model is not a meaningful real-world predictor.

---

## 24. GRU Regression Example

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, GRU, Dense

# 100 samples, 10 time steps, 3 features
X = tf.random.normal((100, 10, 3))

# One target value per sequence
y = tf.random.normal((100, 1))

model = Sequential([
    Input(shape=(10, 3)),
    GRU(16),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

model.fit(
    X,
    y,
    epochs=2,
    batch_size=16
)

prediction = model.predict(X[:1], verbose=0)

print("Prediction:", prediction)
```

The random data is for practice only. Replace it with a properly prepared real dataset for a real application.

---

# PART F: Important Formulas

## 25. LSTM Formulas

Forget Gate:

f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)

Input Gate:

i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)

Candidate Cell State:

C_candidate = tanh(W_C [h_(t-1), x_t] + b_C)

Cell State Update:

C_t = f_t × C_(t-1) + i_t × C_candidate

Output Gate:

o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)

Hidden State Update:

h_t = o_t × tanh(C_t)

## 26. GRU Formulas

Update Gate:

z_t = sigmoid(W_z [x_t, h_(t-1)] + b_z)

Reset Gate:

r_t = sigmoid(W_r [x_t, h_(t-1)] + b_r)

Candidate Hidden State:

h_candidate = tanh(W_h [x_t, r_t × h_(t-1)] + b_h)

Hidden State Update:

h_t = (1 - z_t) × h_(t-1) + z_t × h_candidate

These GRU equations use the convention where z_t weights the candidate hidden state.

---

# PART G: Revision Checklist

## 27. LSTM Revision

* [ ] Explain why LSTM was developed.
* [ ] Explain vanishing gradients.
* [ ] Explain hidden state.
* [ ] Explain cell state.
* [ ] Explain the Forget Gate.
* [ ] Calculate the Forget Gate contribution.
* [ ] Explain the Input Gate.
* [ ] Explain the candidate cell state.
* [ ] Calculate the updated cell state.
* [ ] Explain the Output Gate.
* [ ] Calculate the new hidden state.
* [ ] Write all six LSTM equations.
* [ ] Build an LSTM model using TensorFlow/Keras.
* [ ] Identify real-life LSTM applications.

## 28. GRU Revision

* [ ] Explain why GRU was developed.
* [ ] Explain the Update Gate.
* [ ] Explain the Reset Gate.
* [ ] Calculate the gated previous-state contribution.
* [ ] Calculate the new hidden state.
* [ ] Write all GRU equations.
* [ ] Build a GRU model using TensorFlow/Keras.
* [ ] Compare GRU with LSTM.
* [ ] Identify real-life GRU applications.

## 29. Interview Questions

1. What is LSTM?
2. Why do we use LSTM instead of a standard RNN?
3. What is the vanishing-gradient problem?
4. What is the difference between cell state and hidden state?
5. What are the three main LSTM gates?
6. What does the Forget Gate do?
7. What is the purpose of the Input Gate?
8. Why do we use tanh for the candidate cell state?
9. What is the role of the Output Gate?
10. What is GRU?
11. What are the two main GRU gates?
12. How does the Update Gate work?
13. How does the Reset Gate work?
14. Why does GRU not need a separate cell state?
15. How do LSTM and GRU differ?
16. Which architecture should be selected for a particular dataset, and how would you evaluate that decision?
17. How can LSTM or GRU be used for time-series forecasting?
18. How can sequence models help detect patterns in robot telemetry?

---

# 30. Final Summary

LSTM and GRU are gated recurrent neural network architectures for sequential data.

LSTM:

* Uses a hidden state and a separate cell state.
* Has three main gates: Forget, Input, and Output.
* Controls how information is retained, incorporated, and exposed.

GRU:

* Uses a hidden state without a separate cell state.
* Has two main gates: Update and Reset.
* Controls the mixture of previous and candidate hidden-state information.

Both architectures can be useful for sequence modelling. The appropriate choice depends on the dataset, task, computational requirements, and evaluation results.

**Recommended learning order:**

1. RNN revision
2. LSTM architecture
3. LSTM gates and numerical calculations
4. GRU architecture
5. GRU gates and numerical calculations
6. Python implementation
7. Real-world sequence-modelling project
8. Comparison and interview revision

---

## 👨‍💻 Author

**Shaik Yusufvali**
Aspiring AI & Machine Learning Engineer | Deep Learning Enthusiast

**Topics Covered:** Artificial Neural Networks (ANN), Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).

**Focus:** Understanding neural network architectures, gate mechanisms, mathematical calculations, Python implementation, and real-world applications.

---
