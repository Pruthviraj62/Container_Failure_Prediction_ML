# 🐳 Machine Learning for Predicting Docker Container Failures

**College:** Annasaheb Dange College of Engineering and Technology, Ashta
**Department:** Computer Science and Engineering
**Subject:** Machine Learning
**Student:** Pruthviraj Santosh Patil | URN: 23031042 | Roll No: 3005 | Class: TY-A
**Guide:** Prof. S. D. Khatavkar
**Academic Year:** 2025–2026

---

## 📌 What This Project Is About

When companies run software at large scale, they use **Docker containers** — small, self-contained packages that hold an application and everything it needs to run. A company like Netflix or Google runs hundreds of thousands of these containers at the same time.

The problem is — containers fail. They crash due to memory leaks, CPU overloads, or rising error rates. The old way of handling this was **reactive** — wait for the crash, then fix it. By then, users are already affected.

This project builds a **Machine Learning model using LSTM (Long Short-Term Memory)** that reads container metrics over time and **predicts a failure before it happens** — giving engineers time to act before any crash occurs.

This is called **AIOps** — Artificial Intelligence for IT Operations.

---

## 🎯 Project Goal

> Predict whether a Docker container will fail in the near future by analysing its CPU%, Memory%, Error Count, and Network I/O over time — using a supervised deep learning model (LSTM).

---

## 🧠 What Type of Machine Learning Is Used

| Type | Algorithm Used | Purpose |
|---|---|---|
| **Supervised Learning** | LSTM Neural Network | Main model — trained on labelled historical failure data |
| **Unsupervised** | (Reference only) | Isolation Forest for anomaly detection — mentioned in blog |
| **Reinforcement** | (Reference only) | Auto-healing agents — emerging in industry |

This project specifically implements **Supervised Learning with LSTM**.

---

## 📂 Project Structure

```
Container_Failure_Prediction/
│
├── Container_Failure_Prediction_LSTM.ipynb   ← Main notebook (run this)
├── README.md                                  ← This file
│
└── Output files (auto-generated when notebook runs):
    ├── metrics_over_time.png       ← Plot of CPU, Memory, Errors over time
    ├── metric_distributions.png   ← Healthy vs Failing container comparison
    ├── correlation_matrix.png      ← Feature correlation heatmap
    ├── training_history.png        ← Loss and Accuracy curves
    ├── confusion_matrix.png        ← Model evaluation matrix
    └── prediction_vs_actual.png   ← Predicted vs actual failure labels
```

---

## 🛠️ Requirements

The notebook **installs everything automatically**. You do not need to install anything manually.

Libraries used:

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations and array handling |
| `pandas` | Data loading, manipulation, and DataFrames |
| `matplotlib` | Plotting charts and visualisations |
| `seaborn` | Statistical visualisation (heatmaps, distributions) |
| `scikit-learn` | MinMaxScaler, train/test split, evaluation metrics |
| `tensorflow / keras` | Building and training the LSTM model |

**Python version:** 3.8 or above
**Tested on:** Google Colab, JupyterLab

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended, Zero Setup)
1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload Notebook**
3. Upload `Container_Failure_Prediction_LSTM.ipynb`
4. Click **Runtime → Run All**
5. Wait 2–3 minutes. All outputs appear automatically.

### Option 2 — JupyterLab (Local)
1. Open JupyterLab
2. Upload the `.ipynb` file
3. Click **Kernel → Restart Kernel and Run All Cells**

> ⚠️ Always run **Cell 1 first**. It installs all packages. If you skip it, every other cell will fail with `NameError`.

---

## 📊 Dataset

The dataset is **synthetically generated** inside the notebook itself — no external download needed.

| Property | Value |
|---|---|
| Total Rows | 3000 |
| Time Resolution | Every 5 minutes |
| Total Time Covered | ~250 hours |
| Failure Rate | ~15–17% of rows |

### Features (Input Columns)

| Column | Description | Range |
|---|---|---|
| `CPU_Percent` | CPU usage of the container | 0 – 100% |
| `Memory_Percent` | Memory usage of the container | 0 – 100% |
| `Error_Count` | Number of errors logged per time step | 0 – 20+ |
| `Network_IO` | Network input/output activity | 0 – 100 |

### Label (Target Column)

| Value | Meaning |
|---|---|
| `0` | Container is healthy |
| `1` | Container is in a failure window (last 20% before crash) |

### How the Data Is Generated

- **Healthy containers** produce stable, random readings in normal ranges.
- **Failing containers** show a gradual rise in CPU, Memory, and Error Count over 30–60 time steps — simulating a real memory leak or overload scenario. Only the final 20% of the failure window is labelled as `1`.

---

## 🔄 Project Workflow (Step by Step)

```
Step 1: Install and Import Libraries
          ↓
Step 2: Generate Dataset
        (3000 rows, 4 features + 1 label)
          ↓
Step 3: Exploratory Data Analysis (EDA)
        - Plot metrics over time
        - Compare healthy vs failing distributions
        - Correlation heatmap
          ↓
Step 4: Preprocess Data
        - Normalise features to [0, 1] using MinMaxScaler
        - Create 20-step sequence windows for LSTM input
        - Split 80% train / 20% test
          ↓
Step 5: Build LSTM Model
        - Layer 1: LSTM(64 units) + Dropout(0.2)
        - Layer 2: LSTM(32 units) + Dropout(0.2)
        - Output: Dense(1, sigmoid)
          ↓
Step 6: Train the Model
        - Epochs: 30 (with Early Stopping, patience=5)
        - Batch size: 32
        - Validation split: 15%
          ↓
Step 7: Evaluate the Model
        - Accuracy, Precision, Recall, F1-Score
        - Confusion Matrix
        - Prediction vs Actual plot
          ↓
Step 8: Summary and Conclusion
```

---

## 🏗️ Model Architecture

```
Input Shape: (20 time steps, 4 features)
        ↓
LSTM Layer 1 — 64 units, return_sequences=True
        ↓
Dropout — 20% (prevents overfitting)
        ↓
LSTM Layer 2 — 32 units, return_sequences=False
        ↓
Dropout — 20%
        ↓
Dense Output — 1 unit, sigmoid activation
        ↓
Output: Probability between 0 and 1
        (> 0.5 = predicted failure)
```

### Why LSTM?

A regular neural network looks at one snapshot of data at a time. LSTM reads data **as a sequence over time** — it remembers what happened 20 steps ago and connects it to what is happening now.

The key mechanism is the **Forget Gate**:

```
f_t = σ ( W_f · [ h_(t-1) , x_t ] + b_f )
```

- `x_t` — current input (CPU, Memory, Errors, Network I/O at this moment)
- `h_(t-1)` — what the model remembered from the previous step
- `W_f` — learned weight matrix
- `b_f` — bias term
- `σ` — sigmoid function (output between 0 and 1)

If `f_t` is close to **0** → forget old memory.
If `f_t` is close to **1** → keep old memory.

This is what allows LSTM to remember a memory spike from 3 hours ago and connect it to a crash prediction now — something a plain RNN cannot do.

---

## 📈 Evaluation Metrics Explained

| Metric | Formula | What It Means in This Project |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of all predicted failures, how many were real? (avoids false alarms) |
| **Recall** | TP / (TP + FN) | Of all real failures, how many did we catch? (avoids missing crashes) |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean — the balanced score |

> In this project, **Recall is more important than Precision** — missing a real failure (False Negative) is more dangerous than a false alarm (False Positive).

### Reading the Confusion Matrix

```
                  Predicted Healthy    Predicted Failure
Actual Healthy         TN                   FP  ← False Alarm
Actual Failure         FN  ← Missed!        TP  ← Correctly Caught
```

---

## 💡 Key Concepts for New Members

**What is Docker?**
Docker wraps an application and all its dependencies into a single portable container. It runs the same on any machine.

**What is a Container Failure?**
A container crash caused by memory overflow, CPU throttling, or unhandled errors. At scale, even 0.1% failure rate means hundreds of containers crashing simultaneously.

**What is AIOps?**
Artificial Intelligence for IT Operations — using ML to automate monitoring, anomaly detection, and failure prediction in large-scale systems.

**What is the Vanishing Gradient Problem?**
A problem in standard RNNs where gradients become too small during backpropagation, causing the model to forget long-term patterns. LSTM solves this with its gate mechanisms.

**What is Early Stopping?**
A training technique that stops training when the validation loss stops improving. Prevents the model from overfitting (memorising training data instead of learning general patterns).

**What is MinMaxScaler?**
Normalises all feature values to the range [0, 1] so that no single feature dominates the model due to its raw numeric scale.

---

## 📋 Expected Results

After running the notebook you should see approximately:

| Metric | Expected Value |
|---|---|
| Accuracy | 88 – 93% |
| Precision | 75 – 88% |
| Recall | 80 – 92% |
| F1-Score | 78 – 90% |

> Exact values vary slightly every run due to randomness in data generation and model initialisation — even with the random seed set.

---

## 🔍 Output Plots Explained

| Plot | File | What to Look For |
|---|---|---|
| Metrics Over Time | `metrics_over_time.png` | Red shaded zones = failure windows. Rising CPU and Memory before each red zone. |
| Distributions | `metric_distributions.png` | Failing containers (red) shifted right = higher values than healthy (green). |
| Correlation Matrix | `correlation_matrix.png` | CPU_Percent and Memory_Percent should show highest correlation with Failure. |
| Training History | `training_history.png` | Both loss curves going down and converging = model is learning correctly. |
| Confusion Matrix | `confusion_matrix.png` | High TP, low FN = good Recall. High TN, low FP = good Precision. |
| Prediction vs Actual | `prediction_vs_actual.png` | Blue line crossing 0.5 threshold should align with red actual failure zones. |

---

## ⚠️ Common Issues and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: seaborn` | Cell 1 not run | Run Cell 1 first — it installs everything |
| `NameError: df is not defined` | Cells run out of order | Restart kernel and run all cells from top |
| `NameError: model is not defined` | Skipped a cell | Always run cells in order 1 → 8 |
| Training very slow | No GPU | Use Google Colab (free GPU available under Runtime → Change Runtime Type → T4 GPU) |

---

## 📚 References

1. **Splunk** — AIOps Explained: https://www.splunk.com/en_us/blog/learn/aiops.html
2. **IBM** — What is AIOps: https://www.ibm.com/think/topics/aiops
3. **InfoWorld** — Predictive Monitoring with AIOps: https://www.infoworld.com/article/4075057
4. **Microsoft Azure** — AIOps in Azure Monitor: https://learn.microsoft.com/en-us/azure/azure-monitor/aiops/aiops-machine-learning
5. **ResearchGate** — Fault Prediction in Cloud-Native Settings: https://www.researchgate.net/publication/391320615

---

## 👤 Author

**Pruthviraj Santosh Patil**
URN: 23031042 | Roll No: 3005 | Class: TY-A
Annasaheb Dange College of Engineering and Technology, Ashta
Under Guidance of: Prof. S. D. Khatavkar

---

*"Docker gave us portable containers. Kubernetes gave us automatic restarts. LSTM-based AIOps gives us prediction — we fix the container before it breaks, not after."*
