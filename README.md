# 📊 Customer Churn Prediction Using ANN

An end-to-end **Customer Churn Prediction** project using an **Artificial Neural Network (ANN)** to predict whether a customer is likely to leave a service.

The project includes data preprocessing, ANN-based classification, model serialization, and an interactive **Streamlit web application** for making real-time predictions.

---

## 🚀 Project Overview

Customer churn is an important business problem where companies try to identify customers who are likely to stop using their products or services.

This project uses an **Artificial Neural Network (ANN)** to learn patterns from customer data and predict whether a customer will churn.

The trained model and preprocessing pipeline are saved as serialized files and used by a Streamlit application to provide predictions through a user-friendly interface.

### 🎯 Objective

The main objectives of this project are:

* Analyze customer-related information.
* Preprocess numerical and categorical features.
* Train an Artificial Neural Network for binary classification.
* Save the trained model for future predictions.
* Build an interactive Streamlit application.
* Provide real-time customer churn predictions.

---

## 🧠 Machine Learning Workflow

```text
                Customer Dataset
                       │
                       ▼
              Data Preprocessing
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    Numerical Features       Categorical Features
          │                         │
          └────────────┬────────────┘
                       ▼
             Preprocessed Data
                       │
                       ▼
                ANN Classifier
                       │
                       ▼
              Model Evaluation
                       │
                       ▼
              Saved Model (.keras)
                       │
                       ▼
             Streamlit Web App
                       │
                       ▼
              Churn Prediction
```

---

## ✨ Features

* 🤖 Artificial Neural Network based classification
* 🧹 Data preprocessing pipeline
* 🔢 Numerical and categorical feature processing
* 💾 Serialized trained model
* 💾 Serialized preprocessing pipeline
* 🌐 Interactive Streamlit application
* ⚡ Real-time prediction
* 📈 Probability-based churn prediction
* 🐍 Python-based implementation

---

## 🛠️ Technologies Used

| Technology         | Purpose                               |
| ------------------ | ------------------------------------- |
| Python             | Programming language                  |
| Pandas             | Data manipulation                     |
| NumPy              | Numerical computation                 |
| Scikit-learn       | Data preprocessing and ML utilities   |
| TensorFlow / Keras | Artificial Neural Network             |
| Streamlit          | Web application                       |
| Pickle             | Model serialization                   |
| Jupyter Notebook   | Model development and experimentation |


---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Sohomdas2004/Churn-Prediction-Using-ANN.git
```

Navigate into the project:

```bash
cd Churn-Prediction-Using-ANN
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment.

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, the major dependencies include:

```bash
pip install pandas numpy scikit-learn tensorflow streamlit
```

---

# ▶️ Running the Application

After installing the dependencies, start the Streamlit application.

If `app.py` is inside the `App` directory:

```bash
streamlit run App/app.py
```

If `app.py` is located in the root directory:

```bash
streamlit run app.py
```

Streamlit will start a local server and provide a URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 🔮 How the Prediction Works

The application follows these steps:

### Step 1 — User Input

The user provides the required customer information through the Streamlit interface.

### Step 2 — Preprocessing

The input data is passed through the saved preprocessing pipeline:

```text
User Input
    ↓
preprocessor.pkl
    ↓
Transformed Features
```

This ensures that new input data receives the same preprocessing treatment used during model training.

### Step 3 — ANN Prediction

The processed features are passed to the trained classifier:

```text
Processed Features
        ↓
  classifier.pkl
        ↓
Prediction Probability
        ↓
Churn / No Churn
```

### Step 4 — Result

The application displays the predicted churn status to the user.

---

# 🧠 Artificial Neural Network

The core predictive model is an **Artificial Neural Network**, which is suitable for learning nonlinear relationships between customer characteristics and churn behavior.

The general architecture follows:

```text
Input Features
      │
      ▼
Dense Layer
      │
      ▼
Hidden Layer(s)
      │
      ▼
Output Layer
      │
      ▼
Churn Probability
```

For binary classification, the final output represents the probability that a customer belongs to the churn class.

---

# 📊 Prediction Output

The model performs binary classification:

| Prediction | Meaning                            |
| ---------- | ---------------------------------- |
| `0`        | Customer is predicted not to churn |
| `1`        | Customer is predicted to churn     |

The application can also use the model's prediction probability to provide a more informative result.

For example:

```text
Churn Probability: 82.4%

Prediction: Customer likely to churn
```

---

# 💼 Business Use Case

Customer churn prediction can help businesses identify customers who may be at risk of leaving.

Potential applications include:

* 📞 Targeted customer retention campaigns
* 🎁 Personalized offers
* 💬 Customer support prioritization
* 📈 Customer retention analysis
* 💰 Reducing potential revenue loss
* 🔍 Identifying patterns associated with customer churn

The model should be treated as a decision-support tool rather than a replacement for business judgment.

---

# 📌 Important Considerations

### Data Preprocessing

The same preprocessing used during training must be applied to new customer data.

That's why the project stores:

```text
preprocessor.pkl
```

Using the saved preprocessing object helps prevent inconsistencies between training and prediction.

### Model Serialization

The trained model is stored as:

```text
classifier.pkl
```

This allows the application to load the trained model without retraining it every time the Streamlit application starts.

---

# 🧪 Model Development

The general development pipeline is:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Feature Preprocessing
   ↓
ANN Model
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

# 🌐 Streamlit Application

The Streamlit application provides a simple interface for interacting with the trained model.

Instead of manually executing Python code, users can enter customer information through the web interface and receive a prediction.

Run the application with:

```bash
streamlit run App/app.py
```

---

# 📦 Model Files

### `classifier.keras`

Contains the trained classification model used to predict customer churn.

### `preprocessor.pkl`

Contains the preprocessing pipeline required to transform input features into the format expected by the trained model.

Keeping both files together is important for reliable inference.

---

# 🔧 Future Improvements

Several improvements can be added to make the project more production-ready:

* [ ] Add detailed model evaluation metrics
* [ ] Add confusion matrix visualization
* [ ] Add ROC-AUC curve
* [ ] Add precision, recall and F1-score
* [ ] Perform hyperparameter tuning
* [ ] Compare ANN with Logistic Regression, Random Forest and XGBoost
* [ ] Add feature importance / explainability
* [ ] Add SHAP explanations
* [ ] Add batch prediction using CSV files
* [ ] Add Docker support
* [ ] Deploy the Streamlit application
* [ ] Add automated testing
* [ ] Add CI/CD using GitHub Actions

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

* Python programming
* Data preprocessing
* Supervised machine learning
* Binary classification
* Artificial Neural Networks
* TensorFlow / Keras
* Model serialization
* Streamlit application development
* Machine-learning deployment workflow

---

# 👨‍💻 Author

**Sohom Das**

GitHub:
https://github.com/Sohomdas2004

---

# ⭐ If You Like This Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and learning purposes.
