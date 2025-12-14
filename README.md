# 📱 Social Media Behavior & Productivity Analysis

## 📌 Project Overview

This project analyzes how **social media usage patterns** influence **sleep habits, concentration, procrastination behavior, and overall productivity satisfaction** among students.

Using **Exploratory Data Analysis (EDA)** and **Machine Learning models**, the project delivers:

- Behavioral insights
- Productivity prediction
- User segmentation through clustering
- An interactive **Streamlit web application**

---

## 🎯 Objectives

- Analyze social media usage patterns among students  
- Study its impact on sleep, focus, and productivity  
- Predict productivity satisfaction levels  
- Segment users into meaningful behavioral clusters  
- Build a real-world ML-powered interactive dashboard  

---

## 🗂 Dataset Description

The dataset consists of survey-based responses with the following features:

### 🔹 Demographic

- `age`

### 🔹 Social Media Behavior

- `daily_social_media_hours`
- `primary_social_media_platform`
- `peak_social_media_time`
- `use_social_media_while_studying`

### 🔹 Lifestyle & Habits

- `avg_sleep_hours`
- `phone_use_after_bed`
- `procrastination_frequency`
- `social_media_affects_concentration`

### 🎯 Target Variable

- `productivity_satisfaction`

**Target Classes (Ordinal):**

- Very dissatisfied  
- Not satisfied  
- Neutral  
- Satisfied  
- Highly satisfied  

---

## 🧹 Data Cleaning & Preprocessing

### 1️⃣ Column Cleaning

- Removed extra spaces and hidden newline characters
- Renamed columns into ML-friendly `snake_case`

### 2️⃣ Encoding Strategy

| Feature Type | Encoding Method |
|-------------|----------------|
| Ordinal features | `OrdinalEncoder` |
| Binary features | Manual mapping (`Yes → 1`, `No → 0`) |
| Nominal features | One-hot encoding |
| Numeric features | StandardScaler |

### Ordinal Encoding Order

Logical ordering was preserved for:

- Social media usage duration
- Sleep duration
- Procrastination frequency
- Productivity satisfaction

This ensures **semantic correctness** during model training.

---

## 📊 Exploratory Data Analysis (EDA)

### 🔹 Age Distribution

- Most respondents are between **18–24 years**
- Indicates a student-dominated dataset

### 🔹 Daily Social Media Usage

- Majority spend **2–5 hours daily**
- A significant portion exceeds **5 hours/day**

### 🔹 Platform Preference

- Instagram, WhatsApp, and YouTube are dominant
- Professional platforms show lower usage

### 🔹 Sleep vs Phone Usage

- Phone usage after bedtime correlates with reduced sleep
- Behavioral impact clearly visible

### 🔹 Procrastination vs Productivity

- Higher procrastination frequency leads to
  - Lower productivity satisfaction
  - Increased dissatisfaction

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression (Supervised Learning)

**Objective:**  
Predict `productivity_satisfaction`

**Why Logistic Regression?**

- Handles multi-class classification
- Interpretable and stable
- Works well with ordinal outcomes

**Pipeline:**

1. Feature encoding
2. Feature scaling
3. Train-test split (80–20)
4. Model training
5. Performance evaluation

**Performance Insight:**

- Achieves ~40–45% accuracy across 5 classes
- Significantly better than random guessing
- Performance limited by subjective self-reported data

---

### 2️⃣ K-Means Clustering (Unsupervised Learning)

**Objective:**  
Segment users based on behavioral patterns

**Number of Clusters:** `3`

**Cluster Interpretation:**

| Cluster | Description |
|-------|------------|
| 0 | 📱 High Usage – Low Productivity |
| 1 | ⚖️ Balanced Users |
| 2 | 🎯 Disciplined & Productive Users |

**Evaluation Metrics:**

- Silhouette Score
- Elbow Method for optimal K

---

## 🌐 Streamlit Web Application

### Features

- Multi-tab navigation
- Interactive EDA visualizations
- Real-time productivity prediction
- User behavior clustering
- Robust inference-time preprocessing

### Tabs

1. **Home**
   - Project overview
   - Dataset and model summary

2. **EDA Dashboard**
   - Visual exploration of behavioral patterns

3. **Productivity Prediction**
   - User inputs → ML prediction

4. **User Clustering**
   - Assigns user to a behavioral cluster

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Plotly  
- Streamlit  
- Pickle (Model persistence)  

---

## 📁 Project Structure

    ├── app.py
    ├── MLDataset.csv
    ├── logistic_model.pkl
    ├── kmeans_model.pkl
    ├── scaler.pkl
    ├── ordinal_encoder.pkl
    ├── feature_columns.pkl
    ├── requirements.txt
    └── README.md

---

## 🔍 Key Insights

- Excessive social media usage negatively impacts productivity
- Phone usage after bedtime is linked to reduced sleep
- Balanced digital habits correlate with higher satisfaction
- Behavioral clustering provides actionable user insights

---

## 🔮 Future Enhancements

- Add SHAP-based explainability
- Improve class balance with larger datasets
- Introduce personalized recommendations
- Compare with tree-based and ensemble models
- Add longitudinal behavior tracking

---

## 🏁 Conclusion

This project demonstrates a **complete end-to-end data science workflow**:

- Data cleaning and EDA
- Feature engineering
- Supervised and unsupervised ML
- Real-world deployment using Streamlit

It successfully integrates **data analysis, machine learning, and application development** into a practical and scalable solution.
