# 📊 Unified Mentor Internship Projects

This repository contains two beginner-level Data Science and Machine Learning projects completed as part of the **Unified Mentor Internship Program**:

1. 🌸 Iris Flower Classification  
2. 🏅 Olympics Data Analysis (1976–2008)

---

## 🌸 1. Iris Flower Classification

### 📌 Objective
To classify iris flowers into one of three species based on the length and width of their petals and sepals using machine learning.

### 📂 Dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- Features: Sepal Length, Sepal Width, Petal Length, Petal Width
- Target: Species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`)

### 🛠 Tools & Libraries
- Python, Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

### 🔍 Visualizations
- Pairplot (Seaborn)
- Correlation heatmap
- Count plots and scatter plots by species

### 🤖 ML Algorithms Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### ✅ Result
All models showed high accuracy. SVM and Random Forest gave the best results with >95% accuracy.

---

## 🏅 2. Olympics Data Analysis (1976–2008)

### 📌 Objective
To analyze Summer Olympics medal data and build a predictive model for medal-winning based on country, sport, and gender.

### 📂 Dataset
- Source: [Google Drive Link](https://drive.google.com/file/d/1EHMliUCEb8k6VhkpxK00oaY6GQtkwrhg/view?usp=sharing)
- Data includes all medal winners from 1976 to 2008.
- Columns: Year, City, Sport, Discipline, Event, Athlete, Gender, Country, Medal

### 🛠 Tools & Libraries
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

### 📊 EDA Highlights
- Top cities that hosted Olympics
- Events and disciplines by frequency
- Top medal-winning athletes
- Gender distribution
- Medal trends over time

### 🤖 Machine Learning
- Logistic Regression model to predict whether an athlete would win a medal.
- Synthetic class `0` created for non-winners to handle class imbalance.

### ✅ Result
- Model trained with acceptable accuracy.
- Clear trends observed: USA and Russia dominated; gender bias in events noted.

---

## 📁 Project Structure

Unified_Projects/
│
├── IRIS_CLASSIFICATION_UNIFIED.ipynb # Iris notebook
├── Olympics_Data_Analysis_Project.ipynb # Olympics notebook
├── iris.csv # Iris dataset (if needed)
├── olympics_data.csv # Olympics dataset
├── README.md # Combined project summary
└── requirements.txt # Python packages list

yaml
Copy
Edit

---

## 🚀 Future Improvements

### For Iris Project:
- Apply GridSearchCV for hyperparameter tuning
- Deploy as a web app using Streamlit

### For Olympics Project:
- Add Random Forest or Decision Trees
- Analyze by athlete nationality and gender more deeply
- Build interactive dashboards using Plotly

---

## 🙌 Credits

- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- Public Olympics data from Kaggle & Google Drive
- Guided by Unified Mentor Internship Program

---

## 📜 License
This repository is for educational purposes only.
