# 🎓 Prediction of Students at Risk of Academic Failure
### Early Warning System

## 📌 Project Overview
Education is a foundational pillar of socio-economic development. This project implements a **proactive, AI-driven Early Warning System** to predict student academic failure. By leveraging early behavioral indicators (e.g., attendance rate, forum participation, assignment submissions) and academic history (GPA), institutions can shift from reactive remedial models to proactive support.

The project systematically compares traditional **Machine Learning** algorithms with **Deep Learning** architectures (Artificial Neural Networks) specifically designed for structured tabular educational data.

## 🚀 Key Features
- **Exploratory Data Analysis (EDA):** Insightful analysis of 2,000 student records across 11 features.
- **Advanced Preprocessing:** 
  - Standard scaling and optimal categorical encoding (ordinal/binary).
- **Handling Class Imbalance:**
  - Evaluated the 71.15% (Safe) / 28.85% (At-Risk) split.
  - Employed **SMOTE** (Synthetic Minority Over-sampling Technique) for Machine Learning models to prevent majority-class bias.
  - Implemented **Class Weighting (`pos_weight`)** dynamically within PyTorch's `BCEWithLogitsLoss` for Artificial Neural Networks.
- **Algorithm Comparison:**
  - *Machine Learning:* Logistic Regression, Random Forest, Support Vector Machines (SVM).
  - *Deep Learning:* Baseline ANN (3 layers), Deep ANN (5 layers), Regularized ANN (Dropout & Tanh).
- **Evaluation Priority:** System tuned to maximize **Recall (Sensitivity)** alongside Precision and F1-Score, ensuring the system misses as few genuinely struggling students as possible.
- **Interactive Web UI:** Includes a deployed Streamlit dashboard enabling real-time predictions via the best-performing neural network model.

## 📂 Repository Structure
```text
├── academic_risk_prediction.ipynb   # Comprehensive notebook documenting EDA, ML & DL Training, and Model Evaluation
├── app.py                           # Streamlit Web Application Interface
├── export_model.py                  # Script to train and export the final PyTorch architecture
├── requirements.txt                 # Python dependencies
├── risk_students.csv                # Educational dataset (11 features, 2000 records)
├── ann_model.pth                    # Exported PyTorch Model Weights (Generated locally)
├── scaler.pkl                       # Exported Scikit-Learn StandardScaler (Generated locally)
└── README.md                        # Project documentation
```

## 🛠️ Technology Stack
- **Python 3.x**
- **Deep Learning Framework:** PyTorch
- **Machine Learning Library:** Scikit-Learn, Imbalanced-Learn
- **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Web App Dashboard:** Streamlit

## 💻 Local Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ahmedgh10/PS2-Project.git
   cd PS2-Project
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook (Optional):**
   Open `academic_risk_prediction.ipynb` to view the comprehensive model analysis and evaluation metrics.
4. **Export Model Artifacts (if not present):**
   ```bash
   python export_model.py
   ```
5. **Launch the Streamlit Web UI:**
   ```bash
   streamlit run app.py
   ```

## 🌐 Web Deployment
This project is configured to be seamlessly deployed onto **Streamlit Community Cloud**. 
1. Log in to [share.streamlit.io](https://share.streamlit.io/).
2. Create a new app linked to this GitHub repository.
3. Set the main file path to `app.py` and hit **Deploy**.

## ⚖️ Ethical Guidelines
This model operates strictly as an Early Warning System—acting as a digital triage mechanism to highlight behavioral risk markers. It is explicitly designed to be **supportive, not punitive**. Decision-making frameworks driven by this repository should continuously monitor against algorithmic biases and ensure demographic variables do not negatively impact student profiles.