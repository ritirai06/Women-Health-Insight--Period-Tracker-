# 🩺 Women Health Insight System
## AI-Powered Period Tracker & Menstrual Health Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [PDF Report Features](#pdf-report-features)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🌟 Overview

**Women Health Insight System** is an AI-powered web application designed to help women track their menstrual cycles, predict period delays, and receive personalized health recommendations. The system uses machine learning algorithms to analyze lifestyle factors and provide actionable insights for better reproductive health management.

### 🎯 Mission
To empower women with data-driven insights about their menstrual health through accessible, non-diagnostic AI technology.

### ⚠️ Important Note
This is an **educational and awareness-based system**. It is NOT a medical diagnostic tool and should NOT replace professional medical advice.

---

## ✨ Key Features

### 🔮 **AI-Powered Predictions**
- **Cycle Delay Prediction**: Machine learning model predicts potential menstrual delays
- **Risk Level Assessment**: Categorizes health risk (Low, Medium, High, Critical)
- **Wellness Score**: Comprehensive health score (0-100) based on multiple factors

### 📊 **Comprehensive Health Tracking**
- Period cycle length and duration monitoring
- Symptom tracking (cramps, mood changes, flow levels)
- Lifestyle factors analysis (sleep, exercise, stress, nutrition)
- BMI and physical health metrics
- Medical condition tracking (PCOS, Endometriosis, Thyroid disorders)

### 📈 **Visual Analytics**
- Interactive Streamlit dashboard with dark theme
- Real-time data visualization (line charts, bar charts, area charts)
- Cycle history tracking with trend analysis
- Health metrics comparison over time

### 📄 **Professional PDF Reports with Visual Charts**
- **🎯 Wellness Gauge Chart** - Semi-circular gauge showing overall health score (0-100)
- **📊 Lifestyle Radar Chart** - Pentagon chart for sleep, exercise, nutrition, hydration, and stress
- **📈 Health Metrics Bar Chart** - Comparison of your values vs optimal ranges
- Detailed health analysis and clinical interpretations
- Personalized recommendations with priority levels
- Professional medical-grade formatting
- Downloadable and shareable reports

### 💡 **Personalized Recommendations**
- Evidence-based health advice
- Priority-based action steps (Critical, High, Moderate, Low)
- Lifestyle modification suggestions
- Stress management techniques
- Nutrition and exercise guidance

### 🗄️ **Data Management**
- SQLite database for secure patient data storage
- Patient history tracking and retrieval
- Export reports to PDF format
- Admin authentication system

---

## 🛠️ Technology Stack

### **Frontend**
- **Streamlit** - Interactive web application framework
- **Custom CSS** - Dark theme with glassmorphism effects
- **Plotly/Matplotlib** - Data visualization and charting

### **Backend**
- **Python 3.10+** - Core programming language
- **SQLite** - Lightweight database for data persistence
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations

### **Machine Learning**
- **scikit-learn** - Machine learning algorithms
- **Random Forest Regressor** - Delay prediction model
- **Feature Engineering** - One-hot encoding, normalization

### **PDF Generation**
- **ReportLab** - PDF document creation with charts
- **Matplotlib** - Chart generation (gauge, radar, bar charts)
- **PIL/ImageReader** - Image handling for BytesIO objects

---

## 📥 Installation

### **Prerequisites**
- Python 3.10 or higher
- pip (Python package manager)

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/ritirai06/Women-Health-Insight--Period-Tracker-.git
cd "Women Health Insight"
```

### **Step 2: Install Dependencies**
```bash
pip install streamlit pandas numpy scikit-learn matplotlib reportlab Pillow
```

### **Step 3: Verify Installation**
```bash
python3 -c "import streamlit; import pandas; import sklearn; print('✅ All dependencies installed!')"
```

---

## 🚀 Usage

### **Running the Application**

1. **Navigate to the app directory:**
```bash
cd "Women Health Insight/app"
```

2. **Start the Streamlit server:**
```bash
streamlit run app.py
```

3. **Access the application:**
   - Open your browser and go to: `http://localhost:8501`
   - Default credentials:
     - **Username:** `admin`
     - **Password:** `admin123`

### **Using the Dashboard**

#### **📝 Health Input Form**
1. Enter patient information (Name, Age, Patient ID)
2. Fill in menstrual cycle details
3. Provide lifestyle information
4. Add symptoms and medical conditions
5. Click "🔮 Predict Cycle Delay" button

#### **📊 View Results**
- **Prediction Summary**: Cycle delay prediction and risk level
- **Wellness Score**: Overall health assessment (0-100)
- **Interactive Charts**: Visual representation of health metrics
- **Recommendations**: Personalized health advice

#### **📄 Generate Report**
1. After prediction, click "📄 Generate PDF Report"
2. Report automatically downloads with:
   - Patient information summary
   - **3 Visual Charts** (Wellness Gauge, Lifestyle Radar, Health Metrics Bar)
   - Prediction details
   - Detailed health analysis
   - Personalized recommendations
   - Clinical notes and disclaimers

#### **🗄️ Patient History**
- View past predictions and reports
- Track health trends over time
- Access previous PDF reports

---

## 📁 Project Structure

```
Women Health Insight/
│
├── app/
│   ├── app.py                 # Main Streamlit application
│   ├── db.py                  # Database operations (SQLite)
│   ├── report.py              # PDF report generation with charts
│   ├── recommendations.py     # AI recommendation engine
│   └── assests/              # Static files (images, icons)
│
├── model/
│   ├── delay_predictor.py    # ML model training script
│   └── feature_names.joblib  # Saved feature names
│
├── data/
│   ├── women_health_dataset.csv  # Training dataset
│   └── db_data/
│       ├── patient_history.csv   # Patient records
│       └── reports/              # Generated PDF reports
│
├── notebooks/
│   └── create_datasets.py    # Data generation scripts
│
├── analysis/
│   └── eda_analysis.py       # Exploratory Data Analysis
│
├── README.md                 # Project documentation
└── LICENSE                   # License file
```

---

## 📊 Model Performance

### **Dataset Features**
9 key health indicators used for predictions:
- `cycle_length` - Length of menstrual cycle (days)
- `period_duration` - Duration of period (days)
- `sleep_hours` - Average sleep per night
- `flow_level` - Menstrual flow intensity
- `stress_level` - Stress intensity
- `physical_activity` - Exercise frequency
- `pain_level` - Menstrual cramp severity
- `mood_changes` - Emotional state variations
- `user_id` - Patient identifier

### **Model Algorithm**
- **Random Forest Regressor** (`n_estimators=100`)
- Ensemble of 100 decision trees
- Handles non-linear relationships
- Robust to overfitting

### **Evaluation Metrics**
```
Mean Absolute Error (MAE):  2.18 days
Mean Squared Error (MSE):   6.36
Root Mean Squared Error:    2.52 days
```

✅ **Interpretation:** The model predicts cycle delays with an average error of ~2.5 days, which is acceptable for educational purposes.

### **Feature Importance**
Top factors influencing predictions:

| Feature | Importance | Insight |
|---------|-----------|---------|
| `sleep_hours` | 30.7% | **Most influential** - Sleep directly affects hormonal balance |
| `cycle_length` | 30.3% | **Critical indicator** - Natural cycle variability |
| `period_duration` | 16.4% | **Secondary factor** - Cycle irregularity marker |
| `flow_level` | 5-8% | **Minor impact** - Symptom indicator |
| `stress_level` | 4.5% | **Tertiary factor** - Cortisol affects hormones |

🧠 **AI Insight:** The model correctly identifies sleep and cycle length as primary predictors, aligning with real-world medical research.

---

## 📄 PDF Report Features

### **🎨 Visual Charts (Enhanced Feature)**

#### **1. 🎯 Wellness Gauge Chart**
- **Design**: Semi-circular gauge with modern blue color scheme
- **Color Coding**:
  - 🟢 **Green (75-100):** Excellent health
  - 🟠 **Orange (50-74):** Fair health  
  - 🔴 **Red (0-49):** Needs attention
- **Display**: Large score (e.g., 87/100) with status label
- **Purpose**: Quick visual assessment of overall health

#### **2. 📊 Lifestyle Radar Chart**
- **Design**: Pentagon/spider chart with blue filled area
- **Factors Tracked**:
  - 😴 **Sleep Quality** - Based on hours (optimal: 7-9 hrs)
  - 💪 **Physical Activity** - Exercise frequency scoring
  - 🥗 **Nutrition** - Diet quality assessment
  - 💧 **Hydration** - Water intake (optimal: 8 glasses)
  - 🧘 **Stress Management** - Stress level impact
- **Scale**: 0-100 for each factor
- **Purpose**: Holistic view of lifestyle balance

#### **3. 📈 Health Metrics Bar Chart**
- **Design**: Side-by-side comparison bars
- **Your Values** (Blue bars) vs **Optimal Ranges** (Green bars)
- **Metrics Compared**:
  - Cycle Length (days)
  - Period Duration (days)
  - Sleep Hours (hrs/night)
  - Water Intake (glasses/day)
  - Predicted Delay (days)
- **Value Labels**: Numerical display on each bar
- **Purpose**: Identify areas needing improvement

### **📋 Report Sections**

1. **Header & Patient Information**
   - Professional modern blue theme (#3B82F6)
   - Patient details (Name, ID, Age)
   - Generation timestamp
   - Confidentiality notice

2. **Prediction Summary**
   - Color-coded urgency indicators:
     - 🚨 **URGENT** (>60 days delay) - Red
     - ⚠️ **HIGH PRIORITY** (45-60 days) - Orange
     - 🟡 **ATTENTION NEEDED** (15-45 days) - Yellow
     - 🟢 **NORMAL RANGE** (<15 days) - Green
   - Predicted delay in days
   - Risk level assessment
   - Wellness score with visual gauge

3. **Key Health Metrics**
   - Two-column layout for efficient space use
   - All input parameters displayed
   - Medical conditions highlighted in red

4. **Clinical Interpretation**
   - Severity-based medical explanations
   - Context-specific guidance:
     - Severe irregularity (>60 days)
     - Significant delay (45-60 days)
     - Moderate irregularity (15-45 days)
     - Normal range (<15 days)

5. **Detailed Health Analysis (Page 2)**
   - **Lifestyle Factors** with color-coded assessments:
     - Sleep: Critical/Insufficient/Optimal/Excessive
     - Stress: High/Moderate/Low
     - Exercise: Sedentary/Light/Moderate/Very Active
     - Nutrition & Hydration: Combined assessment
   - **Lifestyle Radar Chart** for visual overview
   - **Physical Health Metrics**:
     - BMI calculation and category
     - Weight and height
     - Health impact analysis
   - **Health Metrics Bar Chart** for comparisons

6. **Symptoms & Quality of Life**
   - Cramp severity (0-10 scale) with interpretation
   - Mood state tracking
   - Current symptoms list

7. **Personalized Recommendations (Page 3)**
   - Priority-based ordering:
     - 🚨 **CRITICAL** - Immediate action required
     - ⚠️ **HIGH** - Important to address soon
     - 🟡 **MODERATE** - Should be considered
     - 🟢 **LOW** - General wellness tips
   - Each recommendation includes:
     - Category (e.g., Sleep, Stress, Exercise)
     - Title (brief description)
     - Why? (Medical reasoning)
     - Action Steps (Practical, actionable advice)

8. **Clinical Notes & Disclaimer**
   - Additional healthcare provider observations
   - Important legal disclaimer
   - Emergency guidance
   - Professional consultation reminder

---

## 🔮 Future Enhancements

### **Phase 1: Advanced Analytics**
- [ ] Time-series forecasting for next 3 cycles
- [ ] Anomaly detection for irregular patterns
- [ ] Fertility window predictions
- [ ] Ovulation tracking

### **Phase 2: Enhanced Features**
- [ ] Mobile app (React Native/Flutter)
- [ ] Medication reminder system
- [ ] Export data to CSV/Excel
- [ ] Multi-language support
- [ ] Dark/Light theme toggle

### **Phase 3: AI Improvements**
- [ ] Deep learning models (LSTM, GRU)
- [ ] Transfer learning from medical datasets
- [ ] Real-time model retraining
- [ ] Personalized model per user
- [ ] Natural Language Processing for symptom input

### **Phase 4: Integration**
- [ ] Wearable device integration (Fitbit, Apple Watch)
- [ ] Healthcare provider portal
- [ ] Telemedicine consultation booking
- [ ] API for third-party apps
- [ ] Cloud storage (AWS/Azure)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/AmazingFeature`
3. **Commit your changes:** `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch:** `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### **Areas for Contribution**
- 🐛 Bug fixes and testing
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- 🧠 ML model improvements
- 🌐 Internationalization (i18n)
- ♿ Accessibility features

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **scikit-learn** team for machine learning tools
- **Streamlit** for the amazing web framework
- **ReportLab** for PDF generation capabilities
- **Matplotlib** for data visualization
- Women's health research community for medical insights

---

## ⚠️ Disclaimer

**Important Medical Notice:**

This application is designed for **educational and informational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

- ❌ Do NOT use this system for self-diagnosis
- ❌ Do NOT make medical decisions based solely on app predictions
- ✅ Always consult with qualified healthcare providers
- ✅ Seek immediate medical attention for severe symptoms
- ✅ Use this tool to track and understand your health patterns

**Emergency Situations:**
If you experience severe pain, heavy bleeding, or symptoms of medical emergency, seek immediate professional medical care.

---

## 🌟 Star This Project

If you find this project helpful, please give it a ⭐ on GitHub!

---

**Made with ❤️ for Women's Health**  
*Empowering women through AI and technology*

---

**Last Updated:** January 28, 2026  
**Version:** 2.0.0 (Enhanced with Visual Charts in PDF Reports)
