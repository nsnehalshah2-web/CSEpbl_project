# CardioVanguard Pro
This is a cardiac risk assessment project developed for CSE PBL. It helps predict the risk of heart disease by looking at both medical data and lifestyle factors like stress, diet, and smoking.

## Developer Information
- **Name:** Nehal Shah
- **Registration Number:** 2427030423
- **Guide:** Dr. Susheela Vishnoi

## What this project does
The system uses a Random Forest model trained on the standard UCI Heart Disease dataset (accuracy around 83%). It takes 13 clinical parameters (like blood pressure, cholesterol, etc.) and adjusts the final score based on lifestyle factors like exercise habits, family history, and stress levels to give a more complete picture of heart health.

## How to run it

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Model
Run the training script to generate the model file (`model.pkl`):
```bash
python train_model.py
```

### 3. Start the Application
Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Files
- `app.py`: The main application dashboard built with Streamlit.
- `predictor.py`: The core logic that calculates the risk scores.
- `train_model.py`: Script to train the data and save the model bundle.
- `heart.csv`: The dataset used for training.
- `model.pkl`: The saved model and metadata bundle (generated after training).

*Note: This tool is for educational purposes only and is not a substitute for professional medical advice.*
