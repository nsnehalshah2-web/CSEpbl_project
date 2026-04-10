import joblib
import pandas as pd
import os

class CardiacPredictor:
    def __init__(self, model_bundle_path="model.pkl"):
        if not os.path.exists(model_bundle_path):
            raise FileNotFoundError(f"Model bundle '{model_bundle_path}' not found. Run train_model.py first.")
        
        bundle = joblib.load(model_bundle_path)
        
        self.model = bundle['model']
        self.metadata = bundle['metadata']
        self.feature_names = self.metadata['feature_names_in']
        self.categorical_features = self.metadata['categorical_features']

    def predict_risk(self, clinical_data: dict, holistic_data: dict = None):
        """
        clinical_data: dict matching numeric and categorical features from training.
        holistic_data: optional dict for lifestyle factors.
        """
        df = pd.DataFrame([clinical_data])
        
        df = df.reindex(columns=self.feature_names)
        df[self.categorical_features] = df[self.categorical_features].astype(str)
        
        # model is a Pipeline [preprocessor -> classifier]
        base_prob = self.model.predict_proba(df)[0][1]
        
        multiplier = 1.0
        details = []

        if holistic_data:
            fam_hist = holistic_data.get("family_history", [])
            if "Heart Disease" in fam_hist:
                multiplier += 0.15
                details.append("Genetic predisposition (+15%)")
            
            smoking = holistic_data.get("smoking", "Never")
            if smoking == "Current":
                multiplier += 0.25
                details.append("Active smoking risk (+25%)")
            elif smoking == "Past":
                multiplier += 0.10
                details.append("Historical smoking impact (+10%)")
            
            stress = holistic_data.get("stress", "Low")
            if stress == "Extreme":
                multiplier += 0.15
                details.append("Critical stress levels (+15%)")
            elif stress == "High":
                multiplier += 0.10
                details.append("Elevated stress (+10%)")
            
            if holistic_data.get("anxiety", False):
                multiplier += 0.05
                details.append("Chronic anxiety factor (+5%)")

            pollution = holistic_data.get("pollution", "Clean")
            if pollution == "Heavy Industrial":
                multiplier += 0.08
                details.append("Poor air quality exposure (+8%)")

            sleep = holistic_data.get("sleep", 7)
            if sleep < 6:
                multiplier += 0.10
                details.append("Sleep deprivation impact (+10%)")
            elif sleep > 9:
                 multiplier += 0.02

            exercise = holistic_data.get("exercise", "None")
            if exercise == "Athlete":
                multiplier -= 0.15
                details.append("Athletic conditioning (-15%)")
            elif exercise == "Frequent":
                multiplier -= 0.10
                details.append("Regular exercise benefit (-10%)")

            diet = holistic_data.get("diet", "Standard")
            if diet in ["Mediterranean", "Vegan"]:
                multiplier -= 0.05
                details.append("Heart-healthy diet (-5%)")

        final_risk = min(max(base_prob * multiplier, 0.0), 1.0)
        
        age = clinical_data.get('age', 45)
        age_shift = 0
        if final_risk > 0.7: age_shift = 12
        elif final_risk > 0.4: age_shift = 5
        elif final_risk < 0.2: age_shift = -3
        
        return {
            "base_probability": float(base_prob),
            "final_risk_score": float(final_risk),
            "holistic_multiplier": float(multiplier),
            "adjustments": details,
            "heart_age": int(age + age_shift),
            "age_gap": int(age_shift),
            "status": "High Risk" if final_risk > 0.5 else "Stable"
        }

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = CardiacPredictor()
    return _predictor
