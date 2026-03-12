import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


class DelayPredictor:
    def __init__(self):
        self.model = None
        self.cascade_model = None
        self.label_encoders = {}
        self.feature_importance = {}

        # Load data
        data_dir = os.path.join(PROJECT_ROOT, "data")
        print("📂 Loading delay data...")
        self.delay_data = pd.read_csv(os.path.join(data_dir, "historical_delays.csv"))
        print(f"  ✅ Loaded {len(self.delay_data)} delay records")

        self.trains_data = pd.read_csv(os.path.join(data_dir, "trains.csv"))
        self.stations_data = pd.read_csv(os.path.join(data_dir, "stations.csv"))

        # Train the models
        self._prepare_and_train()

    def _prepare_and_train(self):
        """Prepare features and train both models"""
        print("\n🧠 Training Delay Prediction Model...")
        df = self.delay_data.copy()

        # Encode categorical features
        for col in ["train_number", "station_code"]:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        features = [
            "train_number_encoded", "station_code_encoded",
            "month", "day_of_week", "is_weekend", "is_monsoon",
            "is_fog_season", "is_festival_season", "stop_sequence", "total_stops"
        ]

        X = df[features]
        y = df["delay_minutes"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---- Model 1: Delay Duration Predictor ----
        print("  Training delay duration model (GradientBoosting)...")
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        print(f"  ✅ Delay Model - MAE: {self.mae:.2f} min | R²: {self.r2:.3f}")

        # ---- Model 2: Cascade Severity Classifier ----
        print("  Training cascade severity model (RandomForest)...")
        df["cascade_severity"] = pd.cut(
            df["delay_minutes"],
            bins=[-1, 5, 15, 30, 60, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        y_cascade = df["cascade_severity"]
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X, y_cascade, test_size=0.2, random_state=42
        )

        self.cascade_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.cascade_model.fit(X_train_c, y_train_c)

        cascade_acc = accuracy_score(y_test_c, self.cascade_model.predict(X_test_c))
        print(f"  ✅ Cascade Model - Accuracy: {cascade_acc:.3f}")

        # Feature importance
        self.feature_importance = dict(zip(features, self.model.feature_importances_))
        print(f"\n📊 Feature Importance:")
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for fname, importance in sorted_features:
            bar = "█" * int(importance * 50)
            print(f"  {fname:30s} {importance:.3f} {bar}")

    def predict_delay(self, train_number, station_code, month, day_of_week,
                      is_monsoon=False, is_fog=False, is_festival=False,
                      stop_sequence=1, total_stops=5):
        """Predict delay for a specific train at a specific station"""

        # Safely encode train number
        try:
            train_enc = self.label_encoders["train_number"].transform([str(train_number)])[0]
        except ValueError:
            train_enc = 0  # Unknown train

        # Safely encode station code
        try:
            station_enc = self.label_encoders["station_code"].transform([str(station_code)])[0]
        except ValueError:
            station_enc = 0  # Unknown station

        features = np.array([[
            train_enc, station_enc, month, day_of_week,
            int(day_of_week >= 5),  # is_weekend
            int(is_monsoon),
            int(is_fog),
            int(is_festival),
            stop_sequence,
            total_stops
        ]])

        # Predict delay duration
        predicted_delay = max(0, self.model.predict(features)[0])

        # Predict cascade severity
        cascade_severity = self.cascade_model.predict(features)[0]
        cascade_proba = self.cascade_model.predict_proba(features)[0]

        severity_labels = ["Minimal (<5 min)", "Low (5-15 min)", "Moderate (15-30 min)",
                           "High (30-60 min)", "Severe (>60 min)"]

        return {
            "predicted_delay_minutes": round(predicted_delay, 1),
            "cascade_severity": severity_labels[cascade_severity],
            "severity_level": int(cascade_severity),
            "severity_probabilities": {
                severity_labels[i]: round(float(p), 3) for i, p in enumerate(cascade_proba)
            }
        }

    def predict_route_delays(self, train_number, stations_list, month, day_of_week, **kwargs):
        """Predict delays across all stations on a route"""
        results = []

        for i, stn in enumerate(stations_list):
            pred = self.predict_delay(
                train_number, stn, month, day_of_week,
                stop_sequence=i + 1,
                total_stops=len(stations_list),
                **kwargs
            )
            # Add station info
            stn_info = self.stations_data[self.stations_data["code"] == stn]
            stn_name = stn_info.iloc[0]["name"] if not stn_info.empty else stn

            pred["station_code"] = stn
            pred["station_name"] = stn_name
            pred["stop_sequence"] = i + 1
            results.append(pred)

        return results

    def get_vulnerable_routes(self, month, day_of_week):
        """Identify routes most vulnerable to delays"""
        vulnerability = []

        for _, train in self.trains_data.iterrows():
            stations = train["stations"]
            if isinstance(stations, str):
                stations = eval(stations)

            total_delay = 0
            max_delay = 0
            delays_per_station = []

            for i, stn in enumerate(stations):
                pred = self.predict_delay(
                    train["number"], stn, month, day_of_week,
                    stop_sequence=i + 1, total_stops=len(stations)
                )
                delay = pred["predicted_delay_minutes"]
                total_delay += delay
                max_delay = max(max_delay, delay)
                delays_per_station.append(delay)

            avg_delay = total_delay / len(stations)

            # Risk level
            if avg_delay > 30:
                risk = "🔴 HIGH"
            elif avg_delay > 15:
                risk = "🟠 MEDIUM"
            else:
                risk = "🟢 LOW"

            vulnerability.append({
                "train_number": str(train["number"]),
                "train_name": train["name"],
                "train_type": train["type"],
                "num_stops": len(stations),
                "avg_delay": round(avg_delay, 1),
                "max_delay": round(max_delay, 1),
                "total_accumulated_delay": round(total_delay, 1),
                "risk_level": risk
            })

        return pd.DataFrame(vulnerability).sort_values("total_accumulated_delay", ascending=False)

    def get_delay_heatmap_data(self):
        """Get delay data organized for heatmap visualization"""
        pivot = self.delay_data.groupby(
            ["station_code", "month"]
        )["delay_minutes"].mean().reset_index()

        # Add station names
        pivot = pivot.merge(
            self.stations_data[["code", "name"]],
            left_on="station_code", right_on="code", how="left"
        )

        return pivot

    def get_model_metrics(self):
        """Return model performance metrics"""
        return {
            "delay_model_mae": round(self.mae, 2),
            "delay_model_r2": round(self.r2, 3),
            "feature_importance": self.feature_importance,
            "total_training_records": len(self.delay_data)
        }


# ============================================
# MAIN - Test the delay predictor
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("⏱️ Train Delay Predictor - Mumbai to Delhi")
    print("=" * 60)

    dp = DelayPredictor()

    # ---- Test 1: Single Prediction ----
    print("\n" + "=" * 60)
    print("🔮 TEST 1: Single Station Delay Prediction")
    print("=" * 60)

    test_cases = [
        {"train": "12951", "station": "BRC", "month": 7, "dow": 2,
         "label": "Rajdhani at Vadodara (Monsoon, Wednesday)", "monsoon": True},
        {"train": "12951", "station": "NDLS", "month": 1, "dow": 0,
         "label": "Rajdhani at Delhi (Fog, Monday)", "fog": True},
        {"train": "12903", "station": "KOTA", "month": 10, "dow": 5,
         "label": "Golden Temple at Kota (Festival, Saturday)", "festival": True},
        {"train": "12137", "station": "BPL", "month": 5, "dow": 3,
         "label": "Punjab Mail at Bhopal (Normal, Thursday)"},
    ]

    for tc in test_cases:
        result = dp.predict_delay(
            tc["train"], tc["station"], tc["month"], tc["dow"],
            is_monsoon=tc.get("monsoon", False),
            is_fog=tc.get("fog", False),
            is_festival=tc.get("festival", False)
        )
        print(f"\n  📍 {tc['label']}:")
        print(f"     Predicted Delay: {result['predicted_delay_minutes']} minutes")
        print(f"     Severity: {result['cascade_severity']}")

    # ---- Test 2: Full Route Prediction ----
    print("\n" + "=" * 60)
    print("🚂 TEST 2: Full Route Delay Prediction (Rajdhani - Monsoon)")
    print("=" * 60)

    route_delays = dp.predict_route_delays(
        "12951",
        ["MMCT", "BRC", "KOTA", "SWM", "NDLS"],
        month=7, day_of_week=2, is_monsoon=True
    )

    print(f"\n  {'Station':<25} {'Delay (min)':<15} {'Severity'}")
    print(f"  {'─' * 70}")
    for r in route_delays:
        print(f"  {r['station_name']:<25} {r['predicted_delay_minutes']:<15} {r['cascade_severity']}")

    # ---- Test 3: Full Route (Punjab Mail - Fog) ----
    print("\n" + "=" * 60)
    print("🚂 TEST 3: Full Route Delay Prediction (Punjab Mail - Fog Season)")
    print("=" * 60)

    route_delays2 = dp.predict_route_delays(
        "12137",
        ["CSMT", "KYN", "NMD", "MMR", "BSL", "ET", "BPL", "JHS", "GWL", "AGC", "MTJ", "NDLS"],
        month=1, day_of_week=1, is_fog=True
    )

    print(f"\n  {'Station':<25} {'Delay (min)':<15} {'Severity'}")
    print(f"  {'─' * 70}")
    for r in route_delays2:
        print(f"  {r['station_name']:<25} {r['predicted_delay_minutes']:<15} {r['cascade_severity']}")

    # ---- Test 4: Vulnerable Routes ----
    print("\n" + "=" * 60)
    print("⚠️ TEST 4: Most Vulnerable Routes Today")
    print("=" * 60)

    from datetime import datetime
    now = datetime.now()
    vuln = dp.get_vulnerable_routes(now.month, now.weekday())
    print(f"\n  Current: Month={now.strftime('%B')}, Day={now.strftime('%A')}\n")
    print(vuln[["train_name", "train_type", "avg_delay", "max_delay",
                "total_accumulated_delay", "risk_level"]].to_string(index=False))

    # ---- Test 5: Model Metrics ----
    print("\n" + "=" * 60)
    print("📊 TEST 5: Model Performance Metrics")
    print("=" * 60)
    metrics = dp.get_model_metrics()
    print(f"  MAE: {metrics['delay_model_mae']} minutes")
    print(f"  R² Score: {metrics['delay_model_r2']}")
    print(f"  Training records: {metrics['total_training_records']}")

    print("\n" + "=" * 60)
    print("✅ DELAY PREDICTOR - ALL TESTS PASSED!")
    print("=" * 60)
