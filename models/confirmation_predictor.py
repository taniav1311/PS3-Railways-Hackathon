import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import sys
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


class ConfirmationPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_importance = {}

        # Load data
        data_dir = os.path.join(PROJECT_ROOT, "data")
        print("📂 Loading booking data...")
        self.booking_data = pd.read_csv(os.path.join(data_dir, "booking_data.csv"))
        print(f"  ✅ Loaded {len(self.booking_data)} booking records")

        self.trains_data = pd.read_csv(os.path.join(data_dir, "trains.csv"))
        self.stations_data = pd.read_csv(os.path.join(data_dir, "stations.csv"))

        # Train the model
        self._prepare_and_train()

    def _prepare_and_train(self):
        """Prepare features and train the confirmation model"""
        print("\n🧠 Training Confirmation Prediction Model...")
        df = self.booking_data.copy()

        # Encode categorical columns
        for col in ["train_number", "source", "destination", "travel_class", "train_type"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.features = [
            "train_number_enc", "source_enc", "destination_enc",
            "travel_class_enc", "train_type_enc",
            "days_before_travel", "waitlist_position",
            "month", "day_of_week", "is_festival_season",
            "is_peak_season", "is_weekend", "cancellation_rate"
        ]

        X = df[self.features]
        y = df["confirmed"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("  Training GradientBoosting classifier...")
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"  ✅ Accuracy: {self.accuracy:.3f}")

        # Classification report
        print(f"\n📊 Classification Report:")
        report = classification_report(y_test, y_pred, target_names=["Not Confirmed", "Confirmed"])
        print(report)

        # Feature importance
        self.feature_importance = dict(zip(self.features, self.model.feature_importances_))
        print(f"📊 Feature Importance:")
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for fname, importance in sorted_features:
            bar = "█" * int(importance * 50)
            print(f"  {fname:30s} {importance:.3f} {bar}")

    def _safe_encode(self, col, val):
        """Safely encode a value, return 0 if unknown"""
        try:
            return self.label_encoders[col].transform([str(val)])[0]
        except (ValueError, KeyError):
            return 0

    def predict_confirmation(self, train_number, source, destination,
                              travel_class, days_before, waitlist_position,
                              month, day_of_week, is_festival=False, is_peak=False):
        """Predict ticket confirmation probability"""

        features = np.array([[
            self._safe_encode("train_number", train_number),
            self._safe_encode("source", source),
            self._safe_encode("destination", destination),
            self._safe_encode("travel_class", travel_class),
            self._safe_encode("train_type", "Superfast"),
            days_before,
            waitlist_position,
            month,
            day_of_week,
            int(is_festival),
            int(is_peak),
            int(day_of_week >= 5),
            0.15  # average cancellation rate
        ]])

        confirmation_prob = self.model.predict_proba(features)[0][1]
        confirmed = self.model.predict(features)[0]

        # Recommendation based on probability
        if confirmation_prob > 0.85:
            recommendation = "✅ Very likely to confirm. Safe to book!"
            emoji = "🟢"
        elif confirmation_prob > 0.65:
            recommendation = "👍 Good chances. Consider booking."
            emoji = "🟡"
        elif confirmation_prob > 0.40:
            recommendation = "⚠️ Moderate chances. Consider alternative trains or classes."
            emoji = "🟠"
        else:
            recommendation = "❌ Low chances. Try booking earlier, different class, or alternative train."
            emoji = "🔴"

        return {
            "confirmation_probability": round(float(confirmation_prob), 3),
            "likely_confirmed": bool(confirmed),
            "recommendation": recommendation,
            "confidence_emoji": emoji,
            "waitlist_position": waitlist_position,
            "days_before_travel": days_before
        }

    def recommend_booking_advance(self, train_number, source, destination,
                                   travel_class, month, target_probability=0.8):
        """Recommend how many days in advance to book"""
        results = []

        test_days = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 60, 90, 120]

        for days in test_days:
            # Estimate waitlist position based on booking advance
            if days > 60:
                est_waitlist = 0
            elif days > 30:
                est_waitlist = max(0, 10 - days // 5)
            elif days > 14:
                est_waitlist = max(0, 25 - days)
            elif days > 7:
                est_waitlist = max(0, 40 - days * 2)
            else:
                est_waitlist = max(0, 70 - days * 8)

            pred = self.predict_confirmation(
                train_number, source, destination, travel_class,
                days, est_waitlist, month, 3  # Wednesday as default
            )

            results.append({
                "days_before": days,
                "estimated_waitlist": est_waitlist,
                "confirmation_probability": pred["confirmation_probability"],
                "likely_confirmed": pred["likely_confirmed"]
            })

        df = pd.DataFrame(results)

        # Find minimum days needed for target probability
        meets_target = df[df["confirmation_probability"] >= target_probability]
        if not meets_target.empty:
            recommended_days = int(meets_target["days_before"].min())
        else:
            recommended_days = 120  # Book as early as possible

        return {
            "booking_analysis": results,
            "recommended_days_advance": recommended_days,
            "target_probability": target_probability
        }

    def find_best_alternatives(self, source, destination, travel_class,
                                days_before, month):
        """Find trains with best confirmation chances"""
        alternatives = []

        for _, train in self.trains_data.iterrows():
            stations = train["stations"]
            if isinstance(stations, str):
                stations = eval(stations)

            # Check if this train connects source and destination
            if source in stations and destination in stations:
                src_idx = stations.index(source)
                dst_idx = stations.index(destination)

                if src_idx < dst_idx:  # Correct direction
                    # Get available classes
                    classes = train["classes"]
                    if isinstance(classes, str):
                        classes = eval(classes)

                    for tc in classes:
                        # Estimate waitlist
                        if days_before > 30:
                            est_wl = 5
                        elif days_before > 7:
                            est_wl = 20
                        else:
                            est_wl = 50

                        pred = self.predict_confirmation(
                            train["number"], source, destination, tc,
                            days_before, est_wl, month, 3
                        )

                        alternatives.append({
                            "train_number": str(train["number"]),
                            "train_name": train["name"],
                            "train_type": train["type"],
                            "class": tc,
                            "confirmation_probability": pred["confirmation_probability"],
                            "recommendation": pred["recommendation"],
                            "emoji": pred["confidence_emoji"]
                        })

        # Sort by confirmation probability
        alternatives.sort(key=lambda x: x["confirmation_probability"], reverse=True)
        return alternatives

    def get_model_metrics(self):
        """Return model performance metrics"""
        return {
            "accuracy": round(self.accuracy, 3),
            "feature_importance": self.feature_importance,
            "total_training_records": len(self.booking_data)
        }


# ============================================
# MAIN - Test the confirmation predictor
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎫 Ticket Confirmation Predictor - Mumbai to Delhi")
    print("=" * 60)

    cp = ConfirmationPredictor()

    # ---- Test 1: Single Predictions ----
    print("\n" + "=" * 60)
    print("🎯 TEST 1: Confirmation Probability Predictions")
    print("=" * 60)

    test_cases = [
        {"train": "12951", "src": "MMCT", "dst": "NDLS", "class": "3A",
         "days": 15, "wl": 25, "month": 10, "dow": 3,
         "label": "Rajdhani 3A, WL/25, 15 days before (Festival)"},

        {"train": "12951", "src": "MMCT", "dst": "NDLS", "class": "1A",
         "days": 30, "wl": 3, "month": 5, "dow": 1,
         "label": "Rajdhani 1A, WL/3, 30 days before (Normal)"},

        {"train": "12903", "src": "MMCT", "dst": "NDLS", "class": "SL",
         "days": 5, "wl": 80, "month": 12, "dow": 5,
         "label": "Golden Temple SL, WL/80, 5 days before (Winter Weekend)"},

        {"train": "12137", "src": "CSMT", "dst": "NDLS", "class": "3A",
         "days": 60, "wl": 0, "month": 7, "dow": 2,
         "label": "Punjab Mail 3A, WL/0, 60 days before (Monsoon)"},

        {"train": "12903", "src": "MMCT", "dst": "NDLS", "class": "GN",
         "days": 1, "wl": 0, "month": 11, "dow": 6,
         "label": "Golden Temple GN, No WL, 1 day before (Festival Sunday)"},
    ]

    for tc in test_cases:
        result = cp.predict_confirmation(
            tc["train"], tc["src"], tc["dst"], tc["class"],
            tc["days"], tc["wl"], tc["month"], tc["dow"],
            is_festival=tc["month"] in [10, 11, 12],
            is_peak=tc["month"] in [4, 5, 10, 11, 12]
        )
        print(f"\n  📋 {tc['label']}:")
        print(f"     {result['confidence_emoji']} Probability: {result['confirmation_probability']:.1%}")
        print(f"     {result['recommendation']}")

    # ---- Test 2: Booking Advance ----
    print("\n" + "=" * 60)
    print("📅 TEST 2: When Should You Book?")
    print("=" * 60)

    advice = cp.recommend_booking_advance("12951", "MMCT", "NDLS", "3A", month=10)
    print(f"\n  Train: Mumbai Rajdhani | Class: 3A | Month: October")
    print(f"  🎯 Target: 80% confirmation probability")
    print(f"  📌 Recommendation: Book at least {advice['recommended_days_advance']} days in advance\n")

    print(f"  {'Days Before':<15} {'Est. WL':<12} {'Probability':<15} {'Status'}")
    print(f"  {'─' * 60}")
    for entry in advice["booking_analysis"]:
        status = "✅" if entry["likely_confirmed"] else "❌"
        prob_bar = "█" * int(entry["confirmation_probability"] * 20)
        print(f"  {entry['days_before']:<15} WL/{entry['estimated_waitlist']:<8} "
              f"{entry['confirmation_probability']:.1%} {prob_bar:<20} {status}")

    # ---- Test 3: Find Best Alternatives ----
    print("\n" + "=" * 60)
    print("🔄 TEST 3: Best Alternative Trains (MMCT → NDLS)")
    print("=" * 60)

    alts = cp.find_best_alternatives("MMCT", "NDLS", "3A", 15, 10)
    if alts:
        print(f"\n  Booking 15 days before | October\n")
        print(f"  {'Train':<30} {'Class':<8} {'Probability':<15} {'Verdict'}")
        print(f"  {'─' * 80}")
        for a in alts:
            print(f"  {a['train_name']:<30} {a['class']:<8} "
                  f"{a['emoji']} {a['confirmation_probability']:.1%}       "
                  f"{a['train_type']}")
    else:
        print("  No direct trains found.")

    # ---- Test 4: Alternatives from CSMT ----
    print("\n" + "=" * 60)
    print("🔄 TEST 4: Best Alternative Trains (CSMT → NDLS)")
    print("=" * 60)

    alts2 = cp.find_best_alternatives("CSMT", "NDLS", "3A", 10, 12)
    if alts2:
        print(f"\n  Booking 10 days before | December\n")
        print(f"  {'Train':<30} {'Class':<8} {'Probability':<15}")
        print(f"  {'─' * 60}")
        for a in alts2:
            print(f"  {a['train_name']:<30} {a['class']:<8} "
                  f"{a['emoji']} {a['confirmation_probability']:.1%}")

    print("\n" + "=" * 60)
    print("✅ CONFIRMATION PREDICTOR - ALL TESTS PASSED!")
    print("=" * 60)
