import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


class CongestionAnalyzer:
    def __init__(self):
        data_dir = os.path.join(PROJECT_ROOT, "data")
        self.passenger_flow = pd.read_csv(os.path.join(data_dir, "passenger_flow.csv"))
        self.stations = pd.read_csv(os.path.join(data_dir, "stations.csv"))
        self._compute_occupancy()

    def _compute_occupancy(self):
        """Compute normalized occupancy that produces realistic percentages"""

        # Step 1: Average passenger counts across all days per station per hour
        self.avg_flow = self.passenger_flow.groupby(
            ["station_code", "station_name", "hour"]
        )["passenger_count"].mean().reset_index()

        # Step 2: Within-station normalization (each station's peak = 1.0)
        station_peaks = self.avg_flow.groupby("station_code")["passenger_count"].max()
        self.avg_flow["within_norm"] = self.avg_flow.apply(
            lambda r: r["passenger_count"] / station_peaks.get(r["station_code"], 1), axis=1
        )

        # Step 3: Station importance factor based on footfall (0.25 to 1.0)
        max_footfall = self.stations["daily_footfall"].max()
        min_footfall = self.stations["daily_footfall"].min()
        footfall_range = max(max_footfall - min_footfall, 1)

        importance_map = {}
        for _, stn in self.stations.iterrows():
            importance = 0.25 + 0.75 * (stn["daily_footfall"] - min_footfall) / footfall_range
            importance_map[stn["code"]] = round(importance, 4)

        self.avg_flow["importance"] = self.avg_flow["station_code"].map(importance_map)

        # Step 4: Final occupancy = within_norm * importance
        # Big stations at peak ~ 90-95%, small stations at peak ~ 25-35%
        self.avg_flow["occupancy_pct"] = (
            self.avg_flow["within_norm"] * self.avg_flow["importance"]
        ).clip(0.02, 0.95)

        # Step 5: Classify congestion levels
        self.avg_flow["congestion_level"] = self.avg_flow["occupancy_pct"].apply(
            lambda x: "Critical" if x > 0.82 else "High" if x > 0.60 else "Moderate" if x > 0.30 else "Low"
        )

        # Round for display
        self.avg_flow["occupancy_pct"] = self.avg_flow["occupancy_pct"].round(3)
        self.avg_flow["passenger_count"] = self.avg_flow["passenger_count"].astype(int)

    def get_current_congestion(self, hour=None):
        """Get congestion levels for all stations at given hour"""
        if hour is None:
            hour = datetime.now().hour

        current = self.avg_flow[self.avg_flow["hour"] == hour].copy()
        current = current.sort_values("occupancy_pct", ascending=False)

        return current[["station_code", "station_name", "passenger_count",
                         "occupancy_pct", "congestion_level"]].reset_index(drop=True)

    def get_station_pattern(self, station_code):
        """Get 24-hour congestion pattern for a station"""
        stn_data = self.avg_flow[
            self.avg_flow["station_code"] == station_code
        ][["hour", "passenger_count", "occupancy_pct"]].copy()

        stn_data = stn_data.sort_values("hour").reset_index(drop=True)
        return stn_data

    def predict_peak_hours(self, station_code):
        """Predict peak congestion hours for a station"""
        pattern = self.get_station_pattern(station_code)
        if pattern.empty:
            return {"peak_hours": [], "off_peak_hours": [],
                    "avg_occupancy": 0, "max_occupancy": 0, "min_occupancy": 0}

        stn_info = self.stations[self.stations["code"] == station_code]
        stn_name = stn_info.iloc[0]["name"] if not stn_info.empty else station_code

        peak_hours = pattern.nlargest(5, "occupancy_pct")
        off_peak = pattern.nsmallest(5, "occupancy_pct")

        return {
            "station_code": station_code,
            "station_name": stn_name,
            "peak_hours": peak_hours[["hour", "occupancy_pct", "passenger_count"]].to_dict("records"),
            "off_peak_hours": off_peak[["hour", "occupancy_pct", "passenger_count"]].to_dict("records"),
            "avg_occupancy": round(float(pattern["occupancy_pct"].mean()), 3),
            "max_occupancy": round(float(pattern["occupancy_pct"].max()), 3),
            "min_occupancy": round(float(pattern["occupancy_pct"].min()), 3)
        }

    def get_hotspots(self, hour=None, threshold=0.60):
        """Identify congestion hotspots above threshold"""
        congestion = self.get_current_congestion(hour)
        return congestion[congestion["occupancy_pct"] >= threshold].copy()

    def get_corridor_congestion(self):
        """Analyze congestion along both Mumbai-Delhi corridors"""
        western = ["MMCT", "BVI", "ST", "BRC", "RTM", "KOTA", "SWM", "MTJ", "NDLS"]
        central = ["CSMT", "KYN", "NMD", "MMR", "BSL", "ET", "BPL", "JHS", "GWL", "AGC", "NDLS"]

        results = {}
        for name, corridor in [("Western", western), ("Central", central)]:
            corridor_data = self.avg_flow[
                self.avg_flow["station_code"].isin(corridor)
            ].groupby(["station_code", "station_name"]).agg({
                "occupancy_pct": "mean",
                "passenger_count": "mean",
                "congestion_level": lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
            }).reset_index()

            corridor_data["order"] = corridor_data["station_code"].apply(
                lambda x: corridor.index(x) if x in corridor else 99
            )
            corridor_data = corridor_data.sort_values("order").drop(columns=["order"])
            corridor_data["occupancy_pct"] = corridor_data["occupancy_pct"].round(3)
            corridor_data["passenger_count"] = corridor_data["passenger_count"].astype(int)

            results[name] = corridor_data.reset_index(drop=True)

        return results

    def compare_stations(self, station_codes):
        """Compare congestion across multiple stations"""
        comparison = []
        for code in station_codes:
            stn_info = self.stations[self.stations["code"] == code]
            if stn_info.empty:
                continue
            peaks = self.predict_peak_hours(code)
            comparison.append({
                "station_code": code,
                "station_name": stn_info.iloc[0]["name"],
                "type": stn_info.iloc[0]["type"],
                "platforms": stn_info.iloc[0]["platforms"],
                "daily_footfall": stn_info.iloc[0]["daily_footfall"],
                "avg_occupancy": peaks["avg_occupancy"],
                "max_occupancy": peaks["max_occupancy"],
                "peak_hour": int(peaks["peak_hours"][0]["hour"]) if peaks["peak_hours"] else None,
                "best_hour": int(peaks["off_peak_hours"][0]["hour"]) if peaks["off_peak_hours"] else None
            })
        return pd.DataFrame(comparison).sort_values("avg_occupancy", ascending=False)

    def get_congestion_summary(self):
        """Get overall congestion summary"""
        avg_by_station = self.avg_flow.groupby("station_code").agg({
            "occupancy_pct": "mean",
            "passenger_count": "sum"
        }).reset_index()
        avg_by_station = avg_by_station.merge(
            self.stations[["code", "name", "type"]],
            left_on="station_code", right_on="code", how="left"
        )
        most = avg_by_station.nlargest(5, "occupancy_pct")
        least = avg_by_station.nsmallest(5, "occupancy_pct")
        return {
            "most_congested": most[["name", "type", "occupancy_pct"]].to_dict("records"),
            "least_congested": least[["name", "type", "occupancy_pct"]].to_dict("records"),
            "avg_network_occupancy": round(float(avg_by_station["occupancy_pct"].mean()), 3),
            "total_passengers_30days": int(avg_by_station["passenger_count"].sum())
        }


if __name__ == "__main__":
    ca = CongestionAnalyzer()
    now = datetime.now()

    print("=" * 60)
    print("Station Congestion Analyzer")
    print("=" * 60)

    print(f"\nCongestion at {now.hour}:00:")
    current = ca.get_current_congestion(now.hour)
    for _, row in current.iterrows():
        print(f"  {row['station_name']:<25} {row['occupancy_pct']:.1%}  {row['congestion_level']}")

    print(f"\nPeak hours for New Delhi:")
    peaks = ca.predict_peak_hours("NDLS")
    for p in peaks["peak_hours"][:3]:
        print(f"  {int(p['hour'])}:00 — {p['occupancy_pct']:.1%}")

    print(f"\nHotspots at 18:00:")
    hotspots = ca.get_hotspots(18)
    for _, h in hotspots.iterrows():
        print(f"  {h['station_name']}: {h['occupancy_pct']:.1%}")
