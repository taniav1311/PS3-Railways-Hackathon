import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ============================================
# REAL Mumbai-Delhi Stations
# ============================================

stations_data = [
    # Western Route
    {"code": "MMCT", "name": "Mumbai Central", "lat": 18.9712, "lon": 72.8197, "zone": "WR", "type": "terminus", "daily_footfall": 150000, "platforms": 8},
    {"code": "BVI", "name": "Borivali", "lat": 19.2288, "lon": 72.8567, "zone": "WR", "type": "junction", "daily_footfall": 80000, "platforms": 6},
    {"code": "BL", "name": "Valsad", "lat": 20.6100, "lon": 72.9264, "zone": "WR", "type": "regular", "daily_footfall": 15000, "platforms": 3},
    {"code": "ST", "name": "Surat", "lat": 21.2063, "lon": 72.8411, "zone": "WR", "type": "major", "daily_footfall": 95000, "platforms": 6},
    {"code": "BRC", "name": "Vadodara Junction", "lat": 22.3104, "lon": 73.1812, "zone": "WR", "type": "junction", "daily_footfall": 85000, "platforms": 7},
    {"code": "RTM", "name": "Ratlam Junction", "lat": 23.3260, "lon": 75.0367, "zone": "WR", "type": "junction", "daily_footfall": 35000, "platforms": 5},
    {"code": "KOTA", "name": "Kota Junction", "lat": 25.1800, "lon": 75.8648, "zone": "WCR", "type": "junction", "daily_footfall": 50000, "platforms": 6},
    {"code": "SWM", "name": "Sawai Madhopur", "lat": 26.0173, "lon": 76.3466, "zone": "WCR", "type": "regular", "daily_footfall": 20000, "platforms": 4},
    {"code": "MTJ", "name": "Mathura Junction", "lat": 27.4924, "lon": 77.6737, "zone": "NCR", "type": "junction", "daily_footfall": 55000, "platforms": 6},
    {"code": "NDLS", "name": "New Delhi", "lat": 28.6442, "lon": 77.2217, "zone": "NR", "type": "terminus", "daily_footfall": 450000, "platforms": 16},

    # Central Route stations
    {"code": "CSMT", "name": "Mumbai CSMT", "lat": 18.9402, "lon": 72.8356, "zone": "CR", "type": "terminus", "daily_footfall": 350000, "platforms": 18},
    {"code": "KYN", "name": "Kalyan Junction", "lat": 19.2437, "lon": 73.1355, "zone": "CR", "type": "junction", "daily_footfall": 100000, "platforms": 8},
    {"code": "NMD", "name": "Nashik Road", "lat": 19.9907, "lon": 73.7620, "zone": "CR", "type": "regular", "daily_footfall": 40000, "platforms": 4},
    {"code": "MMR", "name": "Manmad Junction", "lat": 20.2532, "lon": 74.4370, "zone": "CR", "type": "junction", "daily_footfall": 30000, "platforms": 5},
    {"code": "BSL", "name": "Bhusawal Junction", "lat": 21.0446, "lon": 75.7714, "zone": "CR", "type": "junction", "daily_footfall": 45000, "platforms": 6},
    {"code": "ET", "name": "Itarsi Junction", "lat": 22.6158, "lon": 77.7647, "zone": "WCR", "type": "junction", "daily_footfall": 40000, "platforms": 7},
    {"code": "BPL", "name": "Bhopal Junction", "lat": 23.2689, "lon": 77.4124, "zone": "WCR", "type": "major", "daily_footfall": 70000, "platforms": 6},
    {"code": "JHS", "name": "Jhansi Junction", "lat": 25.4358, "lon": 78.5685, "zone": "NCR", "type": "junction", "daily_footfall": 55000, "platforms": 8},
    {"code": "GWL", "name": "Gwalior Junction", "lat": 26.2183, "lon": 78.1828, "zone": "NCR", "type": "major", "daily_footfall": 45000, "platforms": 5},
    {"code": "AGC", "name": "Agra Cantt", "lat": 27.1631, "lon": 78.0081, "zone": "NCR", "type": "major", "daily_footfall": 60000, "platforms": 6},
]

# ============================================
# REAL Trains on Mumbai-Delhi Route
# ============================================

trains_data = [
    {"number": "12951", "name": "Mumbai Rajdhani", "type": "Rajdhani", "route": "western",
     "stations": ["MMCT", "BRC", "KOTA", "SWM", "NDLS"],
     "departure": "16:35", "arrival": "08:35", "duration_hrs": 16, "frequency": "daily",
     "total_seats": 952, "classes": ["1A", "2A", "3A"]},

    {"number": "12953", "name": "August Kranti Rajdhani", "type": "Rajdhani", "route": "western",
     "stations": ["MMCT", "ST", "BRC", "RTM", "KOTA", "NDLS"],
     "departure": "17:40", "arrival": "10:55", "duration_hrs": 17.25, "frequency": "daily",
     "total_seats": 905, "classes": ["1A", "2A", "3A"]},

    {"number": "12267", "name": "Mumbai Duronto", "type": "Duronto", "route": "western",
     "stations": ["MMCT", "NDLS"],
     "departure": "23:15", "arrival": "16:25", "duration_hrs": 17.17, "frequency": "daily",
     "total_seats": 816, "classes": ["1A", "2A", "3A", "SL"]},

    {"number": "12903", "name": "Golden Temple Mail", "type": "Superfast", "route": "western",
     "stations": ["MMCT", "BVI", "BL", "ST", "BRC", "RTM", "KOTA", "SWM", "MTJ", "NDLS"],
     "departure": "21:30", "arrival": "06:55", "duration_hrs": 33.42, "frequency": "daily",
     "total_seats": 1500, "classes": ["2A", "3A", "SL", "GN"]},

    {"number": "12925", "name": "Paschim Express", "type": "Superfast", "route": "western",
     "stations": ["MMCT", "BVI", "ST", "BRC", "RTM", "KOTA", "SWM", "MTJ", "NDLS"],
     "departure": "11:30", "arrival": "06:10", "duration_hrs": 42.67, "frequency": "daily",
     "total_seats": 1800, "classes": ["2A", "3A", "SL", "GN"]},

    {"number": "12909", "name": "Garib Rath", "type": "Garib Rath", "route": "western",
     "stations": ["MMCT", "BVI", "ST", "BRC", "KOTA", "NDLS"],
     "departure": "15:45", "arrival": "07:15", "duration_hrs": 15.5, "frequency": "weekly",
     "total_seats": 1200, "classes": ["3A"]},

    {"number": "12137", "name": "Punjab Mail", "type": "Mail", "route": "central",
     "stations": ["CSMT", "KYN", "NMD", "MMR", "BSL", "ET", "BPL", "JHS", "GWL", "AGC", "MTJ", "NDLS"],
     "departure": "19:10", "arrival": "06:05", "duration_hrs": 34.92, "frequency": "daily",
     "total_seats": 1600, "classes": ["1A", "2A", "3A", "SL", "GN"]},

    {"number": "12621", "name": "Tamil Nadu Express", "type": "Superfast", "route": "central",
     "stations": ["CSMT", "KYN", "MMR", "BSL", "ET", "BPL", "JHS", "AGC", "NDLS"],
     "departure": "22:00", "arrival": "07:30", "duration_hrs": 33.5, "frequency": "daily",
     "total_seats": 1400, "classes": ["2A", "3A", "SL", "GN"]},

    {"number": "12261", "name": "Mumbai Duronto Central", "type": "Duronto", "route": "central",
     "stations": ["CSMT", "NDLS"],
     "departure": "23:05", "arrival": "16:00", "duration_hrs": 16.92, "frequency": "3days",
     "total_seats": 780, "classes": ["1A", "2A", "3A"]},

    {"number": "12187", "name": "Mumbai Garib Rath", "type": "Garib Rath", "route": "central",
     "stations": ["CSMT", "KYN", "BSL", "BPL", "JHS", "GWL", "NDLS"],
     "departure": "13:55", "arrival": "05:20", "duration_hrs": 15.42, "frequency": "weekly",
     "total_seats": 1100, "classes": ["3A"]},
]


def generate_schedule_data():
    """Generate detailed station-wise schedule for each train"""
    print("  → Generating schedules...")
    schedules = []

    for train in trains_data:
        dep_time = datetime.strptime(train["departure"], "%H:%M")
        total_minutes = train["duration_hrs"] * 60
        n_stations = len(train["stations"])

        for i, stn in enumerate(train["stations"]):
            if i == 0:
                arrival = None
                departure = dep_time
                day = 1
            elif i == n_stations - 1:
                minutes_elapsed = total_minutes
                arrival = dep_time + timedelta(minutes=minutes_elapsed)
                departure = None
                day = 1 + int(minutes_elapsed // 1440)
            else:
                minutes_elapsed = total_minutes * (i / (n_stations - 1))
                halt = random.choice([2, 3, 5, 10, 15])
                arrival = dep_time + timedelta(minutes=minutes_elapsed)
                departure = arrival + timedelta(minutes=halt)
                day = 1 + int(minutes_elapsed // 1440)

            schedules.append({
                "train_number": train["number"],
                "train_name": train["name"],
                "station_code": stn,
                "stop_sequence": i + 1,
                "scheduled_arrival": arrival.strftime("%H:%M") if arrival else None,
                "scheduled_departure": departure.strftime("%H:%M") if departure else None,
                "day": day,
                "distance_km": int((i / (n_stations - 1)) * random.randint(1300, 1500)) if n_stations > 1 else 0
            })

    return pd.DataFrame(schedules)


def generate_delay_data(n_days=90):
    """Generate historical delay data for past 90 days"""
    print(f"  → Generating delay data for {n_days} days...")
    delays = []

    for day_offset in range(n_days):
        date = datetime.now() - timedelta(days=day_offset)

        month = date.month
        is_festival = month in [10, 11, 12, 3, 4]
        is_monsoon = month in [6, 7, 8, 9]
        is_fog = month in [12, 1]
        day_of_week = date.weekday()
        is_weekend = day_of_week >= 5

        for train in trains_data:
            base_delay = np.random.exponential(10)

            if is_monsoon:
                base_delay *= np.random.uniform(1.5, 3.0)
            if is_fog:
                base_delay *= np.random.uniform(1.3, 2.5)
            if is_festival:
                base_delay *= np.random.uniform(1.2, 1.8)
            if train["type"] == "Rajdhani":
                base_delay *= 0.5
            if train["type"] in ["Mail", "Superfast"]:
                base_delay *= 1.3

            cumulative_delay = 0
            for i, stn in enumerate(train["stations"]):
                if i == 0:
                    station_delay = max(0, np.random.normal(base_delay * 0.3, 5))
                else:
                    added_delay = max(-5, np.random.normal(base_delay * 0.15, 8))
                    stn_info = next((s for s in stations_data if s["code"] == stn), None)
                    if stn_info and stn_info["type"] in ["junction", "terminus"]:
                        added_delay += np.random.exponential(5)
                    cumulative_delay = max(0, cumulative_delay + added_delay)
                    station_delay = cumulative_delay

                delays.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "train_number": train["number"],
                    "train_name": train["name"],
                    "station_code": stn,
                    "delay_minutes": round(station_delay, 1),
                    "month": month,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "is_monsoon": is_monsoon,
                    "is_fog_season": is_fog,
                    "is_festival_season": is_festival,
                    "stop_sequence": i + 1,
                    "total_stops": len(train["stations"])
                })

    return pd.DataFrame(delays)


def generate_booking_data(n_records=10000):
    """Generate ticket booking and confirmation data"""
    print(f"  → Generating {n_records} booking records...")
    bookings = []

    for _ in range(n_records):
        train = random.choice(trains_data)
        stations = train["stations"]

        src_idx = random.randint(0, len(stations) - 2)
        dst_idx = random.randint(src_idx + 1, len(stations) - 1)

        travel_class = random.choice(train["classes"])
        days_before = random.randint(0, 120)

        travel_date = datetime.now() + timedelta(days=random.randint(-60, 30))
        booking_date = travel_date - timedelta(days=days_before)

        month = travel_date.month
        is_festival = month in [10, 11, 12, 3, 4]
        is_peak = month in [4, 5, 6, 10, 11, 12]
        day_of_week = travel_date.weekday()

        if days_before > 60:
            waitlist = 0
        elif days_before > 30:
            waitlist = random.choices([0, random.randint(1, 30)], weights=[0.7, 0.3])[0]
        elif days_before > 7:
            waitlist = random.choices([0, random.randint(1, 80)], weights=[0.4, 0.6])[0]
        else:
            waitlist = random.choices([0, random.randint(10, 150)], weights=[0.2, 0.8])[0]

        if is_festival:
            waitlist = int(waitlist * 1.5)

        if waitlist == 0:
            confirmed = True
        elif travel_class == "1A":
            confirmed = waitlist <= 5
        elif travel_class == "2A":
            confirmed = waitlist <= 15
        elif travel_class == "3A":
            confirmed = waitlist <= 40
        elif travel_class == "SL":
            confirmed = waitlist <= 80
        else:
            confirmed = True

        if not confirmed and random.random() < 0.2:
            confirmed = True
        if confirmed and random.random() < 0.05:
            confirmed = False

        bookings.append({
            "train_number": train["number"],
            "train_name": train["name"],
            "source": stations[src_idx],
            "destination": stations[dst_idx],
            "travel_class": travel_class,
            "booking_date": booking_date.strftime("%Y-%m-%d"),
            "travel_date": travel_date.strftime("%Y-%m-%d"),
            "days_before_travel": days_before,
            "waitlist_position": waitlist,
            "month": month,
            "day_of_week": day_of_week,
            "is_festival_season": is_festival,
            "is_peak_season": is_peak,
            "is_weekend": day_of_week >= 5,
            "train_type": train["type"],
            "confirmed": confirmed,
            "total_seats_class": random.randint(50, 400),
            "cancellation_rate": round(random.uniform(0.05, 0.35), 3)
        })

    return pd.DataFrame(bookings)


def generate_passenger_flow(n_days=30):
    """Generate passenger flow / crowd data"""
    print(f"  → Generating passenger flow for {n_days} days...")
    flows = []

    for day_offset in range(n_days):
        date = datetime.now() - timedelta(days=day_offset)

        for station in stations_data:
            for hour in range(24):
                base_crowd = station["daily_footfall"] / 24

                if hour in [7, 8, 9, 17, 18, 19]:
                    multiplier = random.uniform(2.0, 3.5)
                elif hour in [10, 11, 15, 16, 20, 21]:
                    multiplier = random.uniform(1.2, 1.8)
                elif hour in [0, 1, 2, 3, 4]:
                    multiplier = random.uniform(0.1, 0.3)
                else:
                    multiplier = random.uniform(0.8, 1.2)

                crowd = int(base_crowd * multiplier * random.uniform(0.8, 1.2))

                capacity = station["platforms"] * 500
                utilization = crowd / capacity

                if utilization > 0.9:
                    level = "Critical"
                elif utilization > 0.7:
                    level = "High"
                elif utilization > 0.4:
                    level = "Moderate"
                else:
                    level = "Low"

                flows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "station_code": station["code"],
                    "station_name": station["name"],
                    "hour": hour,
                    "passenger_count": crowd,
                    "platform_capacity": capacity,
                    "utilization": round(utilization, 3),
                    "crowd_level": level,
                    "day_of_week": date.weekday()
                })

    return pd.DataFrame(flows)


# ============================================
# MAIN - Run this to generate all data
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("🚂 Railway Data Generator - Mumbai to Delhi")
    print("=" * 50)

    # Make sure data folder exists
    os.makedirs("data", exist_ok=True)

    print("\n[1/6] Saving stations data...")
    stations_df = pd.DataFrame(stations_data)
    stations_df.to_csv("data/stations.csv", index=False)
    print(f"  ✅ {len(stations_df)} stations saved")

    print("\n[2/6] Saving trains data...")
    trains_df = pd.DataFrame(trains_data)
    trains_df.to_csv("data/trains.csv", index=False)
    print(f"  ✅ {len(trains_df)} trains saved")

    print("\n[3/6] Generating schedule data...")
    schedule_df = generate_schedule_data()
    schedule_df.to_csv("data/schedules.csv", index=False)
    print(f"  ✅ {len(schedule_df)} schedule entries saved")

    print("\n[4/6] Generating historical delay data (90 days)...")
    delay_df = generate_delay_data(90)
    delay_df.to_csv("data/historical_delays.csv", index=False)
    print(f"  ✅ {len(delay_df)} delay records saved")

    print("\n[5/6] Generating booking data...")
    booking_df = generate_booking_data(10000)
    booking_df.to_csv("data/booking_data.csv", index=False)
    print(f"  ✅ {len(booking_df)} booking records saved")

    print("\n[6/6] Generating passenger flow data...")
    flow_df = generate_passenger_flow(30)
    flow_df.to_csv("data/passenger_flow.csv", index=False)
    print(f"  ✅ {len(flow_df)} flow records saved")

    print("\n" + "=" * 50)
    print("✅ ALL DATA GENERATED SUCCESSFULLY!")
    print("=" * 50)
    print(f"\nFiles saved in 'data/' folder:")
    print(f"  📄 stations.csv       ({len(stations_df)} rows)")
    print(f"  📄 trains.csv         ({len(trains_df)} rows)")
    print(f"  📄 schedules.csv      ({len(schedule_df)} rows)")
    print(f"  📄 historical_delays.csv ({len(delay_df)} rows)")
    print(f"  📄 booking_data.csv   ({len(booking_df)} rows)")
    print(f"  📄 passenger_flow.csv ({len(flow_df)} rows)")
    print("\n🚀 You can now proceed to Phase 2!")