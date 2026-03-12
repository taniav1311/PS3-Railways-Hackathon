import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime
import sys
import os
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="RailIntel - Railway Intelligence",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #FF6B35;
        text-align: center;
        padding: 0.5rem 0;
        letter-spacing: 1px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .train-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FF6B35;
        text-align: center;
        padding: 0.5rem 0;
        border-bottom: 2px solid #FF6B35;
        margin-bottom: 1rem;
    }
    .stMetric > div {
        background-color: #0e1117;
        border-radius: 8px;
        padding: 8px;
        border-left: 3px solid #FF6B35;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    d = os.path.join(PROJECT_ROOT, "data")
    return (pd.read_csv(os.path.join(d, "stations.csv")),
            pd.read_csv(os.path.join(d, "trains.csv")),
            pd.read_csv(os.path.join(d, "schedules.csv")),
            pd.read_csv(os.path.join(d, "historical_delays.csv")),
            pd.read_csv(os.path.join(d, "booking_data.csv")),
            pd.read_csv(os.path.join(d, "passenger_flow.csv")))


@st.cache_resource
def load_models():
    from graph.railway_graph import RailwayNetwork
    from models.delay_predictor import DelayPredictor
    from models.confirmation_predictor import ConfirmationPredictor
    from models.congestion_model import CongestionAnalyzer
    return (RailwayNetwork(), DelayPredictor(),
            ConfirmationPredictor(), CongestionAnalyzer())


stations, trains, schedules, delays, bookings, passenger_flow = load_data()
network, delay_pred, conf_pred, congestion_analyzer = load_models()
now = datetime.now()


def stn_name(code):
    s = stations[stations["code"] == code]
    return s.iloc[0]["name"] if not s.empty else code


# ============================================
# SIDEBAR — Navigation + Conditions Only
# ============================================
st.sidebar.markdown("## RailMitra")
st.sidebar.markdown("*Railway Intelligence Platform*")
st.sidebar.markdown("---")

st.sidebar.markdown("### Navigation")
mode = st.sidebar.radio(
    "Select Section:",
    ["Train Overview", "Delay Analysis", "Ticket Intelligence",
     "Station Congestion", "Route Map", "Network Resilience", "AI Assistant"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Current Conditions")

if now.month in [6, 7, 8, 9]:
    season_str = "Monsoon"
elif now.month in [12, 1]:
    season_str = "Fog Season"
elif now.month in [10, 11, 3, 4]:
    season_str = "Festival Season"
else:
    season_str = "Normal"


st.sidebar.metric("Date", now.strftime("%d %b %Y"))
st.sidebar.metric("Time", now.strftime("%H:%M"))

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")
st.sidebar.success("Delay Predictor — Active")
st.sidebar.success("Confirmation Model — Active")
st.sidebar.success("Congestion Analyzer — Active")
st.sidebar.success("Network Graph — Active")




# ============================================
# MAIN HEADER + TRAIN SELECTOR
# ============================================
st.markdown('<p class="main-header">RailMitra — AI Railway Intelligence Platform</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Mumbai — Delhi Corridor  |  Real-Time Analytics</p>',
            unsafe_allow_html=True)

train_display_list = [f"{t['number']} — {t['name']} ({t['type']})" for _, t in trains.iterrows()]
selected_train_display = st.selectbox("**Select Train**", train_display_list, index=0)

selected_train_number = selected_train_display.split(" — ")[0].strip()
selected_train_row = trains[trains["number"].astype(str) == str(selected_train_number)].iloc[0]
selected_train_name = selected_train_row["name"]
selected_train_type = selected_train_row["type"]
selected_train_route = selected_train_row.get("route", "unknown")

selected_train_stations = selected_train_row["stations"]
if isinstance(selected_train_stations, str):
    selected_train_stations = eval(selected_train_stations)

selected_train_classes = selected_train_row["classes"]
if isinstance(selected_train_classes, str):
    selected_train_classes = eval(selected_train_classes)

st.markdown("---")

# Filter data for selected train
train_delays = delays[delays["train_number"].astype(str) == str(selected_train_number)].copy()
train_bookings = bookings[bookings["train_number"].astype(str) == str(selected_train_number)].copy()
train_schedules = schedules[schedules["train_number"].astype(str) == str(selected_train_number)].copy()
train_station_flow = passenger_flow[passenger_flow["station_code"].isin(selected_train_stations)].copy()


# ============================================
# TAB 1: TRAIN OVERVIEW
# ============================================
if mode == "Train Overview":
    st.markdown(f'<p class="train-header">{selected_train_name} ({selected_train_number})</p>',
                unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    train_avg_delay = train_delays["delay_minutes"].mean() if not train_delays.empty else 0
    train_ontime = (len(train_delays[train_delays["delay_minutes"] < 5]) / max(len(train_delays), 1)) * 100
    train_conf = train_bookings["confirmed"].mean() * 100 if not train_bookings.empty else 0

    col1.metric("Avg Delay", f"{train_avg_delay:.1f} min",
                delta=f"{train_avg_delay - 15:.1f} vs target", delta_color="inverse")
    col2.metric("On-Time Rate", f"{train_ontime:.1f}%")
    col3.metric("Total Stops", f"{len(selected_train_stations)}")
    col4.metric("Route", selected_train_route.title())
    col5.metric("Confirmation Rate", f"{train_conf:.1f}%")

    st.markdown("---")

    # Details + Route
    detail_col, route_col = st.columns([1, 1])

    with detail_col:
        st.subheader("Train Details")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | **Train Name** | {selected_train_name} |
        | **Train Number** | {selected_train_number} |
        | **Type** | {selected_train_type} |
        | **Route** | {selected_train_route.title()} Route |
        | **Departure** | {selected_train_row['departure']} |
        | **Arrival** | {selected_train_row['arrival']} |
        | **Duration** | {selected_train_row['duration_hrs']} hours |
        | **Stops** | {len(selected_train_stations)} stations |
        | **Classes** | {', '.join(selected_train_classes)} |
        | **Frequency** | {selected_train_row['frequency'].title()} |
        | **Total Seats** | {selected_train_row['total_seats']} |
        | **Avg Delay** | {train_avg_delay:.1f} min |
        | **Confirmation Rate** | {train_conf:.1f}% |
        """)

    with route_col:
        st.subheader("Route")
        for i, sc in enumerate(selected_train_stations):
            name = stn_name(sc)
            sd = train_delays[train_delays["station_code"] == sc]["delay_minutes"].mean()
            sd = sd if not np.isnan(sd) else 0

            if i == 0:
                tag, marker = "ORIGIN", "[START]"
            elif i == len(selected_train_stations) - 1:
                tag, marker = "DESTINATION", "[END]"
            else:
                stn_info = stations[stations["code"] == sc]
                stn_type = stn_info.iloc[0]["type"] if not stn_info.empty else "regular"
                tag = stn_type.upper()
                marker = "[STOP]"

            delay_tag = "OK" if sd < 10 else "MODERATE" if sd < 20 else "HIGH"
            if i > 0:
                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;|")
            st.markdown(f"{marker} **{name}** (`{sc}`) — *{tag}* — Delay: {sd:.1f} min [{delay_tag}]")

    st.markdown("---")

    # Schedule
    st.subheader("Schedule")
    if not train_schedules.empty:
        ds = train_schedules.copy()
        ds["Station Name"] = ds["station_code"].apply(stn_name)

        delay_vals = []
        for _, row in ds.iterrows():
            d = train_delays[train_delays["station_code"] == row["station_code"]]["delay_minutes"].mean()
            delay_vals.append(round(d, 1) if not np.isnan(d) else 0)
        ds["Avg Delay (min)"] = delay_vals

        congestion_vals = []
        for _, row in ds.iterrows():
            cong = congestion_analyzer.get_current_congestion(now.hour)
            match = cong[cong["station_code"] == row["station_code"]]
            if not match.empty:
                congestion_vals.append(f"{match.iloc[0]['occupancy_pct']:.0%} ({match.iloc[0]['congestion_level']})")
            else:
                congestion_vals.append("N/A")
        ds["Congestion"] = congestion_vals

        final = ds[["stop_sequence", "Station Name", "station_code",
                     "scheduled_arrival", "scheduled_departure",
                     "day", "distance_km", "Avg Delay (min)", "Congestion"]].copy()
        final.columns = ["Stop", "Station", "Code", "Arrival", "Departure",
                         "Day", "Distance (km)", "Avg Delay (min)", "Congestion"]

        st.dataframe(final, use_container_width=True, hide_index=True,
                     height=min(len(final) * 40 + 50, 500))

    st.markdown("---")
    st.subheader("Quick Statistics")
    q1, q2, q3, q4 = st.columns(4)
    max_d = train_delays["delay_minutes"].max() if not train_delays.empty else 0
    q1.metric("Max Recorded Delay", f"{max_d:.0f} min")
    q2.metric("Delay Records (90 days)", f"{len(train_delays):,}")
    q3.metric("Total Bookings", f"{len(train_bookings):,}")
    avg_wl = train_bookings["waitlist_position"].mean() if not train_bookings.empty else 0
    q4.metric("Avg Waitlist Position", f"WL/{avg_wl:.0f}")


# ============================================
# TAB 2: DELAY ANALYSIS
# ============================================
elif mode == "Delay Analysis":
    st.markdown(f'<p class="train-header">Delay Analysis — {selected_train_name}</p>', unsafe_allow_html=True)

    dt1, dt2, dt3 = st.tabs(["Predict Delays", "Season Comparison", "Historical Patterns"])

    with dt1:
        st.subheader("Predicted Delays for Selected Conditions")
        c1, c2 = st.columns([2, 1])
        with c1:
            sel_month = st.slider("Month", 1, 12, now.month, key="dm")
            sel_dow = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                            "Friday", "Saturday", "Sunday"],
                                    index=now.weekday(), key="dd")
            dow_n = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"].index(sel_dow)
        with c2:
            is_m = st.checkbox("Monsoon Season", sel_month in [6, 7, 8, 9])
            is_f = st.checkbox("Fog Season", sel_month in [12, 1])
            is_fv = st.checkbox("Festival Season", sel_month in [10, 11, 12, 3, 4])

        if st.button("Run Delay Prediction", type="primary", use_container_width=True):
            rd = delay_pred.predict_route_delays(
                selected_train_number, selected_train_stations, sel_month, dow_n,
                is_monsoon=is_m, is_fog=is_f, is_festival=is_fv)
            ddf = pd.DataFrame(rd)

            colors = ddf["predicted_delay_minutes"].apply(
                lambda x: "#4CAF50" if x < 10 else "#FF9800" if x < 30 else "#f44336").tolist()

            fig = go.Figure()
            fig.add_trace(go.Bar(x=ddf["station_name"], y=ddf["predicted_delay_minutes"],
                                 marker_color=colors,
                                 text=ddf["predicted_delay_minutes"].apply(lambda x: f"{x:.0f} min"),
                                 textposition="outside"))
            fig.add_hline(y=15, line_dash="dash", line_color="yellow", annotation_text="Target: 15 min")
            fig.update_layout(title=f"Predicted Delays — {selected_train_name}",
                             xaxis_title="Station", yaxis_title="Delay (min)",
                             height=450, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cascade Severity")
            n_c = min(len(rd), 6)
            cols = st.columns(n_c)
            for i, p in enumerate(rd):
                with cols[i % n_c]:
                    st.metric(p['station_name'][:12], f"{p['predicted_delay_minutes']:.0f} min")
                    st.caption(p["cascade_severity"])

            st.markdown("---")
            total = sum(p["predicted_delay_minutes"] for p in rd)
            mx = max(p["predicted_delay_minutes"] for p in rd)
            avg = total / len(rd)
            risk = "HIGH" if avg > 30 else "MEDIUM" if avg > 15 else "LOW"
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Accumulated Delay", f"{total:.0f} min")
            s2.metric("Maximum at Any Station", f"{mx:.0f} min")
            s3.metric("Risk Level", risk)

    with dt2:
        st.subheader("Delay Cascade — Season Comparison")
        scenarios = {
            "Normal": {"is_monsoon": False, "is_fog": False, "is_festival": False},
            "Monsoon": {"is_monsoon": True, "is_fog": False, "is_festival": False},
            "Fog": {"is_monsoon": False, "is_fog": True, "is_festival": False},
            "Festival": {"is_monsoon": False, "is_fog": False, "is_festival": True},
        }
        cm = {"Normal": "#4CAF50", "Monsoon": "#2196F3", "Fog": "#9E9E9E", "Festival": "#FF9800"}

        fig = go.Figure()
        for sn, kw in scenarios.items():
            preds = delay_pred.predict_route_delays(
                selected_train_number, selected_train_stations, now.month, now.weekday(), **kw)
            fig.add_trace(go.Scatter(
                x=[p["station_name"] for p in preds],
                y=[p["predicted_delay_minutes"] for p in preds],
                mode="lines+markers", name=sn,
                line=dict(color=cm[sn], width=3), marker=dict(size=8)))

        fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Target")
        fig.update_layout(title="Delay Cascade Under Different Conditions",
                         xaxis_title="Station", yaxis_title="Delay (min)",
                         height=500, template="plotly_dark",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with dt3:
        st.subheader("Historical Patterns")
        if not train_delays.empty:
            c1, c2 = st.columns(2)
            with c1:
                dow = train_delays.groupby("day_of_week")["delay_minutes"].mean().reset_index()
                dm = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
                dow["day_name"] = dow["day_of_week"].map(dm)
                fig = px.bar(dow, x="day_name", y="delay_minutes", color="delay_minutes",
                             color_continuous_scale="RdYlGn_r", title="Delay by Day of Week")
                fig.update_layout(height=350, template="plotly_dark", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                train_delays["season"] = train_delays.apply(
                    lambda r: "Monsoon" if r["is_monsoon"] else (
                        "Fog" if r["is_fog_season"] else (
                            "Festival" if r["is_festival_season"] else "Normal")), axis=1)
                sd = train_delays.groupby("season")["delay_minutes"].mean().reset_index()
                fig = px.bar(sd, x="season", y="delay_minutes", color="season",
                             color_discrete_map={"Normal": "#4CAF50", "Monsoon": "#2196F3",
                                                 "Fog": "#9E9E9E", "Festival": "#FF9800"},
                             title="Delay by Season")
                fig.update_layout(height=350, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Station x Month Delay Heatmap")
            hm = train_delays.groupby(["station_code", "month"])["delay_minutes"].mean().reset_index()
            hp = hm.pivot(index="station_code", columns="month", values="delay_minutes").fillna(0)
            hp.index = hp.index.map(stn_name)
            fig = px.imshow(hp, color_continuous_scale="RdYlGn_r",
                            title=f"Delay Heatmap — {selected_train_name}",
                            labels=dict(x="Month", y="Station", color="Delay (min)"))
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ============================================
# TAB 3: TICKET INTELLIGENCE
# ============================================
elif mode == "Ticket Intelligence":
    st.markdown(f'<p class="train-header">Ticket Intelligence — {selected_train_name}</p>',
                unsafe_allow_html=True)

    tt1, tt2, tt3 = st.tabs(["Confirmation Predictor", "Booking Advisor", "Booking Statistics"])

    with tt1:
        st.subheader("Predict Ticket Confirmation Probability")
        c1, c2 = st.columns(2)
        with c1:
            src = st.selectbox("From", selected_train_stations[:-1],
                               format_func=lambda x: f"{x} — {stn_name(x)}", key="ts")
            rem = [s for s in selected_train_stations
                   if selected_train_stations.index(s) > selected_train_stations.index(src)]
            dst = st.selectbox("To", rem, format_func=lambda x: f"{x} — {stn_name(x)}", key="td")
            tc = st.selectbox("Class", selected_train_classes, key="tcl")
        with c2:
            wl = st.number_input("Waitlist Position (WL/)", 1, 200, 20, key="twl")
            db = st.number_input("Days Before Travel", 0, 120, 15, key="tdb")
            tm = st.slider("Travel Month", 1, 12, now.month, key="ttm")
            fest = st.checkbox("Festival Season", tm in [10, 11, 12, 3, 4], key="tf")

        if st.button("Predict Confirmation", type="primary", use_container_width=True):
            res = conf_pred.predict_confirmation(
                selected_train_number, src, dst, tc, db, wl, tm,
                now.weekday(), fest, tm in [4, 5, 6, 10, 11, 12])
            prob = res["confirmation_probability"]

            ca, cb = st.columns([1, 2])
            with ca:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={"suffix": "%", "font": {"size": 48}},
                    title={"text": "Confirmation Probability"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#FF6B35"},
                           "steps": [{"range": [0, 40], "color": "#ffcccc"},
                                     {"range": [40, 70], "color": "#fff3cd"},
                                     {"range": [70, 100], "color": "#d4edda"}],
                           "threshold": {"line": {"color": "red", "width": 4},
                                         "thickness": 0.75, "value": 50}}))
                fig.update_layout(height=280, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with cb:
                if prob > 0.7:
                    st.success(res['recommendation'])
                elif prob > 0.4:
                    st.warning(res['recommendation'])
                else:
                    st.error(res['recommendation'])

                st.markdown(f"""
                | Parameter | Value |
                |-----------|-------|
                | **Train** | {selected_train_name} |
                | **Route** | {stn_name(src)} to {stn_name(dst)} |
                | **Class** | {tc} |
                | **Waitlist** | WL/{wl} |
                | **Days Before** | {db} |
                | **Probability** | **{prob:.1%}** |
                """)

            st.subheader("Comparison Across Classes (Same WL Position)")
            cd = []
            for cl in selected_train_classes:
                p = conf_pred.predict_confirmation(
                    selected_train_number, src, dst, cl, db, wl, tm, now.weekday())
                cd.append({"Class": cl, "Probability": p["confirmation_probability"]})
            cdf = pd.DataFrame(cd)
            fig = px.bar(cdf, x="Class", y="Probability", color="Probability",
                         color_continuous_scale="RdYlGn",
                         text=cdf["Probability"].apply(lambda x: f"{x:.0%}"),
                         title=f"Confirmation by Class (WL/{wl})")
            fig.update_layout(height=350, template="plotly_dark",
                             yaxis=dict(tickformat=".0%"), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tt2:
        st.subheader(f"Optimal Booking Window — {selected_train_name}")
        ac = st.selectbox("Class", selected_train_classes, key="ac")
        tgt = st.slider("Target Confirmation %", 50, 95, 80, key="at")

        if st.button("Analyze Booking Window", type="primary", use_container_width=True):
            adv = conf_pred.recommend_booking_advance(
                selected_train_number, selected_train_stations[0],
                selected_train_stations[-1], ac, now.month, tgt / 100)
            st.success(f"Recommendation: Book at least **{adv['recommended_days_advance']} days** "
                      f"in advance for {tgt}% confirmation ({ac})")

            bdf = pd.DataFrame(adv["booking_analysis"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bdf["days_before"], y=bdf["confirmation_probability"],
                mode="lines+markers+text",
                text=bdf["confirmation_probability"].apply(lambda x: f"{x:.0%}"),
                textposition="top center",
                line=dict(color="#4ECDC4", width=4), marker=dict(size=10),
                fill="tozeroy", fillcolor="rgba(78,205,196,0.15)"))
            fig.add_hline(y=tgt / 100, line_dash="dash", line_color="red",
                         annotation_text=f"Target: {tgt}%")
            fig.add_vline(x=adv["recommended_days_advance"], line_dash="dash",
                         line_color="yellow",
                         annotation_text=f"{adv['recommended_days_advance']}+ days")
            fig.update_layout(title="Confirmation Probability vs Booking Advance",
                             xaxis_title="Days Before Travel",
                             yaxis_title="Probability",
                             height=450, template="plotly_dark",
                             yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

    with tt3:
        st.subheader(f"Booking Statistics — {selected_train_name}")
        if not train_bookings.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Bookings", f"{len(train_bookings):,}")
            c2.metric("Confirmed", f"{train_bookings['confirmed'].sum():,}")
            c3.metric("Confirmation Rate", f"{train_bookings['confirmed'].mean():.1%}")
            c4.metric("Avg Waitlist", f"WL/{train_bookings['waitlist_position'].mean():.0f}")

            cl, cr = st.columns(2)
            with cl:
                cs = train_bookings.groupby("travel_class")["confirmed"].mean().reset_index()
                fig = px.bar(cs, x="travel_class", y="confirmed", color="confirmed",
                             color_continuous_scale="RdYlGn",
                             text=cs["confirmed"].apply(lambda x: f"{x:.0%}"),
                             title="Confirmation Rate by Class")
                fig.update_layout(height=350, template="plotly_dark",
                                 yaxis=dict(tickformat=".0%"), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with cr:
                fig = px.histogram(train_bookings, x="waitlist_position", nbins=30,
                                   color="confirmed",
                                   color_discrete_map={True: "#4CAF50", False: "#f44336"},
                                   title="Waitlist Distribution",
                                   labels={"confirmed": "Confirmed"})
                fig.update_layout(height=350, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)


# ============================================
# TAB 4: STATION CONGESTION (FIXED)
# ============================================
elif mode == "Station Congestion":
    st.markdown(f'<p class="train-header">Station Congestion — {selected_train_name} Route</p>',
                unsafe_allow_html=True)

    ct1, ct2 = st.tabs(["Route Overview", "Station Deep Dive"])

    with ct1:
        sel_hour = st.slider("Hour of Day", 0, 23, now.hour, key="ch")

        # Get properly calculated congestion from fixed model
        all_congestion = congestion_analyzer.get_current_congestion(sel_hour)

        # Filter for this train's route stations
        route_congestion = all_congestion[
            all_congestion["station_code"].isin(selected_train_stations)
        ].copy()

        if not route_congestion.empty:
            # Sort by route order
            route_congestion["order"] = route_congestion["station_code"].apply(
                lambda x: selected_train_stations.index(x) if x in selected_train_stations else 99
            )
            route_congestion = route_congestion.sort_values("order").reset_index(drop=True)

            # Metric cards
            n = min(len(route_congestion), 5)
            cols = st.columns(n)
            for i, (_, row) in enumerate(route_congestion.head(10).iterrows()):
                level = row["congestion_level"]
                with cols[i % n]:
                    st.metric(
                        f"{row['station_code']} [{level}]",
                        f"{row['occupancy_pct']:.1%}",
                        f"{row['passenger_count']:,} passengers/hr"
                    )

            # Bar chart
            fig = px.bar(
                route_congestion, x="station_name", y="occupancy_pct",
                color="congestion_level",
                color_discrete_map={
                    "Critical": "#cc3333",
                    "High": "#cc7700",
                    "Moderate": "#ccaa00",
                    "Low": "#339933"
                },
                title=f"Station Occupancy at {sel_hour}:00 — {selected_train_name} Route",
                text=route_congestion["occupancy_pct"].apply(lambda x: f"{x:.1%}")
            )
            fig.add_hline(y=0.60, line_dash="dash", line_color="red",
                         annotation_text="High Threshold (60%)")
            fig.add_hline(y=0.82, line_dash="dash", line_color="darkred",
                         annotation_text="Critical Threshold (82%)")
            fig.update_layout(
                height=450, template="plotly_dark",
                yaxis=dict(tickformat=".0%", title="Occupancy Rate"),
                xaxis_title="Station"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Classification explanation
            st.markdown("""
            | Level | Occupancy | Interpretation |
            |-------|-----------|---------------|
            | **Critical** | Above 82% | Platforms near capacity, significant crowding expected |
            | **High** | 60% — 82% | Busy conditions, moderate wait times |
            | **Moderate** | 30% — 60% | Normal operations, comfortable movement |
            | **Low** | Below 30% | Minimal crowding, best travel conditions |
            """)

            # Hotspots
            route_hotspots = route_congestion[route_congestion["occupancy_pct"] >= 0.60]
            if not route_hotspots.empty:
                st.warning(f"{len(route_hotspots)} station(s) above high-congestion threshold at {sel_hour}:00")
                for _, h in route_hotspots.iterrows():
                    st.write(f"**{h['station_name']}** — {h['occupancy_pct']:.1%} occupancy "
                            f"({h['passenger_count']:,} avg passengers/hr) — [{h['congestion_level']}]")
            else:
                st.success(f"No congestion alerts on this route at {sel_hour}:00")
        else:
            st.info("No congestion data available for stations on this route.")

    with ct2:
        st.subheader("24-Hour Congestion Profile")

        stn_pick = st.selectbox(
            "Select Station",
            selected_train_stations,
            format_func=lambda x: f"{x} — {stn_name(x)}",
            key="ds"
        )

        pattern = congestion_analyzer.get_station_pattern(stn_pick)
        peaks = congestion_analyzer.predict_peak_hours(stn_pick)

        if not pattern.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pattern["hour"], y=pattern["occupancy_pct"],
                fill="tozeroy", mode="lines+markers",
                line=dict(color="#FF6B35", width=3),
                fillcolor="rgba(255,107,53,0.15)",
                hovertemplate="Hour: %{x}:00<br>Occupancy: %{y:.1%}<extra></extra>"
            ))
            fig.add_hline(y=0.60, line_dash="dash", line_color="orange",
                         annotation_text="High Threshold")
            fig.add_hline(y=0.82, line_dash="dash", line_color="red",
                         annotation_text="Critical Threshold")
            fig.update_layout(
                title=f"24-Hour Occupancy Profile — {stn_name(stn_pick)}",
                xaxis_title="Hour of Day", yaxis_title="Occupancy Rate",
                height=420, template="plotly_dark",
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                xaxis=dict(dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Peak vs quiet display
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Peak Hours (Avoid)")
                if peaks["peak_hours"]:
                    for p in peaks["peak_hours"][:5]:
                        st.write(f"**{int(p['hour'])}:00** — {p['occupancy_pct']:.1%} occupancy "
                                f"({int(p['passenger_count']):,} pax/hr)")
            with c2:
                st.markdown("### Off-Peak Hours (Recommended)")
                if peaks["off_peak_hours"]:
                    for p in peaks["off_peak_hours"][:5]:
                        st.write(f"**{int(p['hour'])}:00** — {p['occupancy_pct']:.1%} occupancy "
                                f"({int(p['passenger_count']):,} pax/hr)")

            # Summary metrics
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Average Occupancy", f"{peaks['avg_occupancy']:.1%}")
            m2.metric("Peak Occupancy", f"{peaks['max_occupancy']:.1%}")
            m3.metric("Minimum Occupancy", f"{peaks['min_occupancy']:.1%}")
        else:
            st.info("No pattern data available for this station.")


# ============================================
# TAB 5: ROUTE MAP
# ============================================
elif mode == "Route Map":
    st.markdown(f'<p class="train-header">Route Map — {selected_train_name}</p>', unsafe_allow_html=True)

    rs = stations[stations["code"].isin(selected_train_stations)]
    m = folium.Map(location=[rs["lat"].mean(), rs["lon"].mean()],
                   zoom_start=6, tiles="CartoDB dark_matter")

    coords = []
    for sc in selected_train_stations:
        s = stations[stations["code"] == sc]
        if not s.empty:
            coords.append([s.iloc[0]["lat"], s.iloc[0]["lon"]])

    if len(coords) >= 2:
        rc = "#FF6B35" if selected_train_route == "western" else "#4ECDC4"
        folium.PolyLine(coords, weight=5, color=rc, opacity=0.9,
                        tooltip=selected_train_name).add_to(m)

    for i, sc in enumerate(selected_train_stations):
        s = stations[stations["code"] == sc]
        if s.empty:
            continue
        s = s.iloc[0]
        sd = train_delays[train_delays["station_code"] == sc]["delay_minutes"].mean()
        sd = sd if not np.isnan(sd) else 0

        if i == 0:
            color, radius = "green", 14
        elif i == len(selected_train_stations) - 1:
            color, radius = "red", 14
        else:
            color, radius = "orange", 10

        popup = f"""<div style="font-family:Arial;width:200px;">
            <h4 style="color:#FF6B35;margin:0;">{s['name']}</h4>
            <p>Code: {s['code']}<br>Zone: {s['zone']}<br>
            Type: {s['type'].title()}<br>Platforms: {s['platforms']}<br>
            Daily Footfall: {s['daily_footfall']:,}<br>
            Avg Delay: {sd:.1f} min</p></div>"""

        folium.CircleMarker(
            location=[s["lat"], s["lon"]], radius=radius,
            color=color, fill=True, fillColor=color, fillOpacity=0.8,
            popup=folium.Popup(popup, max_width=220),
            tooltip=f"{s['name']} — Delay: {sd:.1f} min"
        ).add_to(m)

    st_folium(m, width=None, height=550)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stops", len(selected_train_stations))
    c2.metric("Route", selected_train_route.title())
    c3.metric("Duration", f"{selected_train_row['duration_hrs']} hrs")
    tavg = train_delays["delay_minutes"].mean() if not train_delays.empty else 0
    c4.metric("Avg Route Delay", f"{tavg:.1f} min")


# ============================================
# TAB 6: NETWORK RESILIENCE
# ============================================
elif mode == "Network Resilience":
    st.markdown(f'<p class="train-header">Network Resilience — {selected_train_name}</p>',
                unsafe_allow_html=True)

    rt1, rt2 = st.tabs(["Station Criticality", "Failure Simulation"])

    with rt1:
        cdf = network.get_centrality_analysis()
        rcdf = cdf[cdf["station_code"].isin(selected_train_stations)].sort_values(
            "vulnerability_score", ascending=False)

        fig = px.bar(rcdf, x="station_name", y="vulnerability_score",
                     color="vulnerability_score", color_continuous_scale="RdYlGn_r",
                     text=rcdf["vulnerability_score"].apply(lambda x: f"{x:.3f}"),
                     title="Vulnerability Scores — Route Stations")
        fig.update_layout(height=400, template="plotly_dark", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rcdf[["station_code", "station_name", "type", "trains_through",
                            "degree_centrality", "betweenness_centrality",
                            "vulnerability_score"]], use_container_width=True, hide_index=True)

        if len(rcdf) >= 3:
            cats = ["Degree", "Betweenness", "Closeness", "PageRank"]
            fig = go.Figure()
            for _, row in rcdf.head(5).iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row["degree_centrality"], row["betweenness_centrality"],
                       row["closeness_centrality"], row["pagerank"]],
                    theta=cats, fill='toself', name=row["station_name"]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                             title="Centrality Radar", height=450, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    with rt2:
        fs = st.selectbox("Select Station to Remove", selected_train_stations,
                           format_func=lambda x: f"{x} — {stn_name(x)}", key="fs")

        if st.button("Simulate Failure", type="primary", use_container_width=True):
            r = network.simulate_station_failure(fs)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trains Affected", r["num_affected_trains"])
            c2.metric("Edges Removed", r["edges_removed"])
            c3.metric("Network Fragments", r["network_fragments"])
            c4.metric("Impact Score", r["impact_score"])

            if not r["network_still_connected"]:
                st.error(f"CRITICAL: Removing {stn_name(fs)} fragments the network into "
                        f"{r['network_fragments']} disconnected components.")
            else:
                st.success(f"Network remains connected after removing {stn_name(fs)}.")

            if r["affected_train_details"]:
                st.subheader("Affected Trains")
                st.dataframe(pd.DataFrame(r["affected_train_details"]),
                             use_container_width=True, hide_index=True)

        st.subheader("Failure Impact — All Route Stations")
        impacts = []
        for sc in selected_train_stations:
            res = network.simulate_station_failure(sc)
            impacts.append({"Station": stn_name(sc), "Code": sc,
                            "Trains Affected": res["num_affected_trains"],
                            "Impact Score": res["impact_score"],
                            "Breaks Network": "Yes" if not res["network_still_connected"] else "No"})
        idf = pd.DataFrame(impacts).sort_values("Impact Score", ascending=False)
        fig = px.bar(idf, x="Station", y="Impact Score", color="Breaks Network",
                     color_discrete_map={"Yes": "#cc3333", "No": "#339933"},
                     text="Trains Affected", title="Station Failure Impact Comparison")
        fig.update_traces(texttemplate='%{text} trains', textposition='outside')
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


# ============================================
# TAB 7: AI ASSISTANT
# ============================================
elif mode == "AI Assistant":
    st.markdown(f'<p class="train-header">RailMitra — AI Assistant</p>', unsafe_allow_html=True)
    st.markdown(f"*Analyzing: **{selected_train_name}** ({selected_train_number}) — "
                f"{selected_train_route.title()} Route*")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": f"""Welcome to RailMitra. I am currently analyzing **{selected_train_name} ({selected_train_number})**.

I can assist with:
- **Delay Analysis** — Historical patterns and predictions
- **Ticket Confirmation** — Probability estimates and booking guidance
- **Station Congestion** — Occupancy levels along the route
- **Route Intelligence** — Alternative paths and network analysis
- **Infrastructure Resilience** — Vulnerability assessment

Please enter your query below."""}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("**Suggested Queries:**")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button(f"Delay profile for {selected_train_name}", use_container_width=True, key="q1"):
            st.session_state.pq = f"What is the delay profile for {selected_train_name}?"
        if st.button("WL/25 confirmation chances in 3AC", use_container_width=True, key="q2"):
            st.session_state.pq = f"What are the chances of WL/25 in 3AC getting confirmed on {selected_train_name}?"
        if st.button("Optimal booking window", use_container_width=True, key="q3"):
            st.session_state.pq = f"When should I book {selected_train_name} for highest confirmation?"
    with sc2:
        if st.button("Station congestion on this route", use_container_width=True, key="q4"):
            st.session_state.pq = f"What is the congestion status at stations on {selected_train_name}'s route?"
        if st.button("Alternative trains available", use_container_width=True, key="q5"):
            st.session_state.pq = "What are the alternative trains from Mumbai to Delhi?"
        if st.button("Network vulnerability assessment", use_container_width=True, key="q6"):
            st.session_state.pq = f"Which stations on {selected_train_name}'s route are most critical to network integrity?"

    user_input = st.chat_input(f"Enter your query about {selected_train_name}...")

    if "pq" in st.session_state:
        user_input = st.session_state.pq
        del st.session_state.pq

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing query..."):
                try:
                    from assistant.rail_assistant import RailwayAssistant
                    assistant = RailwayAssistant(
                        delay_predictor=delay_pred, confirmation_predictor=conf_pred,
                        congestion_analyzer=congestion_analyzer, railway_network=network)
                    eq = (f"[Train: {selected_train_name} ({selected_train_number}), "
                          f"Route: {' -> '.join(selected_train_stations)}, "
                          f"Type: {selected_train_type}] {user_input}")
                    response = assistant.chat(eq)
                except Exception as e:
                    response = fallback_response(user_input, str(e))

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


def fallback_response(query, err=""):
    ql = query.lower()
    parts = [f"**RailMitra Analysis — {selected_train_name}**\n"]

    if any(w in ql for w in ["delay", "late"]):
        if not train_delays.empty:
            parts.append(f"Average Delay: **{train_delays['delay_minutes'].mean():.1f} min**")
            parts.append(f"Maximum Recorded: **{train_delays['delay_minutes'].max():.1f} min**")
    elif any(w in ql for w in ["ticket", "confirm", "wl", "book"]):
        p = conf_pred.predict_confirmation(selected_train_number, selected_train_stations[0],
                                            selected_train_stations[-1], "3A", 15, 20, now.month, now.weekday())
        parts.append(f"3AC WL/20, 15 days advance: **{p['confirmation_probability']:.0%}**")
        parts.append(p['recommendation'])
    elif any(w in ql for w in ["crowd", "congestion", "occupancy"]):
        cong = congestion_analyzer.get_current_congestion(now.hour)
        route_cong = cong[cong["station_code"].isin(selected_train_stations)]
        for _, r in route_cong.iterrows():
            parts.append(f"{r['station_name']}: {r['occupancy_pct']:.1%} [{r['congestion_level']}]")
    elif any(w in ql for w in ["route", "alternative"]):
        routes = network.find_alternative_routes(selected_train_stations[0], selected_train_stations[-1])
        for r in routes:
            if "error" not in r:
                parts.append(f"Route {r['route_id']}: {' -> '.join(r['path_names'])}")
    else:
        parts.append("Available topics: delay analysis, ticket confirmation, station congestion, "
                    "route alternatives, network resilience.")
    return "\n\n".join(parts)



