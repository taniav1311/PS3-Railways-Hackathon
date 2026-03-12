import os
import sys
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ============================================
# GROQ API SETUP WITH YOUR KEY
# ============================================

groq_client = None

try:
    from groq import Groq

    GROQ_API_KEY = "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"

    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq API connected successfully!")

except ImportError:
    print("⚠️ groq package not installed. Run: pip install groq")
    print("ℹ️ Falling back to offline mode")
except Exception as e:
    print(f"⚠️ Groq connection error: {e}")
    print("ℹ️ Falling back to offline mode")


class RailwayAssistant:
    """AI-powered railway assistant using Groq LLM + ML model analytics"""

    def __init__(self, delay_predictor=None, confirmation_predictor=None,
                 congestion_analyzer=None, railway_network=None):
        self.delay_predictor = delay_predictor
        self.confirmation_predictor = confirmation_predictor
        self.congestion_analyzer = congestion_analyzer
        self.railway_network = railway_network

        self.system_prompt = """You are RailMitra, an AI-powered Indian Railways intelligent assistant 
for the Mumbai-Delhi corridor. You analyze real-time data from 20 stations and 10 trains 
on both Western and Central routes.

Your capabilities:
1. 🕐 Train delay prediction and cascade analysis
2. 🎫 Ticket confirmation probability and smart booking advice
3. 👥 Station congestion monitoring and crowd intelligence
4. 🛤️ Route recommendations and alternative paths
5. 🛡️ Network vulnerability and resilience analysis

IMPORTANT RULES:
- ALWAYS use the PROVIDED ANALYTICS DATA to answer. Never make up numbers.
- Be concise but thorough. Use bullet points and emojis.
- Include specific numbers, percentages, train names, and station names from the data.
- Give actionable recommendations passengers/operators can use immediately.
- If data for something isn't available, say so honestly.
- For passengers: be friendly, reassuring, and practical.
- For operators: be analytical, professional, and strategic.
- Format responses with clear headers and structure.
- When discussing delays, always mention which trains are safest.
- When discussing tickets, always suggest alternatives if chances are low."""

        self.chat_history = []
        print("🤖 RailMitra Assistant initialized!")

    # ============================================
    # CONTEXT BUILDER - Fetches relevant ML data
    # ============================================

    def _build_context(self, query):
        """Build context from ML models based on query keywords"""
        context_parts = []
        query_lower = query.lower()
        now = datetime.now()

        # ---- DELAY related queries ----
        if any(word in query_lower for word in
               ["delay", "late", "cascade", "vulnerable", "disrupt", "punctual", "on time",
                "which train", "worst", "best train", "reliable", "priority"]):
            if self.delay_predictor:
                try:
                    vuln = self.delay_predictor.get_vulnerable_routes(now.month, now.weekday())
                    context_parts.append(
                        f"📊 DELAY VULNERABILITY RANKING (Month: {now.strftime('%B')}, "
                        f"Day: {now.strftime('%A')}):\n{vuln.to_string(index=False)}"
                    )

                    # Predictions for key trains at key stations
                    predictions = []
                    test_points = [
                        ("12951", "NDLS", "Rajdhani at Delhi"),
                        ("12951", "BRC", "Rajdhani at Vadodara"),
                        ("12903", "KOTA", "Golden Temple at Kota"),
                        ("12137", "BPL", "Punjab Mail at Bhopal"),
                        ("12137", "NDLS", "Punjab Mail at Delhi"),
                        ("12621", "BPL", "Tamil Nadu Exp at Bhopal"),
                    ]
                    for train, stn, label in test_points:
                        pred = self.delay_predictor.predict_delay(
                            train, stn, now.month, now.weekday(),
                            is_monsoon=now.month in [6, 7, 8, 9],
                            is_fog=now.month in [12, 1],
                            is_festival=now.month in [10, 11, 12, 3, 4]
                        )
                        predictions.append(f"  {label}: {pred['predicted_delay_minutes']} min "
                                         f"({pred['cascade_severity']})")

                    context_parts.append(
                        f"\n🔮 SPECIFIC DELAY PREDICTIONS:\n" + "\n".join(predictions)
                    )

                    # Full route prediction for Rajdhani
                    route_delays = self.delay_predictor.predict_route_delays(
                        "12951", ["MMCT", "BRC", "KOTA", "SWM", "NDLS"],
                        now.month, now.weekday(),
                        is_monsoon=now.month in [6, 7, 8, 9]
                    )
                    route_str = "\n".join([
                        f"  {r['station_name']}: {r['predicted_delay_minutes']} min "
                        f"({r['cascade_severity']})"
                        for r in route_delays
                    ])
                    context_parts.append(f"\n🚂 RAJDHANI ROUTE CASCADE:\n{route_str}")

                    metrics = self.delay_predictor.get_model_metrics()
                    context_parts.append(
                        f"\n📈 Model: MAE={metrics['delay_model_mae']}min, R²={metrics['delay_model_r2']}"
                    )
                except Exception as e:
                    context_parts.append(f"⚠️ Delay data error: {e}")

        # ---- CONGESTION related queries ----
        if any(word in query_lower for word in
               ["congestion", "crowd", "busy", "peak", "rush", "passenger", "footfall",
                "quiet", "empty", "best time", "avoid", "platform", "overcrowd"]):
            if self.congestion_analyzer:
                try:
                    # Current congestion
                    congestion = self.congestion_analyzer.get_current_congestion(now.hour)
                    context_parts.append(
                        f"\n👥 CURRENT STATION CONGESTION ({now.hour}:00):\n"
                        f"{congestion.to_string(index=False)}"
                    )

                    # Hotspots
                    hotspots = self.congestion_analyzer.get_hotspots(now.hour)
                    if not hotspots.empty:
                        context_parts.append(
                            f"\n🔥 ACTIVE HOTSPOTS ({len(hotspots)} stations over 70%):\n"
                            f"{hotspots.to_string(index=False)}"
                        )

                    # Evening rush hotspots
                    evening_hotspots = self.congestion_analyzer.get_hotspots(18)
                    if not evening_hotspots.empty:
                        context_parts.append(
                            f"\n🌆 EVENING RUSH HOTSPOTS (18:00): {len(evening_hotspots)} stations"
                        )

                    # Peak hours for major stations
                    context_parts.append("\n⏰ PEAK vs QUIET HOURS:")
                    for stn in ["NDLS", "CSMT", "MMCT", "BRC", "KOTA"]:
                        peaks = self.congestion_analyzer.predict_peak_hours(stn)
                        top = peaks["peak_hours"][0] if peaks["peak_hours"] else {}
                        best = peaks["off_peak_hours"][0] if peaks["off_peak_hours"] else {}
                        context_parts.append(
                            f"  {peaks['station_name']}: "
                            f"Peak={top.get('hour', '?')}:00 ({top.get('utilization', 0):.1%}) | "
                            f"Quiet={best.get('hour', '?')}:00 ({best.get('utilization', 0):.1%}) | "
                            f"Avg={peaks['avg_utilization']:.1%}"
                        )

                    # Corridor comparison
                    corridors = self.congestion_analyzer.get_corridor_congestion()
                    for route_name, data in corridors.items():
                        context_parts.append(
                            f"\n🛤️ {route_name} CORRIDOR CONGESTION:\n{data.to_string(index=False)}"
                        )

                    # Station comparison
                    comparison = self.congestion_analyzer.compare_stations(
                        ["NDLS", "CSMT", "MMCT", "BRC", "KOTA", "BPL", "JHS"]
                    )
                    context_parts.append(
                        f"\n📊 STATION COMPARISON:\n{comparison.to_string(index=False)}"
                    )
                except Exception as e:
                    context_parts.append(f"⚠️ Congestion data error: {e}")

        # ---- TICKET / BOOKING related queries ----
        if any(word in query_lower for word in
               ["ticket", "confirm", "waitlist", "wl", "rac", "book", "reservation",
                "cancel", "available", "chance", "probability", "when should",
                "how many days", "advance", "tatkal"]):
            if self.confirmation_predictor:
                try:
                    # Multiple sample predictions
                    context_parts.append("\n🎫 TICKET CONFIRMATION PREDICTIONS:")
                    samples = [
                        ("12951", "MMCT", "NDLS", "1A", 30, 3, "Rajdhani 1A WL/3 30days"),
                        ("12951", "MMCT", "NDLS", "2A", 20, 10, "Rajdhani 2A WL/10 20days"),
                        ("12951", "MMCT", "NDLS", "3A", 15, 25, "Rajdhani 3A WL/25 15days"),
                        ("12951", "MMCT", "NDLS", "3A", 7, 40, "Rajdhani 3A WL/40 7days"),
                        ("12951", "MMCT", "NDLS", "3A", 3, 60, "Rajdhani 3A WL/60 3days"),
                        ("12903", "MMCT", "NDLS", "SL", 10, 50, "Golden Temple SL WL/50 10days"),
                        ("12903", "MMCT", "NDLS", "3A", 20, 20, "Golden Temple 3A WL/20 20days"),
                        ("12137", "CSMT", "NDLS", "3A", 15, 15, "Punjab Mail 3A WL/15 15days"),
                        ("12137", "CSMT", "NDLS", "SL", 5, 70, "Punjab Mail SL WL/70 5days"),
                    ]

                    for train, src, dst, cls, days, wl, label in samples:
                        pred = self.confirmation_predictor.predict_confirmation(
                            train, src, dst, cls, days, wl, now.month, now.weekday()
                        )
                        context_parts.append(
                            f"  {label}: {pred['confirmation_probability']:.1%} "
                            f"{pred['confidence_emoji']} {pred['recommendation']}"
                        )

                    # Booking advance recommendations
                    context_parts.append("\n📅 BOOKING ADVANCE ANALYSIS:")
                    for cls in ["1A", "2A", "3A", "SL"]:
                        advance = self.confirmation_predictor.recommend_booking_advance(
                            "12951", "MMCT", "NDLS", cls, now.month
                        )
                        context_parts.append(
                            f"  Rajdhani {cls}: Book {advance['recommended_days_advance']}+ days "
                            f"before for 80% confirmation"
                        )

                    # Probability timeline for 3A
                    advance_3a = self.confirmation_predictor.recommend_booking_advance(
                        "12951", "MMCT", "NDLS", "3A", now.month
                    )
                    context_parts.append("\n📊 CONFIRMATION PROBABILITY TIMELINE (Rajdhani 3A):")
                    for entry in advance_3a["booking_analysis"]:
                        bar = "█" * int(entry["confirmation_probability"] * 15)
                        status = "✅" if entry["confirmation_probability"] >= 0.8 else "❌"
                        context_parts.append(
                            f"  {entry['days_before']:>3d} days before: "
                            f"{entry['confirmation_probability']:.0%} {bar} {status}"
                        )

                    # Best alternatives
                    alts_mmct = self.confirmation_predictor.find_best_alternatives(
                        "MMCT", "NDLS", "3A", 15, now.month
                    )
                    if alts_mmct:
                        context_parts.append("\n🔄 BEST ALTERNATIVES (MMCT→NDLS, 15 days before):")
                        for a in alts_mmct[:8]:
                            context_parts.append(
                                f"  {a['train_name']} ({a['class']}): "
                                f"{a['confirmation_probability']:.1%} {a['emoji']}"
                            )

                    alts_csmt = self.confirmation_predictor.find_best_alternatives(
                        "CSMT", "NDLS", "3A", 15, now.month
                    )
                    if alts_csmt:
                        context_parts.append("\n🔄 BEST ALTERNATIVES (CSMT→NDLS, 15 days before):")
                        for a in alts_csmt[:6]:
                            context_parts.append(
                                f"  {a['train_name']} ({a['class']}): "
                                f"{a['confirmation_probability']:.1%} {a['emoji']}"
                            )

                    # Model metrics
                    metrics = self.confirmation_predictor.get_model_metrics()
                    context_parts.append(f"\n📈 Model Accuracy: {metrics['accuracy']:.1%}")

                except Exception as e:
                    context_parts.append(f"⚠️ Booking data error: {e}")

        # ---- ROUTE / NETWORK related queries ----
        if any(word in query_lower for word in
               ["route", "alternative", "path", "network", "station", "critical",
                "failure", "resilience", "community", "central", "western",
                "vulnerability", "corridor", "connection", "infrastructure",
                "shutdown", "break"]):
            if self.railway_network:
                try:
                    # Centrality analysis
                    centrality = self.railway_network.get_centrality_analysis()
                    context_parts.append(
                        f"\n🏛️ STATION CRITICALITY RANKING:\n"
                        f"{centrality[['station_code', 'station_name', 'type', 'trains_through', 'vulnerability_score']].head(10).to_string(index=False)}"
                    )

                    # Routes from Mumbai Central
                    routes_mmct = self.railway_network.find_alternative_routes("MMCT", "NDLS", k=5)
                    context_parts.append("\n🛤️ ROUTES (Mumbai Central → New Delhi):")
                    for r in routes_mmct:
                        if "error" not in r:
                            direct = f" | Direct: {', '.join(r['direct_train_names'])}" if r[
                                'direct_train_names'] else " | No direct train"
                            context_parts.append(
                                f"  Route {r['route_id']}: {' → '.join(r['path_names'])} "
                                f"({r['num_stops']} stops){direct}"
                            )

                    # Routes from CSMT
                    routes_csmt = self.railway_network.find_alternative_routes("CSMT", "NDLS", k=3)
                    context_parts.append("\n🛤️ ROUTES (Mumbai CSMT → New Delhi):")
                    for r in routes_csmt:
                        if "error" not in r:
                            direct = f" | Direct: {', '.join(r['direct_train_names'])}" if r[
                                'direct_train_names'] else ""
                            context_parts.append(
                                f"  Route {r['route_id']}: {' → '.join(r['path_names'])} "
                                f"({r['num_stops']} stops){direct}"
                            )

                    # Busiest corridors
                    corridors = self.railway_network.get_busiest_corridors(8)
                    context_parts.append(
                        f"\n🔥 BUSIEST CORRIDORS:\n"
                        f"{corridors[['from_name', 'to_name', 'num_trains', 'route']].to_string(index=False)}"
                    )

                    # Station failure simulations
                    context_parts.append("\n💥 FAILURE IMPACT ANALYSIS:")
                    for stn in ["NDLS", "BRC", "KOTA", "MTJ", "KYN"]:
                        result = self.railway_network.simulate_station_failure(stn)
                        status = "❌ BREAKS NETWORK" if not result[
                            "network_still_connected"] else "✅ Network survives"
                        context_parts.append(
                            f"  Remove {result['station_name']}: "
                            f"{result['num_affected_trains']} trains affected, "
                            f"Impact={result['impact_score']}, {status}"
                        )

                    # Communities
                    communities = self.railway_network.detect_communities()
                    if communities:
                        context_parts.append("\n🏘️ NETWORK COMMUNITIES:")
                        for comm in communities:
                            names = [s["name"] for s in comm["stations"]]
                            context_parts.append(
                                f"  Cluster {comm['community_id']}: {', '.join(names)}"
                            )

                    # Network stats
                    stats = self.railway_network.get_network_stats()
                    context_parts.append(
                        f"\n📊 NETWORK STATS: {stats['total_stations']} stations, "
                        f"{stats['total_connections']} connections, "
                        f"{stats['total_unique_trains']} trains, "
                        f"Density={stats['network_density']}"
                    )
                except Exception as e:
                    context_parts.append(f"⚠️ Network data error: {e}")

        # ---- If no specific match, give general overview ----
        if not context_parts:
            context_parts.append(self._get_general_overview())

        return "\n".join(context_parts)

    def _get_general_overview(self):
        """General overview when query doesn't match categories"""
        now = datetime.now()
        overview = [f"📊 MUMBAI-DELHI CORRIDOR OVERVIEW ({now.strftime('%d %B %Y, %H:%M')}):\n"]

        if self.delay_predictor:
            vuln = self.delay_predictor.get_vulnerable_routes(now.month, now.weekday())
            overview.append(f"⏱️ DELAYS:")
            overview.append(f"  Most delayed: {vuln.iloc[0]['train_name']} "
                          f"(avg {vuln.iloc[0]['avg_delay']:.0f} min) {vuln.iloc[0]['risk_level']}")
            overview.append(f"  Least delayed: {vuln.iloc[-1]['train_name']} "
                          f"(avg {vuln.iloc[-1]['avg_delay']:.0f} min) {vuln.iloc[-1]['risk_level']}")

        if self.congestion_analyzer:
            congestion = self.congestion_analyzer.get_current_congestion(now.hour)
            if not congestion.empty:
                worst = congestion.iloc[0]
                best = congestion.iloc[-1]
                overview.append(f"\n👥 CONGESTION:")
                overview.append(f"  Most crowded: {worst['station_name']} ({worst['utilization']:.1%})")
                overview.append(f"  Least crowded: {best['station_name']} ({best['utilization']:.1%})")

        if self.confirmation_predictor:
            pred = self.confirmation_predictor.predict_confirmation(
                "12951", "MMCT", "NDLS", "3A", 15, 20, now.month, now.weekday()
            )
            overview.append(f"\n🎫 TICKETS:")
            overview.append(f"  Rajdhani 3A WL/20 (15 days before): {pred['confirmation_probability']:.0%}")

        if self.railway_network:
            stats = self.railway_network.get_network_stats()
            overview.append(f"\n🛤️ NETWORK: {stats['total_stations']} stations, "
                          f"{stats['total_unique_trains']} trains, "
                          f"{'Connected ✅' if stats['is_connected'] else 'Fragmented ❌'}")

        return "\n".join(overview)

    # ============================================
    # MAIN CHAT METHOD
    # ============================================

    def chat(self, user_query):
        """Process user query and return AI response"""

        # Step 1: Build context from ML models
        context = self._build_context(user_query)

        # Step 2: Try Groq LLM first, fallback to offline
        if groq_client:
            return self._chat_groq(user_query, context)
        else:
            return self._chat_offline(user_query, context)

    def _chat_groq(self, query, context):
        """Chat using Groq API (llama-3.1-8b-instant)"""
        try:
            # Add to history
            self.chat_history.append({"role": "user", "content": query})

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Based on the following REAL railway analytics data, 
answer the user's question accurately and helpfully.

═══════════════════════════════════════
REAL-TIME ANALYTICS DATA:
═══════════════════════════════════════
{context}
═══════════════════════════════════════

USER QUESTION: {query}

Instructions:
- Use ONLY the data provided above
- Include specific numbers, percentages, train names
- Give clear actionable recommendations
- Use emojis and formatting for readability
- Be concise but complete"""}
            ]

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.3,
                max_tokens=1200
            )

            answer = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            print(f"⚠️ Groq API error: {e}")
            return self._chat_offline(query, context)

    def _chat_offline(self, query, context):
        """Offline rule-based response using analytics data"""
        query_lower = query.lower()
        now = datetime.now()
        parts = []

        parts.append(f"🤖 **RailMitra Analysis** ({now.strftime('%d %B, %H:%M')})\n")

        # ---- Delay queries ----
        if any(w in query_lower for w in ["delay", "late", "cascade", "vulnerable", "worst",
                                           "reliable", "best train"]):
            if self.delay_predictor:
                vuln = self.delay_predictor.get_vulnerable_routes(now.month, now.weekday())
                parts.append("**⏱️ Delay Vulnerability Analysis:**\n")
                for _, row in vuln.iterrows():
                    parts.append(f"• **{row['train_name']}** ({row['train_type']}): "
                               f"Avg {row['avg_delay']:.0f} min, "
                               f"Max {row['max_delay']:.0f} min "
                               f"{row['risk_level']}")

                parts.append(f"\n💡 **Recommendations:**")
                parts.append(f"• Rajdhani & Duronto trains have lowest delay risk")
                parts.append(f"• Mail/Express trains accumulate most delays at junctions")
                season = "monsoon" if now.month in [6, 7, 8, 9] else "fog" if now.month in [12, 1] else "festival" if now.month in [10, 11, 12] else "normal"
                parts.append(f"• Current season ({season}) affects delay patterns")

        # ---- Ticket queries ----
        elif any(w in query_lower for w in ["ticket", "confirm", "waitlist", "wl", "book",
                                             "chance", "when should", "advance"]):
            if self.confirmation_predictor:
                parts.append("**🎫 Ticket Confirmation Analysis:**\n")

                samples = [
                    ("12951", "MMCT", "NDLS", "1A", 30, 3, "Rajdhani 1A, WL/3, 30 days"),
                    ("12951", "MMCT", "NDLS", "2A", 20, 10, "Rajdhani 2A, WL/10, 20 days"),
                    ("12951", "MMCT", "NDLS", "3A", 15, 25, "Rajdhani 3A, WL/25, 15 days"),
                    ("12903", "MMCT", "NDLS", "SL", 7, 50, "Golden Temple SL, WL/50, 7 days"),
                ]
                for train, src, dst, cls, days, wl, label in samples:
                    pred = self.confirmation_predictor.predict_confirmation(
                        train, src, dst, cls, days, wl, now.month, now.weekday()
                    )
                    parts.append(f"• {label}: **{pred['confirmation_probability']:.0%}** "
                               f"{pred['confidence_emoji']}")

                advance = self.confirmation_predictor.recommend_booking_advance(
                    "12951", "MMCT", "NDLS", "3A", now.month
                )
                parts.append(f"\n📅 **Booking Advice:** Book **{advance['recommended_days_advance']}+ days** "
                           f"in advance for 80% confirmation (Rajdhani 3A)")

                parts.append(f"\n💡 **Smart Tips:**")
                parts.append(f"• 1A class has highest confirmation rate")
                parts.append(f"• Rajdhani/Duronto confirm faster than Express")
                parts.append(f"• Book 30+ days early during festival season")
                parts.append(f"• Check Central route trains (CSMT) as alternatives")

        # ---- Congestion queries ----
        elif any(w in query_lower for w in ["crowd", "congestion", "busy", "rush", "peak",
                                             "quiet", "best time"]):
            if self.congestion_analyzer:
                parts.append(f"**👥 Congestion Status ({now.hour}:00):**\n")
                congestion = self.congestion_analyzer.get_current_congestion(now.hour)
                for _, row in congestion.head(10).iterrows():
                    emoji = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡",
                             "Low": "🟢"}.get(row["crowd_level"], "⚪")
                    parts.append(f"• {emoji} **{row['station_name']}**: "
                               f"{row['utilization']:.0%} ({row['crowd_level']})")

                hotspots = self.congestion_analyzer.get_hotspots(now.hour)
                if not hotspots.empty:
                    parts.append(f"\n⚠️ **{len(hotspots)} Congestion Hotspots Active!**")

                parts.append(f"\n💡 **Travel Tips:**")
                parts.append(f"• Avoid NDLS & CSMT during 7-9 AM and 5-7 PM")
                parts.append(f"• Best travel window: 10 AM - 2 PM or after 9 PM")
                parts.append(f"• Western route stations are generally less crowded")

        # ---- Route queries ----
        elif any(w in query_lower for w in ["route", "alternative", "path", "network",
                                             "station", "failure", "shutdown"]):
            if self.railway_network:
                parts.append("**🛤️ Route & Network Analysis:**\n")

                routes = self.railway_network.find_alternative_routes("MMCT", "NDLS")
                parts.append("*Mumbai Central → New Delhi:*")
                for r in routes:
                    if "error" not in r:
                        parts.append(f"• Route {r['route_id']}: "
                                   f"{' → '.join(r['path_names'])} ({r['num_stops']} stops)")
                        if r['direct_train_names']:
                            parts.append(f"  🚆 Direct: {', '.join(r['direct_train_names'])}")

                centrality = self.railway_network.get_centrality_analysis()
                parts.append(f"\n🏛️ **Most Critical Stations (if disrupted):**")
                for _, row in centrality.head(5).iterrows():
                    parts.append(f"• {row['station_name']}: Vulnerability {row['vulnerability_score']:.3f} "
                               f"({row['trains_through']} trains)")

        # ---- General query ----
        else:
            parts.append("**Welcome to RailMitra!** I can help with:\n")
            parts.append("• 🕐 **Delays** — \"Which trains have most delays?\"")
            parts.append("• 🎫 **Tickets** — \"Will my WL/25 confirm?\"")
            parts.append("• 👥 **Crowds** — \"How busy is Delhi station?\"")
            parts.append("• 🛤️ **Routes** — \"Best route Mumbai to Delhi?\"")
            parts.append(f"\n{self._get_general_overview()}")

        return "\n".join(parts)


# ============================================
# MAIN - Test the assistant
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 RailMitra - AI Railway Assistant (Groq Powered)")
    print("=" * 60)

    # Load all ML models
    print("\n📦 Loading ML models (this takes ~30 seconds)...")
    print("-" * 40)

    from models.delay_predictor import DelayPredictor
    from models.confirmation_predictor import ConfirmationPredictor
    from models.congestion_model import CongestionAnalyzer
    from graph.railway_graph import RailwayNetwork

    delay_pred = DelayPredictor()
    print()
    conf_pred = ConfirmationPredictor()
    print()
    congestion = CongestionAnalyzer()
    print()
    network = RailwayNetwork()

    # Create assistant with all models
    print("\n" + "=" * 60)
    print("🤖 Creating RailMitra Assistant...")
    print("=" * 60)

    assistant = RailwayAssistant(
        delay_predictor=delay_pred,
        confirmation_predictor=conf_pred,
        congestion_analyzer=congestion,
        railway_network=network
    )

    # ---- Automated Test Queries ----
    test_queries = [
        # Operator queries
        "Which stations are most vulnerable to cascading delays today?",
        "Which routes are predicted to experience congestion tomorrow?",

        # Passenger queries
        "What are the chances my waitlisted ticket on Rajdhani will get confirmed?",
        "When should I book my ticket to maximize confirmation chances?",
        "Which route currently has the least crowding?",
        "What are alternative routes from Mumbai to Delhi?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'━' * 60}")
        print(f"❓ QUERY {i}: {query}")
        print(f"{'━' * 60}")

        response = assistant.chat(query)
        print(f"\n{response}")
        print()

    # ---- Interactive Chat Mode ----
    print("\n" + "=" * 60)
    print("💬 INTERACTIVE MODE")
    print("   Type your questions below!")
    print("   Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q", "bye"]:
                print("\n👋 Thank you for using RailMitra! Safe travels! 🚂")
                break

            response = assistant.chat(user_input)
            print(f"\n🤖 RailMitra:\n{response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
