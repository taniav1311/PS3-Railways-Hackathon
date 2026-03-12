import networkx as nx
import pandas as pd
import numpy as np
import os
import sys

# This makes sure Python can find the 'data' folder
# regardless of where you run the script from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


class RailwayNetwork:
    def __init__(self):
        self.G = nx.DiGraph()

        # Load data using absolute paths
        data_dir = os.path.join(PROJECT_ROOT, "data")

        print("📂 Loading data files...")
        self.stations = pd.read_csv(os.path.join(data_dir, "stations.csv"))
        print(f"  ✅ Loaded {len(self.stations)} stations")

        self.trains = pd.read_csv(os.path.join(data_dir, "trains.csv"))
        print(f"  ✅ Loaded {len(self.trains)} trains")

        self.schedules = pd.read_csv(os.path.join(data_dir, "schedules.csv"))
        print(f"  ✅ Loaded {len(self.schedules)} schedule entries")

        print("\n🔨 Building railway network graph...")
        self._build_graph()
        print(f"  ✅ Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def _build_graph(self):
        """Build the railway network graph"""

        # ---- Add station nodes ----
        for _, stn in self.stations.iterrows():
            self.G.add_node(
                stn["code"],
                name=stn["name"],
                lat=stn["lat"],
                lon=stn["lon"],
                zone=stn["zone"],
                type=stn["type"],
                daily_footfall=stn["daily_footfall"],
                platforms=stn["platforms"]
            )

        # ---- Add edges from train routes ----
        for _, train in self.trains.iterrows():
            # Handle stations column - could be string or list
            stations = train["stations"]
            if isinstance(stations, str):
                stations = eval(stations)

            for i in range(len(stations) - 1):
                src = stations[i]
                dst = stations[i + 1]

                # Calculate approximate distance between consecutive stations
                src_data = self.stations[self.stations["code"] == src]
                dst_data = self.stations[self.stations["code"] == dst]

                distance = 0
                if not src_data.empty and not dst_data.empty:
                    # Haversine-like approximation
                    lat1, lon1 = src_data.iloc[0]["lat"], src_data.iloc[0]["lon"]
                    lat2, lon2 = dst_data.iloc[0]["lat"], dst_data.iloc[0]["lon"]
                    distance = int(((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111)

                if self.G.has_edge(src, dst):
                    # Edge already exists - add this train to the list
                    self.G[src][dst]["trains"].append(str(train["number"]))
                    self.G[src][dst]["num_trains"] += 1
                    # More trains = better connectivity = lower weight
                    self.G[src][dst]["weight"] = max(1, 10 - self.G[src][dst]["num_trains"])
                else:
                    # Create new edge
                    self.G.add_edge(
                        src, dst,
                        trains=[str(train["number"])],
                        num_trains=1,
                        weight=10,
                        distance_km=distance,
                        route=train.get("route", "unknown")
                    )

    # ============================================
    # CENTRALITY & VULNERABILITY ANALYSIS
    # ============================================

    def get_centrality_analysis(self):
        """Identify critical stations using centrality metrics"""
        undirected = self.G.to_undirected()

        degree = nx.degree_centrality(undirected)
        betweenness = nx.betweenness_centrality(undirected)
        closeness = nx.closeness_centrality(undirected)

        # Try PageRank on directed graph
        try:
            pagerank = nx.pagerank(self.G)
        except:
            pagerank = {node: 0 for node in self.G.nodes()}

        results = []
        for node in self.G.nodes():
            node_data = self.G.nodes[node]

            # Count total trains through this station
            trains_through = set()
            for u, v, data in self.G.edges(data=True):
                if u == node or v == node:
                    trains_through.update(data.get("trains", []))

            vulnerability_score = (
                0.3 * degree.get(node, 0) +
                0.3 * betweenness.get(node, 0) +
                0.2 * closeness.get(node, 0) +
                0.2 * pagerank.get(node, 0)
            )

            results.append({
                "station_code": node,
                "station_name": node_data.get("name", node),
                "type": node_data.get("type", "unknown"),
                "zone": node_data.get("zone", "unknown"),
                "daily_footfall": node_data.get("daily_footfall", 0),
                "platforms": node_data.get("platforms", 0),
                "degree_centrality": round(degree.get(node, 0), 4),
                "betweenness_centrality": round(betweenness.get(node, 0), 4),
                "closeness_centrality": round(closeness.get(node, 0), 4),
                "pagerank": round(pagerank.get(node, 0), 4),
                "trains_through": len(trains_through),
                "vulnerability_score": round(vulnerability_score, 4)
            })

        return pd.DataFrame(results).sort_values("vulnerability_score", ascending=False)

    # ============================================
    # ROUTE FINDING
    # ============================================

    def find_alternative_routes(self, source, destination, k=3):
        """Find k-shortest paths between two stations"""
        try:
            paths = list(nx.shortest_simple_paths(self.G, source, destination, weight="weight"))
            results = []

            for i, path in enumerate(paths[:k]):
                # Get station names for the path
                path_names = []
                for code in path:
                    name = self.G.nodes[code].get("name", code)
                    path_names.append(name)

                # Find trains that cover the FULL path directly
                direct_trains = None
                for j in range(len(path) - 1):
                    if self.G.has_edge(path[j], path[j + 1]):
                        edge_trains = set(self.G[path[j]][path[j + 1]].get("trains", []))
                        if direct_trains is None:
                            direct_trains = edge_trains
                        else:
                            direct_trains = direct_trains & edge_trains
                    else:
                        direct_trains = set()
                        break

                if direct_trains is None:
                    direct_trains = set()

                # Calculate total distance
                total_distance = 0
                for j in range(len(path) - 1):
                    if self.G.has_edge(path[j], path[j + 1]):
                        total_distance += self.G[path[j]][path[j + 1]].get("distance_km", 0)

                # Get train names
                trains_df = self.trains
                direct_train_names = []
                for t_num in direct_trains:
                    t_info = trains_df[trains_df["number"].astype(str) == str(t_num)]
                    if not t_info.empty:
                        direct_train_names.append(f"{t_info.iloc[0]['name']} ({t_num})")
                    else:
                        direct_train_names.append(t_num)

                results.append({
                    "route_id": i + 1,
                    "path_codes": path,
                    "path_names": path_names,
                    "num_stops": len(path),
                    "total_distance_km": total_distance,
                    "direct_trains": list(direct_trains),
                    "direct_train_names": direct_train_names,
                    "has_direct_train": len(direct_trains) > 0
                })

            return results

        except nx.NetworkXNoPath:
            return [{"error": f"No path found between {source} and {destination}"}]
        except nx.NodeNotFound as e:
            return [{"error": f"Station not found: {e}"}]

    # ============================================
    # NETWORK STATISTICS
    # ============================================

    def get_network_stats(self):
        """Get overall network statistics"""
        undirected = self.G.to_undirected()

        stats = {
            "total_stations": self.G.number_of_nodes(),
            "total_connections": self.G.number_of_edges(),
            "network_density": round(nx.density(undirected), 4),
            "is_connected": nx.is_connected(undirected),
            "num_components": nx.number_connected_components(undirected),
            "avg_clustering": round(nx.average_clustering(undirected), 4),
        }

        if nx.is_connected(undirected):
            stats["diameter"] = nx.diameter(undirected)
            stats["avg_shortest_path"] = round(nx.average_shortest_path_length(undirected), 2)
        else:
            stats["diameter"] = "N/A (disconnected)"
            stats["avg_shortest_path"] = "N/A"

        # Count total unique trains
        all_trains = set()
        for u, v, data in self.G.edges(data=True):
            all_trains.update(data.get("trains", []))
        stats["total_unique_trains"] = len(all_trains)

        # Find most connected station
        degree_dict = dict(undirected.degree())
        most_connected = max(degree_dict, key=degree_dict.get)
        stats["most_connected_station"] = f"{self.G.nodes[most_connected].get('name', most_connected)} ({degree_dict[most_connected]} connections)"

        return stats

    # ============================================
    # STATION FAILURE SIMULATION
    # ============================================

    def simulate_station_failure(self, station_code):
        """Simulate what happens if a station goes down"""

        station_name = self.G.nodes[station_code].get("name", station_code)

        # Find all affected trains
        affected_trains = set()
        affected_edges = []
        for u, v, data in self.G.edges(data=True):
            if u == station_code or v == station_code:
                affected_trains.update(data.get("trains", []))
                affected_edges.append((u, v))

        # Get train names
        affected_train_details = []
        for t_num in affected_trains:
            t_info = self.trains[self.trains["number"].astype(str) == str(t_num)]
            if not t_info.empty:
                affected_train_details.append({
                    "number": t_num,
                    "name": t_info.iloc[0]["name"],
                    "type": t_info.iloc[0]["type"]
                })

        # Remove station and check connectivity
        G_copy = self.G.to_undirected().copy()
        original_components = nx.number_connected_components(G_copy)

        G_copy.remove_node(station_code)
        new_components = nx.number_connected_components(G_copy)
        components = list(nx.connected_components(G_copy))

        # Calculate impact score
        impact_score = (
            len(affected_trains) * 0.4 +
            (new_components - original_components) * 10 * 0.3 +
            len(affected_edges) * 0.3
        )

        # Find isolated stations
        isolated_stations = []
        if new_components > original_components:
            for comp in components:
                if len(comp) <= 2:
                    for stn in comp:
                        isolated_stations.append(self.G.nodes[stn].get("name", stn))

        return {
            "station_removed": station_code,
            "station_name": station_name,
            "affected_trains": list(affected_trains),
            "affected_train_details": affected_train_details,
            "num_affected_trains": len(affected_trains),
            "edges_removed": len(affected_edges),
            "original_components": original_components,
            "network_fragments": new_components,
            "fragment_sizes": sorted([len(c) for c in components], reverse=True),
            "network_still_connected": new_components == original_components,
            "isolated_stations": isolated_stations,
            "impact_score": round(impact_score, 2)
        }

    # ============================================
    # COMMUNITY DETECTION
    # ============================================

    def detect_communities(self):
        """Detect station communities/clusters in the network"""
        undirected = self.G.to_undirected()

        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(undirected))

            results = []
            for i, community in enumerate(communities):
                stations_in_community = []
                for stn in community:
                    stations_in_community.append({
                        "code": stn,
                        "name": self.G.nodes[stn].get("name", stn),
                        "type": self.G.nodes[stn].get("type", "unknown")
                    })

                results.append({
                    "community_id": i + 1,
                    "size": len(community),
                    "stations": stations_in_community,
                    "station_codes": list(community)
                })

            return results
        except Exception as e:
            print(f"  ⚠️ Community detection error: {e}")
            return []

    # ============================================
    # EDGE ANALYSIS
    # ============================================

    def get_busiest_corridors(self, top_n=10):
        """Find the busiest rail corridors (edges with most trains)"""
        corridors = []

        for u, v, data in self.G.edges(data=True):
            u_name = self.G.nodes[u].get("name", u)
            v_name = self.G.nodes[v].get("name", v)

            corridors.append({
                "from_code": u,
                "from_name": u_name,
                "to_code": v,
                "to_name": v_name,
                "num_trains": data.get("num_trains", 0),
                "trains": data.get("trains", []),
                "distance_km": data.get("distance_km", 0),
                "route": data.get("route", "unknown")
            })

        df = pd.DataFrame(corridors).sort_values("num_trains", ascending=False)
        return df.head(top_n)

    # ============================================
    # HELPER: Get all edges for visualization
    # ============================================

    def get_edges_for_visualization(self):
        """Return edges with coordinates for map plotting"""
        edges = []
        for u, v, data in self.G.edges(data=True):
            u_data = self.G.nodes[u]
            v_data = self.G.nodes[v]

            edges.append({
                "from_code": u,
                "from_name": u_data.get("name", u),
                "from_lat": u_data.get("lat", 0),
                "from_lon": u_data.get("lon", 0),
                "to_code": v,
                "to_name": v_data.get("name", v),
                "to_lat": v_data.get("lat", 0),
                "to_lon": v_data.get("lon", 0),
                "num_trains": data.get("num_trains", 0),
                "route": data.get("route", "unknown")
            })

        return pd.DataFrame(edges)


# ============================================
# MAIN - Run this to test the graph
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚂 Railway Network Graph - Mumbai to Delhi Corridor")
    print("=" * 60)

    # Build the network
    rn = RailwayNetwork()

    # ---- 1. Network Statistics ----
    print("\n" + "=" * 60)
    print("📊 NETWORK STATISTICS")
    print("=" * 60)
    stats = rn.get_network_stats()
    for key, value in stats.items():
        print(f"  {key:30s}: {value}")

    # ---- 2. Centrality Analysis ----
    print("\n" + "=" * 60)
    print("🏛️ STATION VULNERABILITY RANKING (Top 10)")
    print("=" * 60)
    centrality = rn.get_centrality_analysis()
    print(centrality[["station_code", "station_name", "type", "trains_through",
                       "degree_centrality", "betweenness_centrality",
                       "vulnerability_score"]].head(10).to_string(index=False))

    # ---- 3. Alternative Routes ----
    print("\n" + "=" * 60)
    print("🛤️ ALTERNATIVE ROUTES: Mumbai Central → New Delhi")
    print("=" * 60)
    routes = rn.find_alternative_routes("MMCT", "NDLS", k=5)
    for r in routes:
        if "error" in r:
            print(f"  ❌ {r['error']}")
        else:
            path_str = " → ".join(r["path_names"])
            print(f"\n  Route {r['route_id']}: {path_str}")
            print(f"    Stops: {r['num_stops']} | Distance: ~{r['total_distance_km']} km")
            if r["has_direct_train"]:
                print(f"    🚆 Direct trains: {', '.join(r['direct_train_names'])}")
            else:
                print(f"    ⚠️ No direct train - requires connection")

    # ---- 4. Routes from CSMT ----
    print("\n" + "=" * 60)
    print("🛤️ ALTERNATIVE ROUTES: Mumbai CSMT → New Delhi")
    print("=" * 60)
    routes2 = rn.find_alternative_routes("CSMT", "NDLS", k=3)
    for r in routes2:
        if "error" not in r:
            path_str = " → ".join(r["path_names"])
            print(f"\n  Route {r['route_id']}: {path_str}")
            print(f"    Stops: {r['num_stops']}")
            if r["has_direct_train"]:
                print(f"    🚆 Direct trains: {', '.join(r['direct_train_names'])}")

    # ---- 5. Busiest Corridors ----
    print("\n" + "=" * 60)
    print("🔥 BUSIEST RAIL CORRIDORS")
    print("=" * 60)
    corridors = rn.get_busiest_corridors()
    print(corridors[["from_name", "to_name", "num_trains", "route"]].to_string(index=False))

    # ---- 6. Station Failure Simulations ----
    print("\n" + "=" * 60)
    print("💥 STATION FAILURE SIMULATIONS")
    print("=" * 60)

    critical_stations = ["NDLS", "BRC", "KOTA", "MTJ", "BSL"]
    for stn in critical_stations:
        result = rn.simulate_station_failure(stn)
        status = "❌ BREAKS NETWORK" if not result["network_still_connected"] else "✅ Network survives"
        print(f"\n  💥 Remove {result['station_name']} ({stn}):")
        print(f"     Affected trains: {result['num_affected_trains']}")
        print(f"     Network fragments: {result['network_fragments']}")
        print(f"     Fragment sizes: {result['fragment_sizes']}")
        print(f"     Impact score: {result['impact_score']}")
        print(f"     Status: {status}")

    # ---- 7. Community Detection ----
    print("\n" + "=" * 60)
    print("🏘️ NETWORK COMMUNITIES")
    print("=" * 60)
    communities = rn.detect_communities()
    for comm in communities:
        station_names = [s["name"] for s in comm["stations"]]
        print(f"\n  Community {comm['community_id']} ({comm['size']} stations):")
        print(f"    {', '.join(station_names)}")

    # ---- Done ----
    print("\n" + "=" * 60)
    print("✅ PHASE 2 COMPLETE - Railway Network Graph Built!")
    print("=" * 60)
    print("\n🚀 You can now proceed to Phase 3 (ML Models)!")