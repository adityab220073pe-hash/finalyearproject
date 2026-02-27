"""
V-SGC Backend: VRP-Aware Seed-and-Graft Clustering + TSPTW Routing
Implements the exact methodology from ME4098D project:
  Phase 1: K-Means seeding → MILP Re-clustering (V-SGC)
  Phase 2: TSPTW routing per cluster (scipy MILP / CBC solver)
"""

import json
import math
import time
import traceback
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.spatial.distance import cdist

app = Flask(__name__, static_folder="static")

# ─────────────────────────────────────────────
#  CORS (manual, no flask-cors needed)
# ─────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/api/solve", methods=["OPTIONS"])
def options_solve():
    return jsonify({}), 200


# ─────────────────────────────────────────────
#  DISTANCE UTILITIES
# ─────────────────────────────────────────────
def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in km — used when coords are lat/lon."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def build_dist_matrix(points, use_haversine=False):
    """Build NxN distance matrix."""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if use_haversine:
                    D[i][j] = haversine(points[i][0], points[i][1],
                                         points[j][0], points[j][1])
                else:
                    D[i][j] = euclidean(points[i], points[j])
    return D


# ─────────────────────────────────────────────
#  PHASE 0: K-MEANS SEEDING
# ─────────────────────────────────────────────
def kmeans_seeds(coords, n_seeds, n_iter=100):
    """
    Simple K-Means to find candidate cluster centroids.
    Returns indices of the nearest real points to each centroid.
    Mirrors the project's Step 2: K-Means Clustering.
    """
    coords = np.array(coords)
    n = len(coords)
    n_seeds = min(n_seeds, n)

    # Initialize centroids via K-Means++ style
    rng = np.random.default_rng(42)
    centroids = [coords[rng.integers(n)]]
    for _ in range(n_seeds - 1):
        dists = np.array([min(np.linalg.norm(c - x)**2 for x in centroids) for c in coords])
        probs = dists / dists.sum()
        centroids.append(coords[rng.choice(n, p=probs)])
    centroids = np.array(centroids)

    for _ in range(n_iter):
        # Assign
        dists = cdist(coords, centroids)
        labels = np.argmin(dists, axis=1)
        # Update
        new_centroids = []
        for k in range(n_seeds):
            members = coords[labels == k]
            new_centroids.append(members.mean(axis=0) if len(members) > 0 else centroids[k])
        new_centroids = np.array(new_centroids)
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Find nearest real point to each centroid (exemplar candidates)
    candidate_indices = []
    for c in centroids:
        dists_to_c = np.linalg.norm(coords - c, axis=1)
        candidate_indices.append(int(np.argmin(dists_to_c)))

    # Deduplicate
    candidate_indices = list(set(candidate_indices))
    return candidate_indices


# ─────────────────────────────────────────────
#  PHASE 1: V-SGC MILP RE-CLUSTERING
# ─────────────────────────────────────────────
def vsgc_milp_cluster(customers, depot, candidate_indices, p,
                       vehicle_capacity, M_cap=1000, M_time=0.1, M_depot=0.5,
                       time_limit=60):
    """
    VRP-Aware Seed-and-Graft Clustering (V-SGC) MILP.

    Objective:
      min Σ_ij d_ij * x_ij          (compactness)
        + Σ_j M_cap * C_j_over      (capacity penalty)
        + Σ_j M_time * T_j_spread   (time spread penalty)
        + Σ_j M_depot * d_0j * y_j  (depot proximity penalty)

    Decision variables (flattened):
      y_j ∈ {0,1}   : j is selected as exemplar   [len = |S|]
      x_ij ∈ {0,1}  : customer i assigned to j     [len = N*|S|]
      C_j_over ≥ 0  : capacity overflow of cluster j
      T_j_spread ≥ 0: time spread of cluster j
      l_j_max ≥ 0   : max latest time in cluster j (linearisation aux)
      e_j_min ≥ 0   : min earliest time in cluster j (linearisation aux)

    Uses scipy.optimize.milp with CBC solver.
    Falls back to greedy if MILP times out or fails.
    """
    N = len(customers)
    S_idx = candidate_indices          # indices into customers list
    ns = len(S_idx)

    if ns == 0 or N == 0:
        return greedy_cluster(customers, depot, p, vehicle_capacity)

    # ── Coordinate arrays ──
    cust_coords = np.array([[c['x'], c['y']] for c in customers])
    cand_coords = cust_coords[S_idx]
    depot_coord = np.array([depot['x'], depot['y']])
    demands = np.array([c['demand'] for c in customers])
    ei = np.array([c.get('time_earliest', 0) for c in customers], dtype=float)
    li = np.array([c.get('time_latest', 100) for c in customers], dtype=float)

    # ── Distance matrices ──
    d_ij = np.zeros((N, ns))          # customer i → candidate j
    for i in range(N):
        for jj, j in enumerate(S_idx):
            d_ij[i, jj] = np.linalg.norm(cust_coords[i] - cust_coords[j])

    d_0j = np.array([np.linalg.norm(cand_coords[jj] - depot_coord)
                     for jj in range(ns)])

    U_e = max(ei) - min(ei) + 1      # upper bound for time linearisation
    U_l = max(li) - min(li) + 1

    # ── Variable layout ──
    # [y_0..y_{ns-1}, x_00..x_{N-1,ns-1}, C_0..C_{ns-1},
    #  T_0..T_{ns-1}, lmax_0..lmax_{ns-1}, emin_0..emin_{ns-1}]
    n_y   = ns
    n_x   = N * ns
    n_c   = ns   # C_over
    n_t   = ns   # T_spread
    n_lm  = ns   # l_max
    n_em  = ns   # e_min
    n_vars = n_y + n_x + n_c + n_t + n_lm + n_em

    iy   = lambda j:          j
    ix   = lambda i, jj:      n_y + i*ns + jj
    ic   = lambda j:          n_y + n_x + j
    it   = lambda j:          n_y + n_x + n_c + j
    ilm  = lambda j:          n_y + n_x + n_c + n_t + j
    iem  = lambda j:          n_y + n_x + n_c + n_t + n_lm + j

    # ── Objective ──
    c_obj = np.zeros(n_vars)
    for i in range(N):
        for jj in range(ns):
            c_obj[ix(i,jj)] = d_ij[i,jj]
    for jj in range(ns):
        c_obj[ic(jj)]  = M_cap
        c_obj[it(jj)]  = M_time
        c_obj[iy(jj)]  += M_depot * d_0j[jj]

    # ── Integrality ──
    integrality = np.zeros(n_vars)
    for jj in range(ns):
        integrality[iy(jj)] = 1       # y binary
    for i in range(N):
        for jj in range(ns):
            integrality[ix(i,jj)] = 1  # x binary

    # ── Bounds ──
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    # Continuous vars are ≥ 0, no upper bound
    for jj in range(ns):
        ub[ic(jj)]  = np.inf
        ub[it(jj)]  = np.inf
        ub[ilm(jj)] = np.inf
        ub[iem(jj)] = np.inf

    # ── Constraints ──
    A_rows, b_lo, b_hi = [], [], []

    def add(row_dict, lo, hi):
        row = np.zeros(n_vars)
        for k, v in row_dict.items():
            row[k] = v
        A_rows.append(row)
        b_lo.append(lo)
        b_hi.append(hi)

    # (2) Σ_j y_j = p
    r = {iy(jj): 1 for jj in range(ns)}
    add(r, p, p)

    # (3) Σ_j x_ij = 1  ∀i
    for i in range(N):
        r = {ix(i,jj): 1 for jj in range(ns)}
        add(r, 1, 1)

    # (4) x_ij ≤ y_j  ∀i,j
    for i in range(N):
        for jj in range(ns):
            add({ix(i,jj): 1, iy(jj): -1}, -np.inf, 0)

    # (5) Σ_i q_i * x_ij - Q ≤ C_j_over  ∀j
    for jj in range(ns):
        r = {ix(i,jj): demands[i] for i in range(N)}
        r[ic(jj)] = -1
        add(r, -np.inf, vehicle_capacity)

    # (6)-(9) Time spread linearisation
    for jj in range(ns):
        # T_j_spread ≥ l_j_max - e_j_min
        add({it(jj): 1, ilm(jj): -1, iem(jj): 1}, 0, np.inf)
        # l_j_max ≥ l_i * x_ij  → l_j_max - l_i*x_ij ≥ 0
        for i in range(N):
            add({ilm(jj): 1, ix(i,jj): -li[i]}, 0, np.inf)
        # e_j_min ≤ e_i + U_e*(1 - x_ij)
        for i in range(N):
            add({iem(jj): 1, ix(i,jj): -U_e}, -np.inf, ei[i])
        # e_j_min ≥ 0 (already in bounds)

    if not A_rows:
        return greedy_cluster(customers, depot, p, vehicle_capacity)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, b_lo, b_hi)
    bounds = Bounds(lb, ub)

    try:
        t0 = time.time()
        result = milp(c_obj,
                      constraints=constraints,
                      integrality=integrality,
                      bounds=bounds,
                      options={"time_limit": time_limit, "disp": False})
        elapsed = time.time() - t0

        if result.status in (0, 3) and result.x is not None:
            xv = result.x
            clusters = {}
            for jj in range(ns):
                if xv[iy(jj)] > 0.5:
                    assigned = [i for i in range(N) if xv[ix(i,jj)] > 0.5]
                    if assigned:
                        clusters[jj] = assigned

            if not clusters:
                return greedy_cluster(customers, depot, p, vehicle_capacity), elapsed

            result_clusters = []
            for jj, members in clusters.items():
                result_clusters.append({
                    "exemplar_idx": S_idx[jj],
                    "members": members,
                    "demand": int(sum(demands[m] for m in members)),
                    "depot_idx": None
                })
            return result_clusters, elapsed

        # MILP failed / infeasible → greedy fallback
        return greedy_cluster(customers, depot, p, vehicle_capacity), time.time()-t0

    except Exception as e:
        return greedy_cluster(customers, depot, p, vehicle_capacity), 0.0


def greedy_cluster(customers, depot, p, vehicle_capacity):
    """Greedy fallback clustering when MILP fails."""
    demands = [c['demand'] for c in customers]
    remaining = list(range(len(customers)))
    clusters = []
    depot_coord = np.array([depot['x'], depot['y']])

    # Sort by distance from depot
    remaining.sort(key=lambda i: np.linalg.norm(
        np.array([customers[i]['x'], customers[i]['y']]) - depot_coord))

    while remaining and len(clusters) < p:
        cap = 0
        members = []
        current = depot_coord.copy()
        unvisited = list(remaining)

        while unvisited:
            best_i, best_d = None, np.inf
            for idx in unvisited:
                c = customers[idx]
                if cap + c['demand'] <= vehicle_capacity:
                    d = np.linalg.norm(np.array([c['x'], c['y']]) - current)
                    if d < best_d:
                        best_d, best_i = d, idx
            if best_i is None:
                break
            members.append(best_i)
            cap += demands[best_i]
            current = np.array([customers[best_i]['x'], customers[best_i]['y']])
            remaining.remove(best_i)
            unvisited.remove(best_i)

        if members:
            clusters.append({
                "exemplar_idx": members[0],
                "members": members,
                "demand": int(sum(demands[m] for m in members)),
                "depot_idx": None
            })

    # Assign leftovers to last cluster
    if remaining:
        clusters[-1]["members"].extend(remaining)
        clusters[-1]["demand"] += int(sum(demands[m] for m in remaining))

    return clusters


# ─────────────────────────────────────────────
#  ASSIGN CLUSTERS TO DEPOTS
# ─────────────────────────────────────────────
def assign_clusters_to_depots(clusters, customers, depots):
    """
    Step 4: For each cluster's exemplar, find nearest depot.
    Mirrors: cluster assigned to depot with minimum distance to exemplar.
    """
    for cl in clusters:
        ex_idx = cl["exemplar_idx"]
        ex_coord = np.array([customers[ex_idx]['x'], customers[ex_idx]['y']])
        best_d, best_di = np.inf, 0
        for di, dep in enumerate(depots):
            d = np.linalg.norm(ex_coord - np.array([dep['x'], dep['y']]))
            if d < best_d:
                best_d, best_di = d, di
        cl["depot_idx"] = best_di
    return clusters


# ─────────────────────────────────────────────
#  PHASE 2: TSPTW ROUTING
# ─────────────────────────────────────────────
def tsptw_milp(cluster_stop_indices, customers, depot, speed=30, time_limit=30):
    """
    TSPTW: Traveling Salesman Problem with Time Windows.
    Minimizes total travel distance.
    Each customer visited exactly once, start and end at depot.
    Big-M constraints enforce time window feasibility.
    Mirrors Phase 2 formulation from ME4098D.

    Returns ordered list of customer indices.
    Falls back to 2-opt nearest-neighbour if MILP fails.
    """
    stops = cluster_stop_indices
    m = len(stops)

    if m == 0:
        return []
    if m == 1:
        return stops

    # Node 0 = depot, nodes 1..m = customers in cluster
    nodes = [depot] + [customers[i] for i in stops]
    n_nodes = len(nodes)   # m+1

    coords = np.array([[nd['x'], nd['y']] for nd in nodes])
    service_times = np.array([0] + [nodes[k].get('service_time', 0) for k in range(1, n_nodes)])
    ei = np.array([0] + [nodes[k].get('time_earliest', 0) for k in range(1, n_nodes)], dtype=float)
    li = np.array([1e6] + [nodes[k].get('time_latest', 1e6) for k in range(1, n_nodes)], dtype=float)

    # Distance matrix
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                D[i][j] = np.linalg.norm(coords[i] - coords[j])

    M_big = max(li) + np.max(D) / (speed/60) + max(service_times) + 1

    # ── Variable layout ──
    # x_ij ∈ {0,1}  i,j ∈ [0..n_nodes-1], i≠j  → route arcs
    # t_i  ≥ 0      i ∈ [0..n_nodes-1]          → arrival times

    def ix(i, j):
        # flatten: skip diagonal
        return i * (n_nodes - 1) + (j if j < i else j - 1)

    n_x = n_nodes * (n_nodes - 1)
    n_t = n_nodes
    n_vars = n_x + n_t

    def it(i):
        return n_x + i

    # ── Objective: min Σ d_ij * x_ij ──
    c_obj = np.zeros(n_vars)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                c_obj[ix(i, j)] = D[i][j]

    # ── Integrality ──
    integrality = np.zeros(n_vars)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                integrality[ix(i, j)] = 1

    # ── Bounds ──
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    for i in range(n_nodes):
        ub[it(i)] = np.inf

    # ── Constraints ──
    A_rows, b_lo, b_hi = [], [], []

    def add(row_dict, lo, hi):
        row = np.zeros(n_vars)
        for k, v in row_dict.items():
            row[k] = v
        A_rows.append(row)
        b_lo.append(lo)
        b_hi.append(hi)

    # (14) Σ_i x_ij = 1  ∀j ∈ customers (exactly one predecessor)
    for j in range(1, n_nodes):
        r = {ix(i, j): 1 for i in range(n_nodes) if i != j}
        add(r, 1, 1)

    # (15) Σ_j x_ij = 1  ∀i ∈ customers (exactly one successor)
    for i in range(1, n_nodes):
        r = {ix(i, j): 1 for j in range(n_nodes) if j != i}
        add(r, 1, 1)

    # (16) Σ_j x_0j = 1  depot leaves exactly once
    r = {ix(0, j): 1 for j in range(1, n_nodes)}
    add(r, 1, 1)

    # (17) Σ_i x_i0 = 1  depot entered exactly once
    r = {ix(i, 0): 1 for i in range(1, n_nodes)}
    add(r, 1, 1)

    # (18) Time propagation + subtour elimination (Big-M)
    # t_j ≥ t_i + s_i + d_ij - M*(1 - x_ij)   ∀i,j≠0, i≠j
    for i in range(n_nodes):
        for j in range(1, n_nodes):
            if i != j:
                travel = D[i][j] / (speed / 60)
                # t_j - t_i - t_travel - M*x_ij ≥ s_i - M
                r = {it(j): 1, it(i): -1, ix(i, j): -M_big}
                add(r, service_times[i] + travel - M_big, np.inf)

    # (19) Time window: e_i ≤ t_i ≤ l_i
    for i in range(1, n_nodes):
        add({it(i): 1}, ei[i], li[i])

    if not A_rows or len(A_rows) > 5000:
        # Too large for MILP — use 2-opt fallback
        return two_opt_nn(stops, customers, depot)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, b_lo, b_hi)
    bounds = Bounds(lb, ub)

    try:
        result = milp(c_obj,
                      constraints=constraints,
                      integrality=integrality,
                      bounds=bounds,
                      options={"time_limit": time_limit, "disp": False})

        if result.status in (0, 3) and result.x is not None:
            xv = result.x
            # Reconstruct route from x variables
            route = []
            current = 0
            visited = set()
            for _ in range(m):
                for j in range(n_nodes):
                    if j != current and j not in visited and xv[ix(current, j)] > 0.5:
                        if j != 0:
                            route.append(stops[j - 1])
                            visited.add(j)
                        current = j
                        break
                else:
                    break

            if len(route) == m:
                return route

        # Fallback
        return two_opt_nn(stops, customers, depot)

    except Exception:
        return two_opt_nn(stops, customers, depot)


def two_opt_nn(stop_indices, customers, depot):
    """Nearest-neighbour + 2-opt improvement as TSPTW fallback."""
    if not stop_indices:
        return []

    # Nearest neighbour
    unvisited = list(stop_indices)
    route = []
    current_coord = np.array([depot['x'], depot['y']])

    while unvisited:
        dists = [np.linalg.norm(
            np.array([customers[i]['x'], customers[i]['y']]) - current_coord)
            for i in unvisited]
        best = unvisited[int(np.argmin(dists))]
        route.append(best)
        current_coord = np.array([customers[best]['x'], customers[best]['y']])
        unvisited.remove(best)

    # 2-opt
    def route_dist(r):
        d = np.linalg.norm(np.array([customers[r[0]]['x'], customers[r[0]]['y']]) -
                           np.array([depot['x'], depot['y']]))
        for k in range(len(r)-1):
            d += np.linalg.norm(np.array([customers[r[k+1]]['x'], customers[r[k+1]]['y']]) -
                                np.array([customers[r[k]]['x'], customers[r[k]]['y']]))
        d += np.linalg.norm(np.array([customers[r[-1]]['x'], customers[r[-1]]['y']]) -
                            np.array([depot['x'], depot['y']]))
        return d

    improved = True
    while improved:
        improved = False
        for i in range(len(route)-1):
            for j in range(i+2, len(route)):
                new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                if route_dist(new_route) < route_dist(route) - 1e-6:
                    route = new_route
                    improved = True
    return route


# ─────────────────────────────────────────────
#  COMPUTE ROUTE DISTANCE
# ─────────────────────────────────────────────
def route_total_distance(ordered_stops, customers, depot):
    if not ordered_stops:
        return 0.0
    d = np.linalg.norm(
        np.array([customers[ordered_stops[0]]['x'], customers[ordered_stops[0]]['y']]) -
        np.array([depot['x'], depot['y']]))
    for k in range(len(ordered_stops)-1):
        d += np.linalg.norm(
            np.array([customers[ordered_stops[k+1]]['x'], customers[ordered_stops[k+1]]['y']]) -
            np.array([customers[ordered_stops[k]]['x'], customers[ordered_stops[k]]['y']]))
    d += np.linalg.norm(
        np.array([customers[ordered_stops[-1]]['x'], customers[ordered_stops[-1]]['y']]) -
        np.array([depot['x'], depot['y']]))
    return float(d)


# ─────────────────────────────────────────────
#  MAIN API ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/solve", methods=["POST"])
def solve():
    try:
        data = request.get_json()

        # Parse input
        customers   = data.get("customers", [])      # [{x,y,demand,label,...}]
        depots      = data.get("depots", [{"x":50,"y":50,"name":"Depot"}])
        num_vehicles= int(data.get("num_vehicles", 3))
        capacity    = int(data.get("vehicle_capacity", 100))
        speed       = float(data.get("speed_kmh", 30))
        milp_time   = float(data.get("milp_time_limit", 30))

        if not customers:
            return jsonify({"error": "No customers provided"}), 400

        N = len(customers)
        p = min(num_vehicles, N)   # number of clusters = number of vehicles

        log_steps = []
        t_total = time.time()

        # ── Step 1: Estimate seeds ──
        est_k = math.ceil(sum(c['demand'] for c in customers) / (capacity * 0.80))
        n_seeds = max(min(est_k, N, p * 3), p)
        log_steps.append(f"N={N} customers, p={p} vehicles, est_k={est_k}, seeds={n_seeds}")

        # ── Step 2: K-Means Seeding ──
        coords = [[c['x'], c['y']] for c in customers]
        t1 = time.time()
        candidate_indices = kmeans_seeds(coords, n_seeds)
        log_steps.append(f"K-Means: {len(candidate_indices)} candidate exemplars in {time.time()-t1:.2f}s")

        # ── Step 3: V-SGC MILP Re-Clustering ──
        t2 = time.time()
        # Use primary depot for clustering phase
        primary_depot = depots[0]
        clusters, milp_time_used = vsgc_milp_cluster(
            customers, primary_depot, candidate_indices, p,
            capacity, time_limit=milp_time)
        log_steps.append(f"V-SGC MILP: {len(clusters)} clusters in {milp_time_used:.2f}s")

        # ── Step 4: Assign clusters to depots ──
        if len(depots) > 1:
            clusters = assign_clusters_to_depots(clusters, customers, depots)
            log_steps.append(f"Cluster→Depot assignment: {len(depots)} depots")
        else:
            for cl in clusters:
                cl["depot_idx"] = 0

        # ── Step 5: TSPTW routing per cluster ──
        routes = []
        t3 = time.time()
        for cl_idx, cl in enumerate(clusters):
            dep_idx = cl.get("depot_idx", 0)
            dep = depots[dep_idx]
            members = cl["members"]

            # Route size cap for MILP (fall back to 2-opt for large clusters)
            route_tl = min(milp_time, 10)
            ordered = tsptw_milp(members, customers, dep, speed=speed,
                                  time_limit=route_tl)

            dist_val = route_total_distance(ordered, customers, dep)

            routes.append({
                "vehicle": cl_idx + 1,
                "depot_idx": dep_idx,
                "depot": dep,
                "stops": ordered,         # customer indices in visit order
                "demand": cl["demand"],
                "distance": round(dist_val, 3),
                "exemplar": cl["exemplar_idx"],
                "utilization": round(cl["demand"] / capacity * 100, 1)
            })

        routing_time = time.time() - t3
        log_steps.append(f"TSPTW routing: {len(routes)} routes in {routing_time:.2f}s")

        total_dist = sum(r["distance"] for r in routes)
        total_time = time.time() - t_total
        log_steps.append(f"Total solve time: {total_time:.2f}s | Total distance: {total_dist:.2f}")

        # Build response
        response = {
            "status": "optimal" if milp_time_used < milp_time else "feasible",
            "routes": routes,
            "customers": customers,
            "depots": depots,
            "summary": {
                "total_distance": round(total_dist, 3),
                "num_routes": len(routes),
                "num_stops": N,
                "total_demand": sum(c['demand'] for c in customers),
                "avg_utilization": round(sum(r["utilization"] for r in routes) / len(routes), 1),
                "solve_time_s": round(total_time, 2),
                "milp_cluster_time_s": round(milp_time_used, 2),
                "routing_time_s": round(routing_time, 2),
                "algorithm": "V-SGC (K-Means + MILP Grafting + TSPTW)"
            },
            "log": log_steps
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "solver": "scipy.optimize.milp (CBC)",
        "algorithm": "V-SGC: K-Means + MILP Grafting + TSPTW",
        "phases": ["K-Means Seeding", "V-SGC MILP Re-Cluster", "Depot Assignment", "TSPTW Routing"]
    })


@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    import os
    os.makedirs("static", exist_ok=True)
    print("=" * 60)
    print("  V-SGC Route Optimizer Backend")
    print("  Algorithm: K-Means + MILP + TSPTW")
    print("  Server: http://localhost:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False)
