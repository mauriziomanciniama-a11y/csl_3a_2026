!pip install -U ortools
!pip install scikit-learn
# RI -> [SOLUZIONE #153] Miglioramento trovato! Costo: 1548044
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import cdist

# ================= CLASSE MONITOR PER VISUALIZZARE I PROGRESSI =================
class SearchMonitor(pywrapcp.SearchMonitor):
    def __init__(self, routing):
        super().__init__(routing.solver())
        self.routing = routing
        self.count = 0

    def AtSolution(self):
        self.count += 1
        cost = self.routing.CostVar().Min()
        # Stampiamo il progresso nel terminale
        print(f" -> [SOLUZIONE #{self.count}] Miglioramento trovato! Costo: {cost}")
        return True

def main():
    # ================= CONFIGURAZIONE =================
    PATH_FILE_EXCEL = "lista cassonetti CSL 3 assi XI RI 2026.xlsx"

    if not os.path.exists(PATH_FILE_EXCEL):
        print(f"Errore: Il file '{PATH_FILE_EXCEL}' non e' stato trovato.")
        return

    MAX_CASS = 105
    TEMPO_CALCOLO = 200
    SCALA_DISTANZA = 100_000
    PESO_CENTROIDE = 500_000 # Forza la compattezza radiale intorno ai centri

    # ---- I TUOI 3 CENTROIDI MANUALI ----
    MIEI_CENTROIDI = np.array([
        [41.8717133, 12.4709896],
        [41.8638707, 12.4733328],
        [41.862639, 12.4655162],
        [41.8577232, 12.450469],
        [41.8480366, 12.4627007],
        [41.8597586, 12.4193671],
        [41.8214988, 12.3947928],
        [41.8445433, 12.4381301]
    ])
    # ==================================================

    print("Avvio elaborazione itinerari 2026 con Centroidi Personalizzati...")

    # 1. Caricamento Dati
    df_punti = pd.read_excel(PATH_FILE_EXCEL).dropna(subset=["Latitudine", "Longitudine"])
    num_punti = len(df_punti)

    # Calcolo veicoli totali necessari
    num_veicoli_necessari = int(np.ceil(num_punti / MAX_CASS))
    num_mancanti = max(0, num_veicoli_necessari - len(MIEI_CENTROIDI))

    # 2. Generazione Centroidi (Ibrida)
    if num_mancanti > 0:
        print(f"Generazione di {num_mancanti} centroidi extra per coprire tutti i punti...")
        kmeans = KMeans(n_clusters=num_mancanti, random_state=42, n_init=10)
        kmeans.fit(df_punti[['Latitudine', 'Longitudine']])
        centroidi = np.vstack([MIEI_CENTROIDI, kmeans.cluster_centers_])
    else:
        centroidi = MIEI_CENTROIDI

    num_veicoli = len(centroidi)
    print(f"Cassonetti: {num_punti} | Itinerari totali: {num_veicoli}")

    # 3. Preparazione Modello
    centroide_medio = df_punti[["Latitudine", "Longitudine"]].mean().values
    df_deposito = pd.DataFrame([{
        "Latitudine": centroide_medio[0],
        "Longitudine": centroide_medio[1],
        "Domanda": 0,
        "VIA": "DEPOSITO"
    }])
    df_punti["Domanda"] = 1
    df_model = pd.concat([df_deposito, df_punti], ignore_index=True)

    coords = df_model[["Latitudine", "Longitudine"]].values
    dist_matrix = (cdist(coords, coords, metric="euclidean") * SCALA_DISTANZA).astype(int)

    # 4. Solver OR-Tools
    manager = pywrapcp.RoutingIndexManager(len(df_model), num_veicoli, 0)
    routing = pywrapcp.RoutingModel(manager)

    def get_weighted_callback(v_idx):
        c_point = centroidi[v_idx]
        def callback(from_idx, to_idx):
            f_node = manager.IndexToNode(from_idx)
            t_node = manager.IndexToNode(to_idx)
            if f_node == 0 or t_node == 0:
                return dist_matrix[f_node][t_node]

            base_dist = dist_matrix[f_node][t_node]
            # Attrazione verso il centroide assegnato
            dist_dal_centro = np.linalg.norm(coords[t_node] - c_point) * PESO_CENTROIDE
            return int(base_dist + dist_dal_centro)
        return callback

    for v in range(num_veicoli):
        cb = routing.RegisterTransitCallback(get_weighted_callback(v))
        routing.SetArcCostEvaluatorOfVehicle(cb, v)

    # Vincolo Capacità
    def demand_callback(from_idx):
        return int(df_model.iloc[manager.IndexToNode(from_idx)]["Domanda"])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, [MAX_CASS] * num_veicoli, True, "Capacity"
    )

    # 5. Parametri di Ricerca
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = TEMPO_CALCOLO

    # --- INTEGRAZIONE MONITOR ---
    monitor = SearchMonitor(routing)
    routing.AddSearchMonitor(monitor)
    # ----------------------------

    print(f"Inizio calcolo (Limite: {TEMPO_CALCOLO}s)...")
    solution = routing.SolveWithParameters(search_params)

    # 6. Output
    if solution:
        output_dir = "itinerari_3A_2026_RI"
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        it_count = 0
        for v in range(num_veicoli):
            idx = routing.Start(v)
            nodes = []
            while not routing.IsEnd(idx):
                node_idx = manager.IndexToNode(idx)
                if node_idx != 0: nodes.append(df_model.iloc[node_idx])
                idx = solution.Value(routing.NextVar(idx))

            if nodes:
                it_count += 1
                df_res = pd.DataFrame(nodes).drop(columns=["Domanda"])
                df_res.to_excel(f"{output_dir}/itinerario_{it_count}.xlsx", index=False)
                print(f"OK - Itinerario {it_count}: {len(nodes)} cassonetti")

        shutil.make_archive(output_dir, 'zip', output_dir)
        print(f"\nCompletato! Archivio creato: {output_dir}.zip")
    else:
        print("Nessuna soluzione trovata. Prova ad aumentare il numero di veicoli (centroidi) o il tempo.")

if __name__ == "__main__":
    main()
