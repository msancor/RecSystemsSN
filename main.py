"""
Here we run the simulations of the PROD model. We create the graph, the recommender, and the model. We run the model and save the results.
"""
#We import the necessary libraries
import networkx as nx
import numpy as np
from modules.PROD import PROD
from modules.random_graph import LFRBenchmark  
from modules.measurement import Measurement
from tqdm import tqdm
import sys
import json

def read_intervention(intervention_name:str)->str:
    """
    Function to read the intervention name
    Args:
        intervention_name (str): The name of the intervention
    Returns:
        str: The intervention name
    """
    if (intervention_name != "random") and (intervention_name != "opinion_diversity"):
        return None
    return intervention_name

def run_PROD_k_times(G_list:list, recommender_name:str, intervention_name:str, intervention_prob:float, recommendations:bool, K:int)->dict:
    """
    Function to run the PROD model K times
    Args:
        G_list (list): The list of graphs
        recommender_name (str): The name of the recommender
        intervention_name (str): The name of the intervention
        intervention_prob (float): The probability of the intervention
        recommendations (bool): Giving or not recommendations
        K (int): The number of times to run the model
    Returns:
        dict: The metrics of the model
    """
    m = Measurement()
    clus, gini, nci, rwc = [], [], [], []
    for step in tqdm(range(K)):
        #Here we check if we are giving recommendations
        if recommendations:
            recommendations = int(G_list[step].number_of_edges()*0.4)
        #Here we create the PROD model
        prod = PROD(G_list[step], recommender_n=recommender_name, int_per_step=2,
                    n_recommendations=recommendations, n_steps=5000,
                    intervention= intervention_name, intervention_prob= intervention_prob)
        #Here we run the model
        prod.run()
        #Here we obtain the metrics
        clus.append(m.clustering_coefficient(prod.G))
        gini.append(m.gini_in_degree(prod.G))
        nci.append(m.NCI(prod.G))
        rwc.append(m.RWC(prod.G))
    #Here we save the metrics in a dictionary
    metrics = {"clustering":clus, "gini":gini, "nci":nci, "rwc":rwc}
    return metrics

#Here we run the simulations
if __name__ == "__main__":
    recommender_name = sys.argv[1]
    intervention_name = read_intervention(sys.argv[2])
    intervention_prob = float(sys.argv[3])
    #Here we define a grid of modularity and homophily values to run the simulations
    parameter_values = [(0.95, 0.2),(0.95, 0.5),(0.95, 0.8),(0.55, 0.2),(0.55, 0.5),(0.55, 0.8),(0.05, 0.2),(0.05, 0.5),(0.05, 0.8)]
    #Here we define the number of graphs to generate
    K = 500
    m = Measurement()
    #Here we create a sequence of K random graphs with the LFR Benchmark for each modularity and homophily value
    for modularity, homophily in parameter_values:
        print(f"Running simulations for modularity {modularity} and homophily {homophily}")
        #Here we create a list with K random graphs
        print("Generating random graphs...")
        Gs = [LFRBenchmark(n=400, homophily=homophily, modularity=modularity).G for _ in tqdm(range(K))]
        #Here we obtain the initial metrics
        print("Obtaining initial metrics...")
        init_clustering = [m.clustering_coefficient(G) for G in Gs]
        init_gini = [m.gini_in_degree(G) for G in Gs]
        init_nci = [m.NCI(G) for G in Gs]
        init_rwc = [m.RWC(G) for G in Gs]
        init_metrics = {"clustering":init_clustering, "gini":init_gini, "nci":init_nci, "rwc":init_rwc}
        print("Initial metrics obtained... Running PROD model without recommender")
        metrics_without_recommender = run_PROD_k_times(Gs, recommender_name, intervention_name, intervention_prob, 0, K)
        print("Running PROD model with recommender")
        metrics_with_recommender = run_PROD_k_times(Gs, recommender_name, intervention_name, intervention_prob, 1, K)
        print("Saving results...")
        #Here we save the results in a json file
        results = {"modularity":modularity, "homophily":homophily, "init_metrics":init_metrics,
                    "metrics_without_recommender":metrics_without_recommender, "metrics_with_recommender":metrics_with_recommender}
        with open(f"results/{recommender_name}_{intervention_name}_{intervention_prob}_{modularity}_{homophily}.json", "w") as f:
            json.dump(results, f)

