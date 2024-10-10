"""
Here we run the simulations of the PROD model. We create the graph, the recommender, and the model. We run the model and save the results.
Here we add the evaluation of the intervention. We run the simulations for different intervention probabilities.
"""
#We import the necessary libraries
from modules.random_graph import LFRBenchmark  
from modules.measurement import Measurement
from modules.PROD import PROD
from tqdm import tqdm
import json
import sys


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

def save_results(metrics:dict, recommender_name:str, modularity:float, homophily:float):
    """
    Function to save the results in a json file. If the file exists, it will overwrite it.
    Args:
        metrics (dict): The metrics of the model
        recommender_name (str): The name of the recommender
        modularity (float): The modularity of the graph
        homophily (float): The homophily of the graph
    """
    with open(f"results_{recommender_name}_{modularity}_{homophily}_intervention.json", "w") as f:
        json.dump(metrics, f)

#Here we run the simulations
if __name__ == "__main__":
    recommender_name = sys.argv[1]
    modularity = float(sys.argv[2])
    homophily = float(sys.argv[3])
    #Here we define a range of intervention probabilities
    parameter_values = [("random", 0.15), ("random", 0.35), ("random", 0.55), ("random", 0.75), ("random", 0.95),
                        ("opinion diversity", 0.15), ("opinion diversity", 0.35), ("opinion diversity", 0.55), ("opinion diversity", 0.75), ("opinion diversity", 0.95)]
    #Here we define the number of graphs to generate
    K = 500
    m = Measurement()
    #Here we create a list with K random graphs
    print(f"Generating random graphs with homophily {homophily} and modularity {modularity}")
    Gs = [LFRBenchmark(n=400, homophily=homophily, modularity=modularity).G for _ in tqdm(range(K))]
    #Here we obtain the initial metrics
    print("Obtaining initial metrics...")
    init_clustering = [m.clustering_coefficient(G) for G in Gs]
    init_gini = [m.gini_in_degree(G) for G in Gs]
    init_nci = [m.NCI(G) for G in Gs]
    init_rwc = [m.RWC(G) for G in Gs]
    init_metrics = {"clustering":init_clustering, "gini":init_gini, "nci":init_nci, "rwc":init_rwc}
    print("Initial metrics obtained... Running PROD model without recommender")
    metrics_without_recommender = run_PROD_k_times(Gs, recommender_name, None, 0, 0, K)
    print("Running PROD model with recommender without intervention")
    metrics_with_recommender = run_PROD_k_times(Gs, recommender_name, None, 0, 1, K)
    print("Saving initial results...")
    #Here we save some results
    results = {"modularity":modularity, "homophily":homophily, "init_metrics":init_metrics,
                    "metrics_without_recommender":metrics_without_recommender, "metrics_with_recommender":metrics_with_recommender}
    save_results(results, recommender_name, modularity, homophily)
    for intervention_name, intervention_prob in parameter_values:
        print(f"Running simulations for {intervention_name} intervention with probability {intervention_prob}")
        metrics = run_PROD_k_times(Gs, recommender_name, intervention_name, intervention_prob, 1, K)
        print("Saving results...")
        #Here we save the results in a json file
        results[f"metrics_{intervention_name}_prob_{intervention_prob}"] = metrics
        save_results(results, recommender_name, modularity, homophily)
        

