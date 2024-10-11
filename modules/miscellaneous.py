from modules.random_graph import LFRBenchmark
from modules.measurement import Measurement
import matplotlib.pyplot as plt
from typing import Tuple, List
from modules.PROD import PROD
from scipy.stats import kstest
import networkx as nx
import numpy as np
import matplotlib
import json
import tqdm


def plot_graph(G:nx.DiGraph, n_maxmin_alpha:Tuple[float], e_maxmin_alpha:Tuple[float], ax: plt.Axes = None) -> None:
    """
    Function to plot the LFR Benchmark Graph
    Parameters:
        G (nx.DiGraph): Graph to plot
        n_maxmin_alpha (Tuple[float]): Tuple with the minimum and maximum alpha values for the nodes
        e_maxmin_alpha (Tuple[float]): Tuple with the minimum and maximum alpha values for the edges
        ax (plt.Axes): Matplotlib Axes to plot the graph
    Returns:
        None
    """
    #Get the positions of the nodes
    nodes_positions = __get_community_positions(G)
    #Get the colors and sizes of the nodes along with other attributes
    nodes_color = [G.nodes[n]["opinion"] for n in G.nodes]
    nodes_size = [nx.clustering(G, n)*500 for n in G.nodes]
    nodes_alpha = __get_nodes_alpha(G, n_maxmin_alpha)
    edges_alpha = __get_edges_alpha(G, e_maxmin_alpha)
    black = lambda a: matplotlib.colors.to_hex((0, 0, 0, a), keep_alpha=True)
    edge_colors = [black(edges_alpha[i]) for i in range(G.number_of_edges())]
    #Plot the graph
    nx.draw_networkx_nodes(G, nodes_positions, node_color=nodes_color, node_size=nodes_size,
                           alpha=nodes_alpha, cmap=plt.cm.coolwarm, ax=ax)
    nx.draw_networkx_edges(G, nodes_positions, width=0.15, edge_color=edge_colors, ax=ax, arrows=False)

    if ax is not None:
        ax.axis("on")
    return None

def get_results_matrix(measure:str) -> Tuple[np.array]:
    """
    This function calculates the delta value matrix for the given measure for all recommender systems and parameter values.
    The delta value is calculated as the mean of the difference between the metrics obtained with the recommender and the metrics obtained without the recommender.

    Parameters:
        measure (str): The name of the measure to calculate the delta value.

    Returns:
        Tuple[np.array]: The delta value matrix for the given measure for all recommender systems and parameter values.
    """
    #Here we check if the measure is one of the following: clustering, gini, nci, rwc
    assert measure in ['clustering', 'gini', 'nci', 'rwc'], 'The measure must be one of the following: clustering, gini, nci, rwc'
    #Here we define the possible parameter values for modularity and homophily
    parameter_values = [(0.05, 0.2),(0.05, 0.5),(0.05, 0.8),(0.55, 0.2),(0.55, 0.5),(0.55, 0.8),(0.95, 0.2),(0.95, 0.5),(0.95, 0.8)]
    #Here we define the results matrix for each recommender
    results_random, results_pagerank, results_wtf, results_oba  = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    #Here we calculate the delta for each pair of parameters and add it to the matrix
    for i, (modularity, homophily) in enumerate(parameter_values):
        results_random[i//3, i%3] = __calculate_delta(measure, 'random', modularity, homophily)
        results_pagerank[i//3, i%3] = __calculate_delta(measure, 'pagerank', modularity, homophily)
        results_wtf[i//3, i%3] = __calculate_delta(measure, 'wtf', modularity, homophily)
        results_oba[i//3, i%3] = __calculate_delta(measure, 'oba', modularity, homophily)

    #Here we return the results matrix for each recommender
    return results_random, results_pagerank, results_wtf, results_oba

def run_example_simulations(homophily:float, modularity:float, n:int=400) -> Tuple[list, list, list, list, list]:
    """
    This function runs the example simulations for the given homophily and modularity values.
    The example simulations consist of generating a LFR benchmark graph and running the PROD algorithm with different recommender systems.

    Parameters:
        homophily (float): The homophily value for the LFR benchmark graph.
        modularity (float): The modularity value for the LFR benchmark graph.
        n (int): The number of nodes for the LFR benchmark graph.

    Returns:
        Tuple[list]: A tuple with the generated graphs, clustering coefficients, gini coefficients, RWC scores and NCI scores for each recommender system.
    """
    #We can generate a graph with the LFR benchmark
    lfr = LFRBenchmark(n, homophily=homophily, modularity=modularity, init_opinions=True)
    #We initialize the measurement module
    measurement = Measurement()
    #We can generate a list to save the graphs and some measures for each recommender
    clustering = [measurement.clustering_coefficient(lfr.G)]
    gini = [measurement.gini_in_degree(lfr.G)]
    rwc = [measurement.RWC(lfr.G)]
    nci = [measurement.NCI(lfr.G)]
    graphs = [lfr.G]

    #We can generate a list of recommenders
    recommenders = [None, "wtf", "oba", "pagerank", "node2vec"]
    #We can generate the simulations for each recommender
    for recommender_system in tqdm.tqdm(recommenders):
        n_recommendations = 0
        #We can set the number of recommendations as 40% of the number of edges of the graph
        if recommender_system is not None:
            n_recommendations = int(graphs[0].number_of_edges()*0.4)
        #We can make the simulation
        prod = PROD(graphs[0], recommender_n=recommender_system, int_per_step=2, n_recommendations=n_recommendations, n_steps=5000)
        prod.run()
        #We can save the measures
        graphs.append(prod.G)
        clustering.append(measurement.clustering_coefficient(prod.G))
        gini.append(measurement.gini_in_degree(prod.G))
        rwc.append(measurement.RWC(prod.G))
        nci.append(measurement.NCI(prod.G))
    return graphs, clustering, gini, rwc, nci

def get_maxmin_alphas(graphs:List[nx.DiGraph]) -> Tuple[float]:
    """
    Function to get the maximum and minimum alpha values for the nodes and edges
    Parameters:
        graphs (List[nx.DiGraph]): List of graphs to get the maximum and minimum alpha values
    Returns:
        Tuple[float]: Tuple with the minimum and maximum alpha values for the nodes and edges
    """
    #Now we obtain the max and min alphas for nodes and edges
    final_node_alphas, final_edge_alphas = np.array([]), np.array([])
    #We loop through the graphs in order to get all the possible alphas
    for graph in graphs:
        node_alphas = get_standardized_node_alpha(graph)
        edge_alphas = get_standardized_edge_alpha(graph)
        final_node_alphas = np.append(final_node_alphas, node_alphas)
        final_edge_alphas = np.append(final_edge_alphas, edge_alphas)

    #Now we obtain the minimum max and max min alphas
    n_maxmin_alpha = np.percentile(final_node_alphas, 5), np.percentile(final_node_alphas, 95)
    e_maxmin_alpha = np.percentile(final_edge_alphas, 5), np.percentile(final_edge_alphas, 95)
    return n_maxmin_alpha, e_maxmin_alpha

def get_intervention_results(measure:str) -> Tuple[np.array]:
    """
    Function to get the intervention results for the given measure
    Parameters:
        measure (str): The measure to get the intervention results
    Returns:
        Tuple[np.array]: Tuple with the intervention results for the random and opinion diversity interventions
    """
    #First we open the json file with the results
    with open(f'results/results_pagerank_0.95_0.8_intervention.json') as json_file:
            intervention_results = json.load(json_file)

    intervention_probabilities = [0, 0.15, 0.35, 0.55, 0.75, 0.95]
    #Here we create an empty numpy array to store the results
    results_pagerank_intervention_random = np.zeros(len(intervention_probabilities))
    results_pagerank_intervention_opinion = np.zeros(len(intervention_probabilities))

    #Here we calculate the delta for each intervention probability and add it to the matrix
    #The initial condition is probability = 0
    metrics_without_recommender = np.array(intervention_results.get('metrics_without_recommender').get(measure))
    metrics_with_recommender = np.array(intervention_results.get('metrics_with_recommender').get(measure))
    initial_delta = np.mean(metrics_with_recommender - metrics_without_recommender)

    #Here we add the initial condition to the results
    results_pagerank_intervention_random[0] = initial_delta
    results_pagerank_intervention_opinion[0] = initial_delta
    #Now we calculate the delta for the rest of the intervention probabilities
    for i, intervention_prob in enumerate(intervention_probabilities[1:]):
        random_intervention_metrics = np.array(intervention_results.get(f'metrics_random_prob_{intervention_prob}').get(measure))
        opinion_intervention_metrics = np.array(intervention_results.get(f'metrics_opinion diversity_prob_{intervention_prob}').get(measure))
        results_pagerank_intervention_random[i+1] = np.mean(random_intervention_metrics - metrics_without_recommender)
        results_pagerank_intervention_opinion[i+1] = np.mean(opinion_intervention_metrics - metrics_without_recommender)

    return results_pagerank_intervention_random, results_pagerank_intervention_opinion

def __get_nodes_alpha(G:nx.DiGraph, n_maxmin_alpha:Tuple[float]) -> np.array:
    """
    Function to get the alpha values for the nodes
    Parameters:
        G (nx.DiGraph): Graph to get the alpha values for the nodes
        n_maxmin_alpha (Tuple[float]): Tuple with the minimum and maximum alpha values for the nodes
    Returns:
        np.array: Array with the alpha values for the nodes
    """
    #Here we get the standardized alpha values for the nodes
    alpha = get_standardized_node_alpha(G)
    #Here we normalize the alpha values
    minimum_alpha, maximum_alpha = n_maxmin_alpha
    alpha = (alpha - minimum_alpha)/(maximum_alpha - minimum_alpha)
    #Here we clip the alpha values to be between 0 and 1
    alpha = np.clip(alpha, 0., 1.)
    #Here we return the alpha values
    return alpha

def __get_edges_alpha(G:nx.DiGraph, e_maxmin_alpha:Tuple[float]) -> np.array:
    """
    Function to get the alpha values for the edges
    Parameters:
        G (nx.DiGraph): Graph to get the alpha values for the edges
        e_maxmin_alpha (Tuple[float]): Tuple with the minimum and maximum alpha values for the edges
    Returns:
        np.array: Array with the alpha values for the edges
    """
    #Here we get the standardized alpha values for the edges
    alpha = get_standardized_edge_alpha(G)
    #Here we normalize the alpha values
    minimum_alpha, maximum_alpha = e_maxmin_alpha
    alpha = (alpha - minimum_alpha)/(maximum_alpha - minimum_alpha)
    #Here we clip the alpha values to be between 0 and 1
    alpha = np.clip(alpha, 0., 1.)
    #Here we return the alpha values
    return alpha

def get_standardized_node_alpha(G:nx.DiGraph) -> np.array:
    """
    Function to get the standardized alpha values for the nodes
    Parameters:
        G (nx.DiGraph): Graph to get the standardized alpha values
    Returns:
        np.array: Array with the standardized alpha values for the nodes
    """
    #Here we obtain the opinions of the graph and the average opinion of the neighbors of each node
    opinions = np.array([G.nodes[u]['opinion'] for u in G.nodes()])
    avg_neighbor_opinion_vector = [np.mean([G.nodes[v]['opinion'] for v in G.neighbors(u)]) for u in G.nodes()]
    #Here we standardize the alpha values by subtracting the mean and dividing by the standard deviation 
    alpha = np.array([(opinions[u]-np.mean(opinions))*(avg_neighbor_opinion_vector[u]-np.mean(avg_neighbor_opinion_vector)) for u in G.nodes()])
    alpha = alpha/(np.std(opinions)*np.std(avg_neighbor_opinion_vector))
    #Here we return the standardized alpha values
    return alpha

def get_standardized_edge_alpha(G:nx.DiGraph) -> np.array:
    """
    Function to get the standardized alpha values for the edges
    Parameters:
        G (nx.DiGraph): Graph to get the standardized alpha values
    Returns:
        np.array: Array with the standardized alpha values for the edges
    """
    #Here we obtain the opinions of the graph
    opinions = np.array([G.nodes[u]['opinion'] for u in G.nodes()])
    #Here we standardize the alpha values by subtracting the mean and dividing by the standard deviation
    alpha = [(opinions[u]-np.mean(opinions))*(opinions[v]-np.mean(opinions))/np.std(opinions)**2 for u, v in G.edges()]
    return alpha


def __calculate_delta(measure:str, recommender:str, modularity:float, homophily:float, intervention_name:str=None, intervention_prob:float=0.0) -> float:
    """
    This function calculates the delta value for a given measure, recommender, modularity, homophily, intervention_name and intervention_prob.
    The delta value is calculated as the mean of the difference between the metrics obtained with the recommender and the metrics obtained without the recommender.

    Parameters:
        measure (str): The name of the measure to calculate the delta value.
        recommender (str): The name of the recommender to calculate the delta value.
        modularity (float): The modularity value to calculate the delta value.
        homophily (float): The homophily value to calculate the delta value.
        intervention_name (str): The name of the intervention to calculate the delta value.
        intervention_prob (float): The probability of the intervention to calculate the delta value.

    Returns:
        float: The delta value for the given measure, recommender, modularity, homophily, intervention_name and intervention_prob.
    """
    #Here we define the significance level for the Kolmogorov-Smirnov test
    SIGNIFICANCE_LEVEL = 0.001
    #Here we load the data from the json file
    with open(f'results/{recommender}_{intervention_name}_{intervention_prob}_{modularity}_{homophily}.json') as json_file:
        data = json.load(json_file)
    
    #Here we obtain the metrics for the given measure: initial_metrics, metrics_without_recommender and metrics_with_recommender
    initial_metrics = np.array(data.get('init_metrics').get(measure))
    metrics_without_recommender = np.array(data.get('metrics_without_recommender').get(measure))
    metrics_with_recommender = np.array(data.get('metrics_with_recommender').get(measure))

    #Here we perform a Kolmogorov-Smirnov test to check if the two different distributions are the same
    p_value_kstest = __kolmogorov_smirnoff_test(metrics_with_recommender, metrics_without_recommender, initial_metrics)
    #If the p-value is greater than 0.001, we return a nan value since the two distributions are the same and the delta value is not meaningful
    if p_value_kstest > SIGNIFICANCE_LEVEL:
        return np.nan
    
    #Here we calculate the delta value
    return np.mean(metrics_with_recommender - metrics_without_recommender)

def __kolmogorov_smirnoff_test(metrics_with_recommender:np.array, metrics_without_recommender:np.array, initial_metrics:np.array) -> float:
    """
    This function performs a Kolmogorov-Smirnov test to check if the two given distributions are the same.

    Parameters:
        metrics_with_recommender (np.array): The metrics obtained with the recommender.
        metrics_without_recommender (np.array): The metrics obtained without the recommender.
        initial_metrics (np.array): The initial metrics.

    Returns:
        
    """
    #Here we obtain the two distributions we want to compare
    distr1 = metrics_with_recommender - metrics_without_recommender
    distr2 = metrics_without_recommender - initial_metrics

    #Here we perform the Kolmogorov-Smirnov test
    kolmogorov_smirnov = kstest(distr1, distr2)
    #Here we return the p-value of the Kolmogorov-Smirnov test
    return kolmogorov_smirnov.pvalue

def __get_labeled_communities(G:nx.DiGraph) -> dict:
    """
    Function to get the labeled communities of the graph
    Parameters:
        G (nx.DiGraph): Graph to get the labeled communities
    Returns:
        dict: Dictionary with the labeled communities
    """
    #Here we get the labeled communities
    unique_communities = [tuple(sorted(G.nodes[i]['community'])) for i in G.nodes]
    unique_communities = list(set(unique_communities))
    labeled_communities = {i:community for i, community in enumerate(unique_communities)}
    return labeled_communities

def __get_community_positions(G:nx.DiGraph, seed:int=0) -> dict:
        """
        Function to get the positions of the nodes in the graph
        Parameters:
            G (nx.DiGraph): Graph to get the positions of the nodes
            seed (int): Seed for the random number generator
        Returns:
            dict: Dictionary with the positions of the nodes
        """
        np.random.seed(seed)
        COMMUNITY_DISTANCE_SCALE = 100
        COMMUNITY_DISTANCE_DIVISOR = 5
        NODE_POSITION_OFFSET_SCALE = 0.2
        NODE_POSITION_RANDOM= 0.025
        ADDITIONAL_OFFSET_CHOICES = [-0.1, 0.1]

        #Get the number of communities and the number of nodes per community
        labeled_communities = __get_labeled_communities(G)
        number_of_communities = len(labeled_communities)
        nodes_per_community= {k: len(v) for k, v in labeled_communities.items()}
        #Generate the distance between the communities
        G_distance = nx.Graph()
        for u in range(number_of_communities):
            for v in range(number_of_communities):
                weight = COMMUNITY_DISTANCE_SCALE / (COMMUNITY_DISTANCE_DIVISOR * (nodes_per_community[u] + nodes_per_community[v]))
                G_distance.add_edge(u, v, weight=weight)

        community_positions = nx.spring_layout(G_distance, weight='weight', seed=seed)
        node_positions ={}
        #Generate the positions of the nodes
        for centroid in G_distance.nodes():
            centr_pos = community_positions[centroid]
            nodes_in_community = labeled_communities[centroid]
            sorted_nodes_in_community = sorted(nodes_in_community, key=lambda x: G.degree[x], reverse=True)
            #Map the nodes to the positions with an offset depending on the degree
            for i, node in enumerate(sorted_nodes_in_community):
                offset = NODE_POSITION_OFFSET_SCALE * i * np.random.uniform(low=-NODE_POSITION_RANDOM, high=NODE_POSITION_RANDOM, size=2)
                additional_offset = np.array([np.random.choice(ADDITIONAL_OFFSET_CHOICES) for _ in range(2)])
                node_positions[node] = centr_pos + offset + additional_offset
        
        return node_positions
