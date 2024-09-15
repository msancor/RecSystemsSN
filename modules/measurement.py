from scipy import stats
import networkx as nx
import numpy as np

class Measurement():
    """
    Class to compute different measurements in a graph
    """
    def __init__(self):
        """
        Function to initialize the Measurement class
        """
        pass

    def clustering_coefficient(self, G:nx.DiGraph):
        """
        Function to compute the clustering coefficient of a graph
        Args:
            G (nx.DiGraph): Graph to compute the clustering coefficient
        """
        return nx.average_clustering(G)
    
    def gini_in_degree(self, G:nx.DiGraph):
        """
        Function to compute the Gini coefficient of the in-degree distribution
        Args:
            G (nx.DiGraph): Graph to compute the Gini coefficient
        """
        in_degree_distribution = np.array(list(dict(G.in_degree()).values()))
        return self.__gini(in_degree_distribution)

    def NCI(self, G:nx.DiGraph):
        """
        Function to compute the Neighbors Correlation Index
        Args:
            G (nx.DiGraph): Graph to compute the Neighbors Correlation Index
        """
        #Here we obtain the opinion vector of the nodes
        opinions = list(nx.get_node_attributes(G, 'opinion').values())
        #Now we obtain the average opinion of neighbors vector
        avg_neighbor_opinions = [np.mean([G.nodes[neighbor]['opinion'] for neighbor in G.neighbors(node)]) for node in G.nodes()]
        #Here we obtain the pearson correlation between the opinions and the average neighbor opinions
        nci_score = stats.pearsonr(opinions, avg_neighbor_opinions)[0]
        return nci_score


    def RWC(self, G:nx.DiGraph):
        """
        Function to compute the Random Walk Controversy Score
        Args:
            G (nx.DiGraph): Graph to compute the Random Walk Centrality
        """
        #Here we prune the graph to enforce restarting of the random walk when reaching a hub node
        G_pruned = self.__prune_graph(G)
        #Here we obtain the two partitions of the graph
        #The nodes with opinions below 0.5 and the nodes with opinions above 0.5
        X = [node for node in G_pruned.nodes() if G_pruned.nodes[node]['opinion'] < 0.5]
        Y = [node for node in G_pruned.nodes() if G_pruned.nodes[node]['opinion'] >= 0.5]
        #Here we initialize the random walk pagerank scores
        init_X ={node:(1 if node in X else 0) for node in G_pruned.nodes()}
        init_Y ={node:(1 if node in Y else 0) for node in G_pruned.nodes()}
        #Here we perform the two random walks
        pr_X = nx.pagerank(G_pruned, alpha=0.85, nstart=init_X, personalization=init_X)
        pr_Y = nx.pagerank(G_pruned, alpha=0.85, nstart=init_Y, personalization=init_Y)
        #Here we obtain the probabilities of starting in partition i and reaching partition j
        P_XX = sum([pr_X[node] for node in X])
        P_YY = sum([pr_Y[node] for node in Y])
        P_XY = sum([pr_X[node] for node in Y])
        P_YX = sum([pr_Y[node] for node in X])
        #Here we compute the Random Walk Controversy score
        random_controversy = P_XX*P_YY - P_XY*P_YX
        return random_controversy

    def __prune_graph(self, G:nx.DiGraph, percentile:int=95):
        """
        Function to prune hub nodes out edges over the 95th percentile of the degree distribution
        Args:
            G (nx.DiGraph): Graph to prune
            percentile (int): Percentile to prune the graph
        Returns:
            nx.DiGraph: Pruned graph
        """
        #Initialize the pruned graph
        G_pruned = G.copy()
        #Get the degree distribution
        degree_distribution = dict(G_pruned.degree())
        #Obtain the maximum degree above the percentile
        max_degree = np.percentile(list(degree_distribution.values()), percentile)
        #Find the hub nodes
        hub_nodes = [node for node, degree in degree_distribution.items() if degree > max_degree]
        #Remove the out edges of the hub nodes
        for node in hub_nodes:
            G_pruned.remove_edges_from(list(G_pruned.out_edges(node)))
        return G_pruned
    
    def __gini(self, array:np.array):
        """
        Function that calculates the Gini coefficient of a numpy array.
        This function is taken textually from the Github repository:
        https://github.com/oliviaguest/gini
        
        All credits to the author. This is just an auxiliary function for the project.

        Args:
            array (np.array): Array to calculate the Gini coefficient
        Returns:
            float: Gini coefficient
        """
        # based on bottom eq:
        # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
        # from:
        # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        # All values are treated equally, arrays must be 1d:
        array = array.flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
        # Values cannot be 0:
        array = array + 0.0000001
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1,array.shape[0]+1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
