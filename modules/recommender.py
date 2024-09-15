import networkx as nx
import numpy as np
import heapq as hq
class Recommender():
    """
    Class to implement different recommendation algorithms.
    """
    def __init__(self, name:str):
        """
        Function to initialize the Recommender object.
        Args:
            name (str): Name of the recommender. Choose between "pagerank", "wtf", "oba", and "random".
        """
        self.name = name

    def top_k_recommendation(self, G:nx.DiGraph, node_id:int, k:int = 1):
        """
        Function to get the top-k recommendations for a given node.
        Args:
            G (nx.DiGraph): Directed graph object.
            node_id (int): Node ID for which recommendations are to be made.
            k (int): Number of recommendations to be made.
        Returns:
            list: List of top-k recommendations.
        """
        if self.name == "pagerank":
            return self.__personalized_pagerank_recommender(G, node_id, k, diff_to=node_id)
        elif self.name == "wtf":
            return self.__wtf_recommender(G, node_id, k)
        elif self.name == "oba":
            return self.__oba_recommender(G, node_id, k)
        elif self.name == "random":
            return self.__random_recommender(G, node_id, k)
        else:
            raise ValueError("Recommender Not Found. Please choose between pagerank, wtf, oba, and random.")
        
    def __personalized_pagerank_recommender(self, G:nx.DiGraph, node_id:int, k:int, diff_to:int = -1):
        """
        Function to get the top-k recommendations for a given node using Personalized PageRank.
        Args:
            G (nx.DiGraph): Directed graph object.
            node_id (int): Node ID for which recommendations are to be made.
            k (int): Number of recommendations to be made.
            diff_to (int): Node ID to which the recommendations should be different.
        Returns:
            list: List of top-k recommendations.
        """
        #Here we are using the default damping factor of 0.85 for the PageRank algorithm
        self.DAMPING_FACTOR = 0.85
        #Here we get the neighbors of the node_id
        neighbor_list = self.__get_neighbors(G, node_id)
        #Here we create a personalization dictionary where the node_id will have a value of 1 and all other nodes will have zero value
        personalization_dict = {id:0 for id in G.nodes()}
        personalization_dict[node_id] = 1
        #Here we perform personalized pagerank on the graph
        ppr = nx.pagerank(G, alpha=self.DAMPING_FACTOR, personalization=personalization_dict)
        #Here we remove the neighbors of the node_id and the diff_to node from the recommendation dictionary
        recommendation_dict = {key: value for key, value in ppr.items() if key not in neighbor_list and key != diff_to}
        #Here we get the top-k recommendations
        top_k = self.__get_top_k(recommendation_dict, k)
        return top_k
    
    def __wtf_recommender(self, G:nx.DiGraph, node_id:int, k:int):
        """
        Function to get the top-k recommendations for a given node using the WTF algorithm.
        Args:
            G (nx.DiGraph): Directed graph object.
            node_id (int): Node ID for which recommendations are to be made.
            k (int): Number of recommendations to be made.
        Returns:
            list: List of top-k recommendations.
        """
        #Here we select the number of nodes to be included in the circle of trust of the node_id
        k_number_of_trust = int(G.number_of_nodes()*0.1)
        #Here we get the circle of trust of the node_id by using the personalized pagerank recommender
        circle_of_trust = self.__personalized_pagerank_recommender(G, node_id, k_number_of_trust)
        #Here we create the bipartite graph to use in the SALSA algorithm and get the hub and authority nodes
        bipartite_graph, A = self.__create_bipartite_graph(G, circle_of_trust)
        hub_nodes = [int(id_[:-1]) for id_ in bipartite_graph.nodes() if 'H' in id_]
        authority_nodes = [int(id_[:-1]) for id_ in bipartite_graph.nodes() if 'A' in id_]
        #Here we get the M_prime and M_transpose matrices
        M_prime = A[len(hub_nodes):, :][:, :len(hub_nodes)]
        M_transpose = M_prime.T
        #Here we row normalize the matrices
        M_prime = self.__normalize_matrix(M_prime)
        M_transpose = self.__normalize_matrix(M_transpose)
        #Here we get the sorted relevance scores using the SALSA algorithm
        sorted_relevance_scores = self.__SALSA(node_id, circle_of_trust, hub_nodes, authority_nodes, M_prime, M_transpose)
        #Here we get the top-k recommendations
        neighbor_list = self.__get_neighbors(G, node_id)
        top_k = [node for node in sorted_relevance_scores if node not in neighbor_list and node != node_id][:k]
        return top_k
    
    def __oba_recommender(self, G:nx.DiGraph, node_id:int, k:int):
        """
        Function to get the top-k recommendations for a given node using the OBA algorithm.
        Args:
            G (nx.DiGraph): Directed graph object.
            node_id (int): Node ID for which recommendations are to be made.
            k (int): Number of recommendations to be made.
        Returns:
            list: List of top-k recommendations.
        """
        #Here we set the gamma and epsilon values for the OBA algorithm
        GAMMA = 1.6
        TOL_EPSILON = 1e-4
        #Here we define a baseline distance value in order to avoid division by zero
        base_distance = TOL_EPSILON**(-GAMMA)
        potential_recs, recs_scores = [], []
        #For every non neighbor of the node_id, we calculate the opinion difference and add it to the potential recommendations
        for node in nx.non_neighbors(G, node_id):
            opinion_diff = abs(G.nodes[node_id]['opinion'] - G.nodes[node]['opinion'])
            if opinion_diff < TOL_EPSILON:
                opinion_diff = base_distance
            else:
                opinion_diff = opinion_diff**(-GAMMA)

            potential_recs.append(node)
            recs_scores.append(opinion_diff)
        #Here we normalize the scores and get the top-k recommendations
        normalized_scores = np.array(recs_scores)/np.sum(recs_scores)
        recs_dict = {node: score for node, score in zip(potential_recs, normalized_scores)}
        top_k = self.__get_top_k(recs_dict, k)
        return top_k
    
    def __random_recommender(self, G:nx.DiGraph, node_id:int, k:int):
        """
        Function to get the top-k recommendations for a given node using the Random algorithm.
        Args:
            G (nx.DiGraph): Directed graph object.
            node_id (int): Node ID for which recommendations are to be made.
            k (int): Number of recommendations to be made.
        Returns:
            list: List of top-k recommendations.
        """
        #Here we get the non neighbors of the node_id and randomly select k nodes from them
        non_neighbors = list(nx.non_neighbors(G, node_id))
        top_k = np.random.choice(non_neighbors, k, replace=False)
        return top_k
    
    def __SALSA(self, node_id:int, circle_of_trust:list, hub_nodes:list, authority_nodes:list, M_prime:np.ndarray, M_transpose:np.ndarray):
        """
        Function to get the sorted relevance scores using the SALSA algorithm.
        Args:
            node_id (int): Node ID for which recommendations are to be made.
            circle_of_trust (list): List of nodes in the circle of trust of the node_id.
            hub_nodes (list): List of hub nodes in the bipartite graph.
            authority_nodes (list): List of authority nodes in the bipartite graph.
            M_prime (np.ndarray): M_prime matrix.
            M_transpose (np.ndarray): M_transpose matrix.
        Returns:
            np.ndarray: Sorted relevance scores.
        """
        #Here we initialize the s, r, and d vectors and define the tolerance value for the SALSA algorithm
        s,r,d = self.__initialize_salsa(node_id, circle_of_trust, hub_nodes, authority_nodes)
        TOLERANCE = 1e-8
        #Here we iterate until the convergence of the SALSA algorithm
        while True:
            r_prime = np.dot(M_prime, s)
            s_prime = d+(self.DAMPING_FACTOR)*np.dot(M_transpose, r)
            if np.linalg.norm(abs(s_prime - s)) < TOLERANCE:
                break
            s = s_prime
            r = r_prime
        return np.argsort(r)[::-1]
    
    def __initialize_salsa(self, node_id:int, circle_of_trust:list, hub_nodes:list, authority_nodes:list):
        """
        Function to initialize the s, r, and d vectors for the SALSA algorithm.
        Args:
            node_id (int): Node ID for which recommendations are to be made.
            circle_of_trust (list): List of nodes in the circle of trust of the node_id.
            hub_nodes (list): List of hub nodes in the bipartite graph.
            authority_nodes (list): List of authority nodes in the bipartite graph.
        Returns:
            tuple: Tuple containing the s, r, and d vectors.
        """
        #Here we initialize the s, r, and d vectors
        s = np.zeros(len(hub_nodes))
        r = np.zeros(len(authority_nodes))
        d = np.zeros(len(hub_nodes))
        node_index = circle_of_trust.index(node_id)
        s[node_index] = 1
        d[node_index] = 1 - self.DAMPING_FACTOR
        return s,r,d

    def __normalize_matrix(self, matrix:np.ndarray):
        """
        Function to row normalize a matrix.
        Args:
            matrix (np.ndarray): Matrix to be row normalized.
        Returns:
            np.ndarray: Row normalized matrix.
        """
        #Here we row normalize the matrix
        matrix = matrix.copy()
        row_sums = matrix.sum(axis=0)
        row_sums[row_sums == 0] = 1
        new_matrix = matrix / row_sums[np.newaxis:,]
        return new_matrix
    
    def __create_bipartite_graph(self, G:nx.DiGraph, circle_of_trust:list):
        """
        Function to create a bipartite graph from the circle of trust for the SALSA algorithm.
        Args:
            G (nx.DiGraph): Directed graph object.
            circle_of_trust (list): List of nodes in the circle of trust of the node_id.
        Returns:
            nx.DiGraph: Bipartite graph.
            np.ndarray: Adjacency matrix of the bipartite graph.
        """
        #Here we create a bipartite graph
        G_b = nx.DiGraph()
        #First we add the hub nodes to the bipartite graph
        G_b.add_nodes_from([str(node_id)+"H" for node_id in circle_of_trust])
        
        #Then we add the authority nodes to the bipartite graph and connect them to the hub nodes
        for node in circle_of_trust:
            neighbors = self.__get_neighbors(G, node)
            for neighbor in neighbors:
                G_b.add_edge(str(neighbor)+"A", str(node)+"H")
        #Here we get the adjacency matrix of the bipartite graph
        adjacency_matrix = nx.to_numpy_array(G_b)
        return G_b, adjacency_matrix
    
    
    def __get_neighbors(self, G:nx.DiGraph, node_id:int):
        """
        Function to get the neighbors of a node in a directed graph.
        Args:
            G (nx.DiGraph): Directed graph object.
            node_id (int): Node ID for which neighbors are to be found.
        Returns:
            list: List of neighbors of the node.
        """
        #Here we get the neighbors of the node_id
        return list(G.neighbors(node_id))

    def __get_top_k(self, recommendation_dict:dict, k:int):
        """
        Function to get the top-k recommendations from a dictionary.
        Args:
            recommendation_dict (dict): Dictionary containing the recommendations.
            k (int): Number of recommendations to be made.
        Returns:
            list: List of top-k recommendations.
        """
        #Here we get the top-k recommendations
        heap_list= []
        for key, value in recommendation_dict.items():
            hq.heappush(heap_list, ((-1)*value, key))
        top_k = [key for _, key in hq.nsmallest(k, heap_list)]
        return top_k
