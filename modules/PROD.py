from modules.recommender import Recommender
import networkx as nx
import numpy as np

class PROD():
    def __init__(self, G:nx.DiGraph, recommender_n:str, int_per_step:int, n_recommendations:int, n_steps:int, intervention:str = None, intervention_prob:float = 0):
        """
        This class implements the PROD model. 
        The PROD model is a model that simulates the recommendation of edges in a social network.
        The model is based on the bounded confidence model and the recommendation of edges based on a recommender system.
        The model has an intervention policy that can be random or opinion diversity.
        The intervention policy is used to add edges to nodes that are different from the recommended node.
        
        Args:
            G (nx.DiGraph): The directed graph.
            recommender_n (str): The name of the recommender system.
            int_per_step (int): The number of interactions per step.
            n_recommendations (int): The number of recommendations to make.
            n_steps (int): The number of steps to run the model.
            intervention (str): The intervention policy.
            intervention_prob (float): The probability of intervention.
        """
        self.G = G.copy()
        self.nodes = G.nodes()
        self.n_steps = n_steps
        self.int_per_step = int_per_step
        self.intervention = intervention
        self.intervention_prob = intervention_prob
        self.n_recommendations = n_recommendations
        self.recommender = Recommender(name=recommender_n)

    def run(self):
        """
        Function that runs the model.
        """
        #Here we initialize the number of recommendations
        r = 0
        #Here we initialize the alpha parameter. Alpha is the probability of making a recommendation.
        alpha = self.n_recommendations/((self.n_steps/2)*self.int_per_step*len(self.nodes))
        #Here we run the model for n_steps
        for _ in range(self.n_steps):
            #For each step, we randomly shuffle the nodes to make the interactions
            nodes = np.random.permutation(self.nodes)
            #For each node, we make the interactions
            for node in nodes:
                #If the bernoulli trial is successful, we make a recommendation
                if self.__bernoulli_trial(alpha):
                    #Here we make a recommendation
                    recommended_node = self.recommender.top_k_recommendation(self.G, node_id=node)[0]
                    #Eliminate the edge between the node and a random neighbor to keep attention span
                    random_neighbor = self.__get_random_neighbor(node)
                    #Here we eliminate the edge between the node and the random neighbor
                    self.G.remove_edge(node, random_neighbor)
                    #Here we add the edge between the node and the recommended node
                    self.__add_edge(node, recommended_node)
                    #Here we update the opinions of the nodes
                    self.__bounded_confidence_model(node, recommended_node)
                    #Here we update the number of recommendations
                    r += 1
                    #If the number of recommendations is equal to the number of maximum recommendations, we set alpha to 0
                    if r == self.n_recommendations:
                        alpha = 0
                #If the bernoulli trial is not successful, we make an interaction
                else:
                    random_neighbor = self.__get_random_neighbor(node)
                    self.__bounded_confidence_model(node, random_neighbor)

    def __add_edge(self, node:int, recommended_node:int):
        """
        Function that adds an recommended edge to the graph. With a probability of intervention_prob, the edge is added to a node different from the recommended node.
        This depends on the intervention policy which can be random or opinion diversity.
        
        Args:
            node (int): The node to add the edge.
            recommended_node (int): The recommended node to add the edge.

        Returns:
            bool: True if the edge was added successfully.
        """
        #Here we add the edge to the recommended node and with a probability of intervention_prob we add the edge to a different node
        if self.__bernoulli_trial(self.intervention_prob):
            recommended_node = self.__intervention_policy(node)
        self.G.add_edge(node, recommended_node)
        return True
    
    def __intervention_policy(self, node:int):
        """
        Function that returns the new recommended node based on the intervention policy.

        Args:
            node (int): The node to add the edge.

        Returns:
            int: The new recommended node.
        """
        #Here we check the intervention policy
        if self.intervention == 'random':
            return self.__random_intervention(node)
        elif self.intervention == 'opinion diversity':
            return self.__opinion_diversity_intervention(node)
        else:
            raise ValueError('Intervention policy not found')
        
    def __random_intervention(self, node:int):
        """
        Function that returns a random node that is not a neighbor of the node.

        Args:
            node (int): The node to add the edge.

        Returns:
            int: The new recommended node.
        """
        #First we obtain the nonneighbors of the node
        non_neighbors = list(nx.non_neighbors(self.G, node))
        #Here we return a random non-neighbor
        return np.random.choice(non_neighbors)
    
    def __opinion_diversity_intervention(self, node:int):
        """
        Function that returns a node that is not a neighbor of the node and is selected based on the difference in opinion.

        Args:
            node (int): The node to add the edge.

        Returns:
            int: The new recommended node.
        """
        #First we obtain the no-nneighbors of the node
        non_neighbors = list(nx.non_neighbors(self.G, node))
        #Here we obtain the difference in opinion between the node and the non-neighbors
        diff_opinion = [abs(self.G.nodes[node]['opinion'] - self.G.nodes[non_neighbor]['opinion']) for non_neighbor in non_neighbors]
        #Here we normalize the difference in opinion
        diff_opinion =np.array(diff_opinion)/np.sum(diff_opinion)
        #Here we return a random non-neighbor based on the difference in opinion
        return np.random.choice(non_neighbors, p = diff_opinion)
                    
    
    def __bounded_confidence_model(self, node1: int, node2: int, param: float = 0.2) -> bool:
        """
        This function updates the opinions of two nodes.

        Args:
            node1 (int): The first node.
            node2 (int): The second node.
        """
        #We obtain the opinions of the two nodes
        opinion1 = self.G.nodes[node1]['opinion']
        opinion2 = self.G.nodes[node2]['opinion']
        
        if abs(opinion1 - opinion2) <= param:
            #We update the opinions of the directed node
            self.G.nodes[node1]['opinion'] = opinion1 + param * (opinion2 - opinion1)

        return True
    
    def __get_random_neighbor(self, node:int):
        """
        Function that returns a random neighbor of the node.

        Args:
            node (int): The node to add the edge.

        Returns:
            int: The new recommended node.
        """
        #Here we return a random neighbor
        return np.random.choice(list(self.G.neighbors(node)))
    
    def __bernoulli_trial(self, p:float):
        """
        Function that returns the result of a Bernoulli trial.

        Args:
            p (float): The probability of success.

        Returns:
            bool: The result of the Bernoulli trial.
        """
        #Here we return the result of the Bernoulli trial
        return np.random.binomial(1, p)
