from typing import Tuple
from tqdm import tqdm
import networkx as nx
import random

class BoundedConfidence():
    CONVERGENCE_PARAM = 0.5
    DIVERGENCE_PARAM = 0.5

    def __init__(self, G: nx.Graph, agreement_threshold: float, disagreement_threshold: float):
        self.G = G
        self.edges = list(G.edges())
        self.agreement_threshold = agreement_threshold
        self.disagreement_threshold = disagreement_threshold

    def run(self, max_iter: int) -> bool:
        """
        This function runs the modified bounded confidence model for a given number of iterations.

        Args:
            max_iter (int): The maximum number of iterations.

        Returns:
            bool: True if the model ran successfully.
        """
        #Here we initialize the opinions of the nodes in the graph
        self.__initialize_opinions()
        #Here we initialize a list to store a tuple with the opinions of the nodes
        opinions = []
        #We run the bounded confidence model for the given number of iterations
        for t in tqdm(range(max_iter)):
            #Here we add the time step and opinions of the nodes to the list
            opinions.extend([(t, self.G.nodes[node]['opinion']) for node in self.G.nodes()])
            #Now we repeat the bounded confidence model for 1000 sampled edges
            for _ in range(1000):
                #Here we sample an edge from the graph
                node1, node2 = self.__sample_edge()
                #We update the opinions of the two nodes
                self.__update_opinions(node1, node2)
        
        return opinions
    
    def __update_opinions(self, node1: int, node2: int) -> bool:
        """
        This function updates the opinions of two nodes.

        Args:
            node1 (int): The first node.
            node2 (int): The second node.
        """
        #We obtain the opinions of the two nodes
        opinion1 = self.G.nodes[node1]['opinion']
        opinion2 = self.G.nodes[node2]['opinion']
        
        if abs(opinion1 - opinion2) <= self.agreement_threshold:
            #We update the opinions of the two nodes
            self.G.nodes[node1]['opinion'] = opinion1 + self.CONVERGENCE_PARAM * (opinion2 - opinion1)
            self.G.nodes[node2]['opinion'] = opinion2 + self.CONVERGENCE_PARAM * (opinion1 - opinion2)

        elif abs(opinion1 - opinion2) > self.disagreement_threshold:
            #We update the opinions of the two nodes
            updated_opinion1 = opinion1 - self.DIVERGENCE_PARAM * (opinion2 - opinion1)
            updated_opinion2 = opinion2 - self.DIVERGENCE_PARAM * (opinion1 - opinion2)
            #If the updated opinions are within the range [0, 1], we update the opinions
            #Otherwise, we set the opinions to the closest bound
            if opinion1 > opinion2:
                self.G.nodes[node1]['opinion'] = min(1, updated_opinion1)
                self.G.nodes[node2]['opinion'] = max(0, updated_opinion2)

            else:
                self.G.nodes[node1]['opinion'] = max(0, updated_opinion1)
                self.G.nodes[node2]['opinion'] = min(1, updated_opinion2)
        else:
            pass

        return True
    
    def __sample_edge(self) -> Tuple[int, int]:
        """
        This function samples an edge from the graph.

        Returns:
            Tuple[int, int]: The sampled edge.
        """
        #We sample an edge from the graph
        return random.choice(self.edges)
    
    def __initialize_opinions(self) -> bool:
        """
        This function initializes the opinions of the nodes in the graph.

        Returns:
            bool: True if the opinions were initialized successfully.
        """
        #We initialize the opinions of the nodes in the graph
        for node in self.G.nodes():
            #We initialize the initial opinion of the node
            self.G.nodes[node]['initial_opinion'] = random.uniform(0, 1)
            #We set the opinion of the node to the initial opinion
            self.G.nodes[node]['opinion'] = self.G.nodes[node]['initial_opinion']
        
        return True
