from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
class LFRBenchmark():
    """
    Class to generate a LFR Benchmark Graph with opinions and homophily
    """
    #Here we define the default parameters for the LFR Benchmark Graph
    DEFAULT_POWER_LAW_COEF = 2.75
    DEFAULT_MAX_ITERS = 50
    DEFAULT_AVG_DEG = 12
    DEFAULT_TAU2 = 1.1

    def __init__(self, n:int,  homophily:float, modularity:float, init_opinions:bool=True, seed:int = 123456789, verbose:bool=False):
        """
        Function to initialize the LFR Benchmark Graph
        Args:
            n (int): Number of nodes in the graph
            homophily (float): Homophily parameter. The probability of a node to have the same opinion as the members of its community
            modularity (float): Modularity parameter. The higher the value, the stronger the community structure
            init_opinions (bool): If True, the nodes will be initialized with opinions
            seed (int): Seed for the random number generator
            verbose (bool): If True, print the messages
        """
        self.n = n
        self.seed = seed
        self.verbose = verbose
        self.homophily = homophily
        self.modularity = modularity
        self.init_opinions = init_opinions
        self.G = self.__generate_lfr_benchmark()

    def __initialize_opinions(self, G:nx.DiGraph, labeled_communities:dict):
        """
        Function to initialize the opinions of the nodes in the graph
        Args:
            G (nx.DiGraph): Graph to initialize the opinions
            labeled_communities (dict): Dictionary with the labeled communities
        Returns:
            dict: Dictionary with the initial opinions of the nodes
        """
        #Get the number of communities
        n_communities = len(labeled_communities)
        #Initialize the opinions of the nodes
        initial_opinions ={node:0 for node in G.nodes()}
        #Generate the initial opinions of the communities
        initial_community_opinions = np.random.uniform(size=n_communities)
        #Assign the opinions to the nodes depending on the homophily
        #If the homophily is 1, all the nodes in the community will have the same opinion
        #If the homophily is 0, the opinions will be assigned randomly
        for community in labeled_communities.keys():
            for node in labeled_communities[community]:
                initial_opinions[node] = initial_community_opinions[community]
                if np.random.uniform() < self.homophily:
                    initial_opinions[node] = initial_community_opinions[community]
                else:
                    initial_opinions[node] = np.random.uniform()
        return initial_opinions

    def __generate_lfr_benchmark(self):
        """
        Function to generate the LFR Benchmark Graph
        Returns:
            nx.DiGraph: Generated LFR Benchmark Graph
        """
        #Set the seed for reproducibility
        G = None
        seed = self.seed
        #Generate the LFR Benchmark Graph
        while G is None:
            try:
                G = LFR_benchmark_graph(n=self.n, tau1 = self.DEFAULT_POWER_LAW_COEF,
                                    tau2 = self.DEFAULT_TAU2,
                                    mu = self.modularity,
                                    average_degree = self.DEFAULT_AVG_DEG,
                                    min_community=(self.n//20),
                                    max_iters=self.DEFAULT_MAX_ITERS,
                                    seed=seed)
                if self.verbose:
                    print(f"Generated LFR Benchmark Graph with {self.n} nodes, homophily {self.homophily}, and modularity {self.modularity}")
            #If the number of iterations is exceeded, retry with a different seed multiple times
            except nx.ExceededMaxIterations as e:   
                if self.verbose:
                    print(f"Exceeded Max Iterations. Retrying...")
                seed = np.random.randint(0, 100)
        #Initialize the opinions of the nodes
        if self.init_opinions:
            self.labeled_communities = self.__get_labeled_communities(G)
            initial_opinions = self.__initialize_opinions(G, self.labeled_communities)
            nx.set_node_attributes(G, initial_opinions, 'opinion')
            nx.set_node_attributes(G, initial_opinions, 'initial_opinion')
        #Remove self-loops and convert the graph to a directed graph
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.DiGraph(G)
        return G
    
    def __get_labeled_communities(self, G:nx.DiGraph):
        """
        Function to get the labeled communities of the graph
        Args:
            G (nx.DiGraph): Graph to get the labeled communities
        Returns:
            dict: Dictionary with the labeled communities
        """
        #Here we get the labeled communities
        unique_communities = [tuple(sorted(G.nodes[i]['community'])) for i in G.nodes]
        unique_communities = list(set(unique_communities))
        labeled_communities = {i:community for i, community in enumerate(unique_communities)}
        return labeled_communities
    
    def plot_graph(self, ax: plt.Axes = None, init_opinions:bool = True):
        """
        Function to plot the LFR Benchmark Graph
        Args:
            ax (plt.Axes): Matplotlib Axes to plot the graph
            init_opinions (bool): If True, plot the initial opinions of the nodes
        Returns:
            None
        """
        #Get the positions of the nodes
        nodes_positions = self.__get_community_positions()
        if init_opinions:
            nodes_color = [self.G.nodes[n]["initial_opinion"] for n in self.G.nodes]
        else:
            nodes_color = [self.G.nodes[n]["opinion"] for n in self.G.nodes]
        nodes_size = [self.G.degree[n]*10 for n in self.G.nodes]
        #Plot the graph
        nx.draw(self.G, nodes_positions, node_color=nodes_color, node_size=nodes_size, with_labels=False,
                alpha=0.8, width=0.15, edgecolors="k", cmap="coolwarm", ax=ax, arrows=False)
        if ax is not None:
            ax.axis("on")
        return None

    
    def __get_community_positions(self):
        """
        Function to get the positions of the nodes in the graph
        Returns:
            dict: Dictionary with the positions of the nodes
        """
        np.random.seed(self.seed)
        COMMUNITY_DISTANCE_SCALE = 100
        COMMUNITY_DISTANCE_DIVISOR = 5
        NODE_POSITION_OFFSET_SCALE = 0.2
        NODE_POSITION_RANDOM= 0.025
        ADDITIONAL_OFFSET_CHOICES = [-0.1, 0.1]

        #Get the number of communities and the number of nodes per community
        number_of_communities = len(self.labeled_communities)
        nodes_per_community= {k: len(v) for k, v in self.labeled_communities.items()}
        #Generate the distance between the communities
        G_distance = nx.Graph()
        for u in range(number_of_communities):
            for v in range(number_of_communities):
                weight = COMMUNITY_DISTANCE_SCALE / (COMMUNITY_DISTANCE_DIVISOR * (nodes_per_community[u] + nodes_per_community[v]))
                G_distance.add_edge(u, v, weight=weight)

        community_positions = nx.spring_layout(G_distance, weight='weight', seed=self.seed)
        node_positions ={}
        #Generate the positions of the nodes
        for centroid in G_distance.nodes():
            centr_pos = community_positions[centroid]
            nodes_in_community = self.labeled_communities[centroid]
            sorted_nodes_in_community = sorted(nodes_in_community, key=lambda x: self.G.degree[x], reverse=True)
            #Map the nodes to the positions with an offset depending on the degree
            for i, node in enumerate(sorted_nodes_in_community):
                offset = NODE_POSITION_OFFSET_SCALE * i * np.random.uniform(low=-NODE_POSITION_RANDOM, high=NODE_POSITION_RANDOM, size=2)
                additional_offset = np.array([np.random.choice(ADDITIONAL_OFFSET_CHOICES) for _ in range(2)])
                node_positions[node] = centr_pos + offset + additional_offset
        
        return node_positions
    


