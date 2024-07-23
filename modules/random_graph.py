from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
class LFRBenchmark():
    DEFAULT_POWER_LAW_COEF = 2.75
    DEFAULT_MAX_ITERS = 50
    DEFAULT_AVG_DEG = 12
    DEFAULT_TAU2 = 1.1

    def __init__(self, n:int,  homophily:float, modularity:float, init_opinions:bool=True, seed:int = 128, verbose:bool=False):
        self.n = n
        self.seed = seed
        self.verbose = verbose
        self.homophily = homophily
        self.modularity = modularity
        self.init_opinions = init_opinions
        self.G = self.__generate_lfr_benchmark()

    def __initialize_opinions(self, G:nx.DiGraph, labeled_communities:dict):
        np.random.seed(self.seed)
        n_communities = len(labeled_communities)
        initial_opinions ={node:0 for node in G.nodes()}
        initial_community_opinions = np.random.uniform(size=n_communities)
        for community in labeled_communities.keys():
            for node in labeled_communities[community]:
                initial_opinions[node] = initial_community_opinions[community]
                if np.random.uniform() < self.homophily:
                    initial_opinions[node] = initial_community_opinions[community]
                else:
                    initial_opinions[node] = np.random.uniform()
        return initial_opinions

    def __generate_lfr_benchmark(self):
        G = None
        seed = self.seed
        np.random.seed(seed)
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
            except nx.ExceededMaxIterations as e:   
                if self.verbose:
                    print(f"Exceeded Max Iterations. Retrying...")
                seed += np.random.randint(1, 100)
        if self.init_opinions:
            self.labeled_communities = self.__get_labeled_communities(G)
            initial_opinions = self.__initialize_opinions(G, self.labeled_communities)
            nx.set_node_attributes(G, initial_opinions, 'opinion')

        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.DiGraph(G)
        return G
    
    def __get_labeled_communities(self, G:nx.DiGraph):
        unique_communities = [tuple(sorted(G.nodes[i]['community'])) for i in G.nodes]
        unique_communities = list(set(unique_communities))
        labeled_communities = {i:community for i, community in enumerate(unique_communities)}
        return labeled_communities
    
    def plot_graph(self, ax: plt.Axes = None):
        nodes_positions = self.__get_community_positions()
        nodes_color = [self.G.nodes[n]["opinion"] for n in self.G.nodes]
        nodes_size = [self.G.degree[n]*10 for n in self.G.nodes]
        nx.draw(self.G, nodes_positions, node_color=nodes_color, node_size=nodes_size, with_labels=False,
                alpha=0.8, width=0.15, edgecolors="k", cmap="coolwarm", ax=ax, arrows=False)
        if ax is not None:
            ax.axis("on")
        return None

    
    def __get_community_positions(self):
        np.random.seed(self.seed)
        COMMUNITY_DISTANCE_SCALE = 100
        COMMUNITY_DISTANCE_DIVISOR = 5
        NODE_POSITION_OFFSET_SCALE = 0.2
        NODE_POSITION_RANDOM= 0.025
        ADDITIONAL_OFFSET_CHOICES = [-0.1, 0.1]

        number_of_communities = len(self.labeled_communities)
        nodes_per_community= {k: len(v) for k, v in self.labeled_communities.items()}
        
        G_distance = nx.Graph()
        for u in range(number_of_communities):
            for v in range(number_of_communities):
                weight = COMMUNITY_DISTANCE_SCALE / (COMMUNITY_DISTANCE_DIVISOR * (nodes_per_community[u] + nodes_per_community[v]))
                G_distance.add_edge(u, v, weight=weight)

        community_positions = nx.spring_layout(G_distance, weight='weight', seed=self.seed)
        node_positions ={}
        
        for centroid in G_distance.nodes():
            centr_pos = community_positions[centroid]
            nodes_in_community = self.labeled_communities[centroid]
            sorted_nodes_in_community = sorted(nodes_in_community, key=lambda x: self.G.degree[x], reverse=True)

            for i, node in enumerate(sorted_nodes_in_community):
                offset = NODE_POSITION_OFFSET_SCALE * i * np.random.uniform(low=-NODE_POSITION_RANDOM, high=NODE_POSITION_RANDOM, size=2)
                additional_offset = np.array([np.random.choice(ADDITIONAL_OFFSET_CHOICES) for _ in range(2)])
                node_positions[node] = centr_pos + offset + additional_offset
        
        return node_positions
    


