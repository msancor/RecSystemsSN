import networkx as nx
import numpy as np
import heapq as hq
from node2vec import Node2Vec
class Recommender():
    def __init__(self, name:str):
        self.name = name

    def top_k_recommendation(self, G:nx.DiGraph,recommender_name:str, node_id:int, k:int):
        if recommender_name == "pagerank":
            return self.__personalized_pagerank_recommender(G, node_id, k)
        elif recommender_name == "node2vec":
            return self.__node2vec_recommender(G, node_id, k)
        else:
            raise ValueError("Recommender Not Found. Please choose between pagerank and node2vec")
        
    def __personalized_pagerank_recommender(self, G:nx.DiGraph, node_id:int, k:int):
        DAMPING_FACTOR = 0.85
        neighbor_list = self.__get_neighbors(G, node_id)
        personalization_dict = {id:0 for id in G.nodes()}
        personalization_dict[node_id] = 1
        ppr = nx.pagerank(G, alpha=DAMPING_FACTOR, personalization=personalization_dict)
        recommendation_dict = {key: value for key, value in ppr.items() if key not in neighbor_list and key != node_id}
        top_k = self.__get_top_k(recommendation_dict, k)
        return top_k
    
    def __node2vec_recommender(self, G:nx.DiGraph, node_id:int, k:int):
        neighbor_list = self.__get_neighbors(G, node_id)
        node2vec = Node2Vec(G, seed=128, quiet=True)
        model = node2vec.fit(seed=128)
        recommendation_dict = {key: self.__cosine_similarity(model.wv[str(node_id)], model.wv[str(key)]) for key in G.nodes() if key not in neighbor_list and key != node_id}
        top_k = self.__get_top_k(recommendation_dict, k)
        return top_k     
    
    def __get_neighbors(self, G:nx.DiGraph, node_id:int):
        return list(G.neighbors(node_id))
    
    def __cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def __get_top_k(self, recommendation_dict:dict, k:int):
        heap_list= []
        for key, value in recommendation_dict.items():
            hq.heappush(heap_list, ((-1)*value, key))
        top_k = [(key, (-1)*value) for value, key in hq.nsmallest(k, heap_list)]
        return top_k
