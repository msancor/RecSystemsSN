import networkx as nx
import numpy as np
import pandas as pd
from modules.recommender import Recommender
class PROD():
    def __init__(self, G:nx.DiGraph, G_params:dict, recommender_n:str, int_per_step:int, n_recommendations:int, n_steps:int):
        self.G = G
        self.n_steps = n_steps
        self.G_params = G_params
        self.int_per_step = int_per_step
        self.n_recommendations = n_recommendations
        self.A = nx.to_scipy_sparse_matrix(self.G)
        self.recommender = Recommender(name=recommender_n)




    