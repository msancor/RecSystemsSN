# The Effect of Friend Recommenders on the Formation of Echo Chambers and Polarization in Social Networks

This is a Github repository created to submit the final project of the **Social Networks & Online Markets (SNOM)** course for the MSc. in Data Science at the Sapienza University of Rome. The main goal of this project was to reproduce and extend some of the results presented by the published article: *Cinus, F., Minici, M., Monti, C., & Bonchi, F. (2022). The Effect of People Recommenders on Echo Chambers and Polarization. Proceedings of the International AAAI Conference on Web and Social Media, 16(1), 90-101. https://doi.org/10.1609/icwsm.v16i1.19275*.

We acknowledge the authors as the full responsibles of this project and this reproduction is made only for didactic purposes. If you attempt to reproduce the results presented on this repository, please cite the original article:

```markdown
@article{Cinus_Minici_Monti_Bonchi_2022,
title={The Effect of People Recommenders on Echo Chambers and Polarization},
volume={16},
url={https://ojs.aaai.org/index.php/ICWSM/article/view/19275},
DOI={10.1609/icwsm.v16i1.19275},
number={1},
journal={Proceedings of the International AAAI Conference on Web and Social Media},
author={Cinus, Federico and Minici, Marco and Monti, Corrado and Bonchi, Francesco},
year={2022},
month={May},
pages={90-101}}
```


--- 
## What's inside this repository?

1. `README.md`: A markdown file that explains the content of the repository.

2. `main.ipynb`: A Jupyter Notebook file containing all the relevant code to reproduce the plots included in the `SocialNetworks_FinalProject.pdf` file.

3. ``modules/``: A folder including 5 Python modules used to make simulations and plotting in `main.ipynb`. The files included are:

    - `__init__.py`: A *init* file that allows us to import the modules into our Jupyter Notebook.

    - `random_graph.py`: A Python file including a `LFRBenchmark` class designed to build a modified version of the LFR-Benchmark random network model to allow for homophily of opinions.

    - `recommender.py`: A Python file including a `Recommender` class designed to build friend recommendation algorithms including: WTF, Node2Vec, OBA, PPR and Random recommenders.

    - `measurement.py`: A Python file including a `Measurement` class designed to define metrics to measure the effect of friend recommenders on social network emergent phenomena.

    - `PROD.py`: A Python file including a `PROD` class designed to implement a modified version of the PROD algorithm defined by Cinus et al. (2022) in order to simulate interactions between a social network model, an opinion dynamics model, and a friend recommendation system.

    - `miscellaneous.py`: A Python file including different functions to make plots and other experiments on the `main.ipynb` file.

4. ``scripts/``: A folder including 2 Python scripts used to make simulations and obtain the data used to plot on the `main.ipynb` file. The files included are:

    - ``simulations.py``: A python script used to make simulations by performing the PROD algorithm on different sequences of networks varying modularity and homophily parameters.

    - ``intervention_sims.py``: A python script used to make simulations by performing the PROD algorithm on different sequences of networks by making PPR recommendations and testing different intervention policies.

5. ``.gitignore``: A python gitignore file.

6. `LICENSE`: A file containing an MIT permissive license.

## Datasets

The data used to work in this repository was obtained via simulations by running the `simulations.py` and `intervention_sims.py` folder. In order to obtain the data used for plotting follow the steps below:

**1.** Create the directory where you will save the obtained `json` files after you perform simulations. Specifically, run in your terminal the following command:

```bash
mkdir results
```

Make sure you create these folders in the same directory you've saved the `main.ipynb` file on.

**2.** Run the `simulations.py` file by selecting the name of the recommendation system you want to use, the name of the intervention policy you want to use (if you want one), and the intervention probability. For example, if I want to run simulations for the Personalized PageRank (PPR) algorithm without intervention policies I would have to run the following command:

```bash
python3 scripts/simulations.py pagerank None 0
```

Alternatively, if you want to run the predeterminate simulations of a recommender system with intervention using the `intervention_sims.py`, you just have to run this script by specifying the homophily and modularity values of the network you want to test. For example, if  I want to run simulations for the Personalized PageRank (PPR) algorithm on a network with modularity parameter $\mu = 0.05$ and homophily $\eta = 0.8$ I would have to run the following command:

```bash
python3 scripts/intervention_sims.py pagerank 0.05 0.8
```


---

**Author:** Miguel Ángel Sánchez Cortés

**Email:** sanchezcortes.2049495@studenti.uniroma1.it

*MSc. in Data Science, Sapienza University of Rome*
