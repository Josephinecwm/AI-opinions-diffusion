# -*- coding: utf-8 -*-
"""Untitled0.ipynb

# %%libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman, modularity
from itertools import count
import itertools

# %%read file
G = nx.read_graphml("trading_floor.xml")
G.nodes.data()

# %%knowledge exchange network 
#color mapping for attributes "ai"
groups = set(nx.get_node_attributes(G,'ai').values())
mapping = dict(zip(sorted(groups),count()))
nodes = G.nodes()
colors = [mapping[G.nodes[n]['ai']] for n in nodes]

#set node size based on the degree of the nodes
degree = dict(G.degree)
node_size=[v * 300 for v in degree.values()]

#plot the figure
fig = plt.figure(figsize=(30, 25))
#set the position
pos = nx.spring_layout(G, k=0.5, seed=000)
#draw the network
nc = nx.draw_networkx_nodes(G, 
                            pos=pos, 
                            alpha=0.7,
                            node_color=colors,
                            node_size = node_size,
                            cmap="coolwarm")
nx.draw_networkx_edges(G,
                       pos=pos,
                       edge_color='grey',
                       alpha=0.5)
nx.draw_networkx_labels(G, pos=pos)
# set colorbar to determine their attributes score
plt.colorbar(nc) 
plt.axis("off")
plt.show()
degree = dict(G.degree)

# %%physical layout of the trading floor
#create a list of the x and y position 
po_x= nx.get_node_attributes(G,'x_pos')
po_y= nx.get_node_attributes(G,'y_pos')
X_list=list(po_x.values())
Y_list=list(po_y.values())

#create a new attribute of 'pos' for each node
for node, x, y in zip(G.nodes, X_list, Y_list):
  G.nodes[node]["pos"] = (x,y)

#plot the network
fig = plt.figure(figsize=(30, 25))

#set the position of the network
pos_seat= nx.get_node_attributes(G,'pos')

#draw the network
nc = nx.draw_networkx_nodes(G, 
                            pos=pos_seat, 
                            node_size=node_size,
                            alpha=0.7,
                            node_color=colors,
                            cmap="coolwarm")
nx.draw_networkx_edges(G,
                       pos=pos_seat,
                       edge_color='grey',
                       alpha=0.5)
nx.draw_networkx_labels(G, pos=pos_seat)
# set colorbar to determine their attributes score
#plt.colorbar(nc) 
plt.axis("off")
plt.show()

# %%Node Centrality
#compute centrality measures
import seaborn as sns
from networkx.algorithms import centrality, degree_centrality
from networkx.algorithms import betweenness_centrality
from networkx.algorithms import eigenvector_centrality

# degree centrality
dg = degree_centrality(G)
# betweenness centrality
bc = betweenness_centrality(G)
# eigenvector centrality
ec = eigenvector_centrality(G)

# degree centrality
name = []
centrality = []
for key, value in dg.items():
    name.append(key)
    centrality.append(value)

# betweeness centrality
name = []
betweenness = []
for key, value in bc.items():
    name.append(key)
    betweenness.append(value)

# eigenvector_centrality
name = []
eigenvector = []
for key, value in ec.items():
    name.append(key)
    eigenvector.append(value)

# %% draw the correlation among three centralities
#set share y axis into false to keep their orginal ticks
fig, axs = plt.subplots(3, 3, figsize=(12, 12),sharex='col', sharey=False)
n = G.number_of_nodes() 
# degree centrality
axs[0, 0].hist(centrality) #calculate the percentage of distribution
axs[0, 0].set_ylabel("degree centrality", size=15)
axs[0,0].tick_params(axis='both',labelsize=10)
#degree-betweenness
axs[0, 1].scatter(betweenness, centrality, s=13, marker='o')
axs[0,1].set_title('Correlation among different centralities and its distribution',size=20,pad=20)
axs[0,1].tick_params(axis='both',labelsize=10)
#degree-eigen
axs[0, 2].scatter(eigenvector, centrality, s=13,marker='o')
axs[0,2].tick_params(axis='both',labelsize=10)
#betweenness-degree
axs[1, 0].scatter(centrality, betweenness, s=13,marker='o')
axs[1, 0].set_ylabel("betweenness", size=15)
axs[1,0].tick_params(axis='both',labelsize=10)
#betweenness
axs[1, 1].hist(betweenness,weights= np.ones(n)/n)
axs[1,1].tick_params(axis='both',labelsize=10)
#betweenness-eigen
axs[1, 2].scatter(eigenvector, betweenness, s=13,marker='o')
axs[1,2].tick_params(axis='both',labelsize=10)
#eigen-degree
axs[2, 0].scatter(centrality,eigenvector,s=13,marker='o')
axs[2, 0].set_ylabel("eigenvector", size=15)
axs[2, 0].set_xlabel("degree centrality", size=15)
axs[2,0].tick_params(axis='both',labelsize=10)
#eigen-betweenness
axs[2, 1].scatter(betweenness,eigenvector,s=13,marker='o')
axs[2, 1].set_xlabel("betweenness", size=15)
axs[2,1].tick_params(axis='both',labelsize=10)
#eigen centrality
axs[2, 2].hist(eigenvector,weights= np.ones(n)/n)
axs[2, 2].set_xlabel("eigenvector", size=15)
axs[2,2].tick_params(axis='both',labelsize=10)

# %%Descriptive analysis
#degree distribution
n_bins=20
node_degree = [degree for node, degree in nx.degree(G)]
fig, ax = plt.subplots(tight_layout=True)
ax.hist(node_degree, bins = n_bins)
ax.set_xlabel('Degree(k)')
ax.set_ylabel('Frequency')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_title("Degree distribution plot")
#plt.axis("off")
plt.show()

#cumulative degree distribution
n_bins=20
node_degree = [degree for node, degree in nx.degree(G)]
fig, ax = plt.subplots(tight_layout=True)
ax.hist(node_degree, density=True, cumulative=1, bins=n_bins)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_title("Cumulative Degree distribution plot")
ax.set_xlabel("Degree(k)")
plt.show()

# %% assess dydic similarity
similarity = {}
for u, v in G.edges():
  key = "{}-{}".format(u,v)
  value = np.abs(G.nodes[u]["ai"] - G.nodes[v]["ai"]) 
  similarity[key] = value
print(similarity)
df = pd.DataFrame([i[1] for i in similarity.items()], columns=['Dyadic similarity'])
df.describe()

# %%community detection
solutions = girvan_newman(G)
k = 20
# register modularity scores
modularity_scores = dict()
comm=[]
# iterate over solutions
for community in itertools.islice(solutions, k):
    solution = list(sorted(c) for c in community)
    comm.append(solution)
    score = modularity(G, solution)
    modularity_scores[len(solution)] = score
print(score)
# plot modularity data
fig = plt.figure(figsize=(10,8))
pos = list(modularity_scores.keys())
values = list(modularity_scores.values())
ax = fig.add_subplot(1, 1, 1)
ax.stem(pos, values)
ax.set_xticks(pos)
ax.set_xlabel(r'Number of communities detected')
ax.set_ylabel(r'Modularity score')
plt.show()

# 12 communities are detected
# plot network graph of communities
comm14 = comm[12]
for i in range(len(comm14)):
    comm14[i].insert(0,i)
dict_com = {}
for l2 in comm14:
    dict_com[l2[0]]=l2[1:]
import random
nodes=G.nodes()
numcolor = 14
colorlist = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
for k in range(numcolor)]
nodecolor = []
for n in nodes:
    for i in range(len(dict_com)):
        if n in dict_com[i]:
            nodecolor.append(colorlist[i])

# identfy community base on physical location network
node_size2=[v * 100 for v in degree.values()]
fig = plt.figure(figsize=(20, 15))
label2 = nx.get_node_attributes(G, 'ai')
nx.draw_networkx_nodes(G,
                       pos_seat,
                       node_color=nodecolor,
                       node_size=node_size2,
                       alpha=0.7)
nx.draw_networkx_edges(G,
                      pos_seat,
                      edge_color='grey',
                      alpha=0.5)
#draw labels of AI attribute score
nx.draw_networkx_labels(G, pos=pos_seat, labels=label2)
plt.axis("off")
plt.show()

# identfy community base on knowledge exchange network
fig = plt.figure(figsize=(20, 15))
pos = nx.layout.spring_layout(G, k = 0.5, seed = 000)
nx.draw_networkx_nodes(G,
                       pos,
                       node_color=nodecolor,
                       node_size=node_size2,
                       alpha=0.7)
nx.draw_networkx_edges(G,
                       pos,
                       edge_color='grey',
                       alpha=0.5)
nx.draw_networkx_labels(G, pos=pos, labels=label2)
plt.axis("off")
plt.show()

# %% initialize the diffusion process
# parameters
# --+ pay-off of adopting the new behavior
a = 1
# --+ pay-off of the status quo (not changing)
b = 1

# %% model the diffusion process
#the degree of G
degree = nx.degree(G)

#create a list of adopters
adopters = []
for node in G.nodes:
    G.nodes[node]["adopting"] = 0

#new adopters - nodes with an AI attribute score higher than 5
early_adopters = [n for n in G.nodes() if G.nodes[n]['ai'] > 5]


#expand the set of adopters
adopters.extend(early_adopters)

#adopt node attributes
for adopter in adopters:
    G.nodes[adopter]['adopting'] = 1


#draw the network based on the seating position
colors = []
for n in G.nodes():
    if G.nodes[n]['adopting'] == 1:
        colors.append('orange')
    else:
        colors.append('white')
fig = plt.figure(figsize=(20, 15))
nx.draw_networkx_nodes(G,
                       pos_seat,
                       node_color=colors,
                       node_size=node_size,
                       alpha=0.7)
nx.draw_networkx_edges(G,
                      pos_seat,
                      edge_color='grey',
                      alpha=0.5)
nx.draw_networkx_labels(G, pos=pos_seat)
plt.axis("off")
plt.show()

# %% let's simulate what happens in the following periods as nodes make decisions
for focal in nodes:
    # count adopting neighbors
    focal_nbrs = list(G.neighbors(focal))
    p = np.sum([G.nodes[nbr]['adopting'] for nbr in focal_nbrs])
    # pay-off of adopting new behavior
    d = G.degree(focal)
    a_payoff = p * a
    b_payoff = (d - p ) * b
    # decision to adopt
    if (G.nodes[focal]['adopting'] == 0) & (a_payoff > b_payoff):
        G.nodes[focal]['adopting'] = 1
        adopters.extend(focal)
    else:
        pass
# outcome of the cascading behavior
adopters

# %% draw the network based on seating position
colors = []
for n in G.nodes():
    if G.nodes[n]['adopting'] == 1:
        colors.append('red')
    else:
        colors.append('white')
fig = plt.figure(figsize=(20, 15))
nx.draw_networkx_nodes(G,
                       pos_seat,
                       node_color=colors,
                       node_size=node_size,
                       alpha=0.7)
nx.draw_networkx_edges(G,
                      pos_seat,
                      edge_color='grey',
                      alpha=0.5)
nx.draw_networkx_labels(G, pos=pos_seat)
plt.axis("off")
plt.show()

# %% descriptive statistics for non-adopters
#assign degree and betweenness centrlity as a node attribute
degree_list = [d for n, d in G.degree()]
for node, d in zip(G.nodes, degree_list):
  G.nodes[node]["degree"] = d
for node, c in zip(G.nodes, betweenness):
  G.nodes[node]["betweenness"] = c
G.nodes.data()

#get the descriptive statistics
non_adopter_d=[]
non_adopter_b=[]
for n in G.nodes():
  if G.nodes[n]['adopting'] ==0:
    non_adopter_d.append(G.nodes[n]["degree"])
    non_adopter_b.append(G.nodes[n]["betweenness"])

  else: pass
df_1 = pd.DataFrame(non_adopter_d,columns=['degree'])
df_2 = pd.DataFrame(non_adopter_b,columns=['betweenness'])

df_1.describe()
df_2.describe()

