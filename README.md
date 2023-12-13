# Fashion Recommendation: Outfit Compatibility using GNN
<img width="481" alt="image" src="https://github.com/sakshamarora97/outfit-compatibility-scoring/assets/62840042/97ff2dc7-d321-45a5-83af-ff14b0c8ee3b">

## Abstract
Numerous industries have benefited from the use of machine learning, and the fashion industry is no exception. By gaining a better
understanding of what makes a "good" outfit, companies can provide useful product recommendations to their users. In this project,
we follow two existing approaches that employ graphs to represent outfits and use modified versions of the Graph neural network (GNN) frameworks. Both Node-wise Graph Neural Network
(NGNN)[1] and Hypergraph Neural Network (HGNN)[5] aim to
score a set of items according to the outfit compatibility of items.
The data used is the Polyvore Dataset which consists of curated
outfits with product images and text descriptions for each product
in an outfit. We recreate the analysis on a subset of this data and
compare the two existing models on their performance on two
tasks – (1) Fill-in-the-blank (FITB): finding an item that completes
an outfit, and (2) Compatibility prediction: estimating compatibility
of different items grouped as an outfit. We are able to replicate the
results directionally and find that HGNN does have a slightly better
performance on both tasks. On top of replicating the results of the
two papers we also tried to use embeddings generated from a vision
transformer (ViT) and witness enhanced prediction accuracy across
the board.
## Introduction

### 2.1 Objective
The objective of this paper is to explore the use of different graphbased frameworks for the representation of clothing/accessory
items and outfits in the task of fashion recommendation. We aim
to tackle the practical problem of fashion recommendation, specifically what item matches and compliments an outfit. The main
aim is to estimate an outfit compatibility score, a quantitative metric to measure how well different items in an outfit complement
each other. To do this, we leverage two existing implementations
of Graph Neural Networks (GNNs) to create embeddings of all
items in the same outfit. We then compare the performance of these
two techniques against each other. We also look at how different
modalities of representing items influence performance.

<img width="600" alt="image" src="https://github.com/sakshamarora97/outfit-compatibility-scoring/assets/62840042/add8231e-d355-4b27-93c3-e25ab2f584f7">


### 2.2 Dataset
We use the Polyvore dataset which consists of thousands of curated
outfits by experts. Specifically it contains not only the images of
the products but also their textual descriptions and other metadata.
We have explored the dataset in more detail in Section 4.

<img width="300" alt="image" src="https://github.com/sakshamarora97/outfit-compatibility-scoring/assets/62840042/c10be340-325d-4ee0-8352-0e6ea3a9d475">

### 2.3 Method
We use two different methods involving GNNs to create image and
text embeddings. A GNN learns a target node’s representation by
propagating the neighbouring information in an iterative manner
thus ensuring similarity of representation of items linked closely.
The first approach called Node-wise Graph Neural Network (NGNN)
[1] improves on GNN by eschewing parameter sharing to capture
item characteristics better. It uses the category of the item as a
mainstay in message propagation by using weighted edges based
on the category of nodes and also using category-specific item
representation. Finally, it uses an attention mechanism to score the
outfit.
The second approach, HGNN (Hypergraph-GNN) [5], uses hyperedges to denote each outfit. A hypergraph is a generalization
of a graph in which an edge can join any number of nodes. Using
this approach, theoretically, captures more complex interactions between each item in an outfit leading to key features being captured
accurately in the embeddings.

<img width="1029" alt="image" src="https://github.com/sakshamarora97/outfit-compatibility-scoring/assets/62840042/1546b298-e9bf-41f6-8c6a-d0dc3eb323e1">


### 2.4 Comparison Results
We used two tasks to validate our models, namely (1) Fill-In-TheBlank (FITB) tasks where an item is selected from multiple outfit
choices and (2) Compatibility Prediction tasks where we predict of
the compatibility score of a group of items.
The experimental results, which are presented in section 8, firstly,
show that the HGNN representation is marginal improvement over
the NGNN in terms of accuracy by about 1% and 11% in the fill-inthe-blank and compatibility prediction tasks, respectively. Secondly,
the accuracy and AUC metrics are better using a multi-modal approach rather than text-only or visual-only approaches.

<img width="400" alt="image" src="https://github.com/sakshamarora97/outfit-compatibility-scoring/assets/62840042/bac70c3a-ebba-4eb1-831c-53d980b61f69">
<img width="400" alt="image" src="https://github.com/sakshamarora97/outfit-compatibility-scoring/assets/62840042/afbb4ff8-a5ca-4315-94d2-f342002fefdd">

### 2.5 Impact
The fashion graphs of NGNN and HGNN frameworks demonstrated
shows that the potentially costly and tedious work of putting together an outfit and making a fashion decision can be automated to
some extent. Utilizing these tools can help companies in the fashion
industry provide item recommendations for their customers on online platforms, leading to increased usability, buying desire, and
subsequent profit growth.
