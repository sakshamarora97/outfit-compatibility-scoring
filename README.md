# outfit-compatibility-scoring
Course Project for CSE 6240

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
