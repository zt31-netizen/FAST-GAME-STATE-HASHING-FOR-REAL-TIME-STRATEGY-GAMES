# FAST-GAME-STATE-HASHING-FOR-REAL-TIME-STRATEGY-GAMES
We propose to develop a GPU-parallelized approximate nearest neighbor hashing system for real-time lookup of game board states in auto-battler / strategy games (TFT or chess variants). The goal is to enable fast estimation of win probabilities or move evaluation by retrieving similar past states from a large database, without comparing all states.

Problem

Computing similarity over high-dimensional game states (bitboard encodings, piece positions, future states) is computationally expensive in real time. Exact nearest neighbor search scales linearly in the number of stored states, which is infeasible at large scales. Approximate methods (locality sensitive hashing, product quantization) are commonly used in ML/vision/IR domains to speed up similarity search in high dimensions (e.g. LSH, PQ, graph-based ANN) [Zhang et al]. Meanwhile, GPU architectures and data-parallel hashing techniques have matured (e.g. “Data-Parallel Hashing Techniques for GPU Architectures”) [Lessley].

State representation & embedding.

We will map discrete game states (bitboards, piece vectors, augmented features) into dense vector embeddings (e.g. via learned encoders or handcrafted features).

Hash indexing & ANN scheme.

We will implement and compare multiple hashing / ANN schemes such as Locality Sensitive Hashing (LSH) (e.g. random projections), data-dependent / deep hashing (learning binary codes), and  hybrid / graph + hash methods (e.g. leveraging methods in BANG for GPU-based ANN).
We will build both CPU baseline and GPU-accelerated versions of these indexing/query pipelines, aiming to exploit parallelism in different areas.

Similarity query & win-probability estimator.

Upon a query state, we will retrieve its kkk-nearest neighbors (or top kkk hash-bucket candidates), then compute a small number of exact distances among them and aggregate prior outcomes (win rates) to estimate the win probability of the current state.

Evaluation & comparison.

We will benchmark the methods on large state databases. Metrics to compare include throughput (queries per second), recall / approximation error (how often the true nearest neighbor is returned), and accuracy of win-probability estimates (vs. ground truth computed by exhaustive search or simulation). We may also compare with off-the-shelf ANN libraries (e.g. Faiss or GPU-ANN frameworks) for baseline.

This project will combine approximate similarity search, GPU-accelerated hashing/ANN, and game-state analysis to produce a fast lookup engine that can support real-time strategic evaluation in games like TFT.
