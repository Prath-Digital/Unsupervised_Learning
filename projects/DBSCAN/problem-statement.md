# Unsupervised User Behavior Analysis Using DBSCAN

An organization has collected anonymized daily behavioral data from 500 users. The dataset contains information about screen usage, social-media usage, study/work time, sleep duration, completed tasks, and gaming time.

The dataset is pre-cleaned and contains no predefined class labels, so the analysis must be performed using an unsupervised learning approach.

Using DBSCAN (Density-Based Spatial Clustering of Applications with Noise), perform the following two objectives using the same dataset:

1. DBSCAN Clustering:
   Identify naturally occurring groups of users based on similarities in their behavioral patterns. Determine the clusters discovered by DBSCAN and describe the characteristics of each cluster using the available behavioral features.

2. DBSCAN Anomaly Detection:
   Identify users whose behavioral patterns significantly differ from the dense groups. Treat DBSCAN's noise observations (`label = -1`) as potential anomalies and analyze why these users may be considered unusual compared with the discovered clusters.
