import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/cluster', methods=['POST'])
def kmeans():
    data = request.get_json()
    initialData = data['data']

    # Convert the data to a DataFrame
    df = pd.DataFrame(initialData)

    # Select the features cases and rainfall for clustering
    X = df[["cases", "rainfall", "population"]]

    # Choose the number of clusters (K)
    num_clusters = 3

    # Manually set the initial cluster centers using the three data points you provided
    initial_cluster_centers = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [1, 1, 1]
    ])

    # Create a KMeans instance with the initial cluster centers
    kmeans = KMeans(n_clusters=num_clusters, init=initial_cluster_centers, n_init=1)

    # Fit the KMeans model to your data and get all labels and cluster centers from each iteration
    all_labels = []
    all_values = []
    all_iterations = []

    for _ in range(kmeans.n_init):
        kmeans.fit(X)
        all_labels.append(kmeans.labels_.tolist())
        all_values.append(X.values.tolist())
        all_iterations.append({
            'labels': kmeans.labels_.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'values': X.values.tolist()
        })

    # Get the final cluster assignments for each data point
    labels = kmeans.labels_

    # Add the final cluster labels and values to the DataFrame
    df["Cluster"] = labels
    df["Values"] = X.values.tolist()

    # Get the Kecamatan, Cluster, and Values information as a DataFrame
    kecamatan_cluster_df = df[["district", "Cluster"]]

    # Convert the DataFrame to a dictionary to match the desired format
    kecamatan_cluster_list = kecamatan_cluster_df.to_dict(orient="records")

    # Evaluate the clustering using Silhouette Score
    silhouette_avg = silhouette_score(X, labels)
    
    result_data = {
        "num_clusters": num_clusters,
        'kecamatan_cluster': kecamatan_cluster_list,
        'silhouette_avg': silhouette_avg,
        'all_labels': all_labels,
        'all_values': all_values,
        'all_iterations': all_iterations,
        'initial_cluster_centers': initial_cluster_centers.tolist()  # Convert to list for JSON serialization
    }

    return jsonify(result_data)
    
@app.route('/', methods=['GET'])
def hello():
    
    return "Backend Dengue"


if __name__ == '__main__':
    app.run(debug=True)
