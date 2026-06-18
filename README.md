# Comparative Analysis of Clustering Algorithms

A practical Python project comparing several clustering techniques on a common dataset. The notebook focuses on how different algorithms form clusters, handle noise, and perform under the same evaluation workflow.

## Project objective

The objective is to demonstrate that clustering algorithms behave differently depending on cluster shape, density, noise, and model assumptions. The project compares results visually and uses simple evaluation measures to support interpretation.

## Algorithms covered

- Agglomerative Clustering
- DBSCAN
- OPTICS
- Mean Shift
- Gaussian Mixture Models
- Hierarchical clustering concepts
- K-Medoids concepts and extensions

## Evaluation approach

The notebook compares algorithms using:

- Number of detected clusters
- Number of points classified as noise
- Silhouette score
- Approximate runtime
- Visual inspection of cluster assignments

## Technologies used

- Python
- NumPy
- pandas
- matplotlib
- scikit-learn
- scikit-learn-extra
- Jupyter Notebook

## Repository structure

```text
notebooks/
  clustering_algorithms_comparison.ipynb
README.md
requirements.txt
LICENSE
.gitignore
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/clustering_algorithms_comparison.ipynb
```

Run the notebook cells from top to bottom to generate the synthetic dataset, train the clustering models, calculate evaluation measures, and display the visual comparisons.

## Interpretation

Density-based methods can identify noise and irregular cluster shapes, while hierarchical methods provide an intuitive grouping structure. Gaussian Mixture Models represent probabilistic clusters, and Mean Shift estimates cluster centers without requiring a fixed number of clusters in advance.

## Limitations

- The current comparison uses a synthetic demonstration dataset.
- Hyperparameters are educational defaults rather than fully optimized values.
- Silhouette score alone does not determine the best clustering method.
- Results may change when the dataset scale, density, or dimensionality changes.

## Future improvements

- Add real-world datasets
- Include K-Medoids as a fully executed model
- Add Davies–Bouldin and Calinski–Harabasz scores
- Compare sensitivity to hyperparameters
- Add dimensionality reduction for high-dimensional data
- Export comparison results as a report

## Author

**Muhammad Kamil Shah**  
BS Data Science

## License

This project is licensed under the MIT License.
