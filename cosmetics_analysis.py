"""
Feature selection (mutual information), dimensionality reduction (PCA)
and clustering (KMeans) on a cosmetics chemical dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Change this to the location of your CSV file
DATA_PATH = "Cosmectics_chemical.csv"
TARGET_COL = "PrimaryCategory"

# -----------------------------------------
# Load dataset
# -----------------------------------------
df = pd.read_csv(DATA_PATH)
print(" Dataset Loaded Successfully.")
print(f" Total rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(" Columns:", list(df.columns))

# Drop rows missing essential fields, fill the rest
df = df.dropna(subset=["ProductName", "PrimaryCategory", "ChemicalCount"])
df.fillna("Unknown", inplace=True)
df = df.astype(str)

# -----------------------------------------
# Encode categorical columns
# -----------------------------------------
le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
print("\n Label Encoding Completed.")

# -----------------------------------------
# Feature selection with mutual information
# -----------------------------------------
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]
print(f"\n Target Column: '{TARGET_COL}'")
print(f" Feature Shape: {X.shape}")
print(f" Unique Classes in Target: {y.nunique()}")

print("\n Calculating Mutual Information Scores...")
mi_scores = mutual_info_classif(X, y)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\n Top 5 Important Features:")
print(mi_series.head(5))

plt.figure(figsize=(10, 6))
sns.barplot(x=mi_series, y=mi_series.index, palette="viridis")
plt.title("Mutual Information Scores for Feature Selection")
plt.xlabel("Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# -----------------------------------------
# Dimensionality reduction with PCA
# -----------------------------------------
print("\n Performing PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f" Explained Variance Ratio: {pca.explained_variance_ratio_}")

# -----------------------------------------
# Clustering with KMeans
# -----------------------------------------
print("\n Applying KMeans Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)
print(" First 10 Cluster Labels:", labels[:10])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2", s=100)
plt.title("PCA Clustering of Cosmetics Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("pca_clustering.png")
plt.show()

print("\n Feature Selection, PCA, and Clustering Completed.")
print(" Graphs saved as 'feature_importance.png' and 'pca_clustering.png'")
