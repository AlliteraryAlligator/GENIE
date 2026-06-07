import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
import argparse
from src.utils import parse_chat_completion
from src.utils import get_embedding
from src.utils import save_df
from src.utils import check_overwrite_safety

ALGORITHMS=[
        "k-means",
        "agglomerative",
        "hdbscan"
]

MODES =[
        "feature",
        "description"
]

class Feature(BaseModel):
    description: str
    feature: str

class FeatureList(BaseModel):
    features: list[Feature]


def sample_features(task, model, num_iter):
    # Sample features that represent a given task (e.g. creative writing)

    prompt = f"""
        Identify the core high-level features that define and structure work in {task}. 
        Features should be independent, top-level dimensions — not sub-components of each other. 
        Avoid granular or stylistic details that naturally nest under a broader feature; instead, group them under their parent concept. 
        The goal is a concise, non-redundant set of features that together cover the essential dimensions of {task} without overlap.
        
        Output only the list of features with a one-sentence description of each.
        Follow this format:

        Description: <a short description of what is included in the feature>
        Feature: <the name of the feature>
    """
    feature_iters = []
    for i in tqdm(range(num_iter), total=num_iter):
        response = parse_chat_completion(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=FeatureList,
        )
        features = response.features

        for feature in features:
            feature_iters.append({
                "Feature": feature.feature,
                "Description": feature.description,
                "Iter": i
            })
    feature_iter_df = pd.DataFrame(feature_iters)
    return feature_iter_df

def embed_features(features, model):
    # for clustering based on feature names
    # preprocess the features
    phrases = []
    for phrase in features['Feature'].unique():
        if " and " in phrase:
            phrases.extend(phrase.split(" and "))
        elif " & " in phrase:
            phrases.extend(phrase.split(" & "))
        elif " or " in phrase:
            phrases.extend(phrase.split(" or "))
        elif "/" in phrase:
            phrases.extend(phrase.split("/"))
        else:
            phrases.append(phrase)
            
    emb_data = []
    for phrase in tqdm(phrases, total=len(phrases)):
        emb = get_embedding(text=phrase, model=model)
        emb_data.append({
            "Feature": phrase, 
            "Emb": emb
        })

    return pd.DataFrame(emb_data)

def embed_descriptions(features, model):
    # For clustering based on descriptions
    emb_df = features.copy()
    for index, row in tqdm(features.iterrows(), total=len(features)):
        emb = get_embedding(text=row['Description'], model=model)
        emb_df.loc['Emb'] = emb
    return emb_df


def cluster(emb_df, method="k-means", num_clusters=0):
    embeddings = emb_df['Emb'].tolist()
    phrases = emb_df['Feature'].tolist()

    def cluster_agglomerative():
    
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import Normalizer

        norm_embeddings = Normalizer().fit_transform(embeddings)
        agg_cluster = AgglomerativeClustering(n_clusters=7, linkage='ward')
        cluster_labels = agg_cluster.fit_predict(norm_embeddings)

        clusters = []
        for phrase, label in zip(phrases, cluster_labels):
            clusters.append({
                "Feature": phrase,
                "Label": label
            })

        return pd.DataFrame(clusters)
    
    def cluster_k_means():
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        pca = PCA(n_components=0.95) 
        reduced_embeddings = pca.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=num_clusters, random_state=39, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced_embeddings)

        clusters = []
        for phrase, label in zip(phrases, cluster_labels):
            clusters.append({
                "Feature": phrase,
                "Label": label
            })

        return pd.DataFrame(clusters)

    def cluster_hdbscan():
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3, 
            metric='euclidean', 
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
        clusters = []
        for phrase, label in zip(phrases, cluster_labels):
            clusters.append({
                "Feature": phrase,
                "Label": label
            })

        return pd.DataFrame(clusters)

    if method=='agglomerative':
        clusters = cluster_agglomerative()
    elif method=='k-means':
        clusters = cluster_k_means()
    elif method=='hdbscan':
        clusters = cluster_hdbscan()

    return clusters


def main(args):

    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(args.out_file):
        print("Exiting without discovering features.")
        return

    features = sample_features(args.task, args.model, args.num_samples)
    
    if args.mode=='feature':
        emb_df = embed_features(features, args.emb_model)
    else:
        emb_df = embed_descriptions(features, args.emb_model)

    if args.algorithm=='k-means':
        k = int(round(len(features)/args.num_samples))
        clusters = cluster(emb_df, num_clusters=k)
    else:
        clusters = cluster(emb_df, method=args.algorithm)
    
    save_df(clusters, args.out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Feature Discovery")
    parser.add_argument("--model", default='gpt-4.1-mini', help="Model string for sampling features")
    parser.add_argument("--emb_model", default='text-embedding-3-small', help="Model string for sampling features")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of sampling iterations")
    parser.add_argument("--algorithm", default='k-means', help="Clustering algorithm", choices=ALGORITHMS)
    parser.add_argument("--task", type=str, default='creative writing', help="Task for which features are being discovered")
    parser.add_argument("--out_file", required=True,  help="Path to the output .json file")
    parser.add_argument("--mode", default='feature', help="Clustering based on either the features or their descriptions", choices=MODES)
    
    args = parser.parse_args()
    main(args)