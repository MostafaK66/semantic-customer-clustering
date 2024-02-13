from run_llm_clustering import llm_kmeans
from run_k_means import k_means


def all_clustering_algorithms():
    llm_kmeans()
    k_means()


if __name__ == "__main__":
    all_clustering_algorithms()
