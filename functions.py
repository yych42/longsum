from typing import List, Tuple
from typing import Tuple, List
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from langchain.callbacks import get_openai_callback
import tiktoken
import pandas as pd
import numpy as np
from hashlib import md5
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader  # requires pypdf
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import concurrent.futures

os.environ["OPENAI_API_KEY"] = "sk-YOU NEED TO FILL ME IN AND MAYBE DONT HARD CODE IT LIEK I DID"


def process_uploaded_file(file_bytes):
    """
    Process the uploaded file bytes and return its content chunks, number of pages and identifier.

    Args:
    file_bytes: uploaded file in bytes.

    Returns:
    text_chunks: segmented text from the uploaded file.
    pages_n: number of pages in the uploaded file.
    identifier: MD5 hash of the uploaded file.
    """
    # Write the file to a temporary file and get the path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        path = tmp_file.name
        tmp_file.write(file_bytes)
        identifier = md5(file_bytes).hexdigest()

    # Load the file into a document
    loader = PyPDFLoader(path)
    document = loader.load_and_split()  # Split as pages
    pages_n = len(document)

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    text_chunks = text_splitter.split_documents(document)

    # Clean it up
    os.remove(path)

    return text_chunks, pages_n, identifier


def summarize_chunks(text_chunks, document_type):
    """
    Summarize the chunks of text using OpenAI's language model.

    Args:
    text_chunks (list): A list of text chunks to be summarized.
    document_type (str): The type of document being summarized.

    Returns:
    chunk_summaries (list): A list of summaries for each text chunk.
    """
    llm = ChatOpenAI(temperature=0)
    map_prompt = """We have a long {type} that we want to refine. It's been broken up into multiple parts. We want a concisely refined paragraph for each part, but do include important details. Avoid "it talked about such and such" without any details.

    You are assigned this part:

    "{text}"

    If there's nothing informative to summarize for the part, just return "[]" and nothing else.
    CONCISELY REFINED PARAGRAPH:"""
    MAP_PROMPT = PromptTemplate(template=map_prompt, input_variables=["text", "type"])
    chain = LLMChain(llm=llm, prompt=MAP_PROMPT)

    chunk_summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for chunk in text_chunks:
            future = executor.submit(
                chain.run, text=chunk.page_content, type=document_type
            )
            futures.append(future)
        chunk_summaries = [future.result() for future in futures]

    return chunk_summaries


def create_df_with_embeddings(chunk_summaries):
    """
    Create a Pandas DataFrame from a list of chunk summaries.

    Args:
        chunk_summaries (list): A list containing the summaries for each chunk.

    Returns:
        A Pandas DataFrame with columns for "id", "embedding", and "text".
    """
    embeddings = OpenAIEmbeddings()

    # Obtain embeddings for all summaries in parallel
    doc_results = []
    doc_results = embeddings.embed_documents(chunk_summaries)

    # Create a DataFrame using the obtained embeddings
    df = pd.DataFrame(
        zip(range(len(chunk_summaries)), doc_results, chunk_summaries),
        columns=["id", "embedding", "text"],
    )

    return df


def embeddings_to_numpy_array(df):
    """
    Convert the "embedding" column of a DataFrame to a numpy array.

    Args:
        df (Pandas DataFrame): The DataFrame containing the embeddings.

    Returns:
        A numpy array with the embeddings.
    """
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    matrix = np.stack(df["embedding"].values)
    return matrix


def suggest_number_of_clusters(
    matrix: np.ndarray, lower_n_clusters: int = 2
) -> Tuple[int, List[Tuple[int, float]]]:
    """
    Suggest the number of clusters based on a matrix of data.

    Args:
        matrix (numpy.ndarray): A matrix of data to cluster.
        lower_n_clusters (int, optional): The lower bound for the number of clusters. Defaults to 2.

    Returns:
        tuple: The number of clusters based on the elbow method and a list of plausible numbers of clusters based on the silhouette score.
    """
    upper_n_clusters = matrix.shape[0]
    k_range = range(lower_n_clusters, upper_n_clusters)

    sse = []  # Sum of squared errors
    sil_scores = []
    calinski_harabasz_scores = []

    for k in k_range:
        kmeans = find_kmeans(matrix, k)
        sil_score = silhouette_score(matrix, kmeans.labels_)
        ch_score = calinski_harabasz_score(matrix, kmeans.labels_)
        calinski_harabasz_scores.append(ch_score)
        sil_scores.append(sil_score)
        sse.append(kmeans.inertia_)

    candidate_ks = set(
        [
            x[0]
            for x in sorted(
                zip(k_range, calinski_harabasz_scores), key=lambda x: x[1], reverse=True
            )[:20]
        ]
    ) & set(
        [
            x[0]
            for x in sorted(zip(k_range, sil_scores), key=lambda x: x[1], reverse=True)[
                :20
            ]
        ]
    )

    # Sort with the highest Calinski Harabasz score first and then the highest Silhouette score.
    candidate_ks_sorted = sorted(
        candidate_ks,
        key=lambda x: (
            calinski_harabasz_scores[x - lower_n_clusters],
            sil_scores[x - lower_n_clusters],
        ),
        reverse=True,
    )

    # If the candidates are less than 20, then fill in the rest with the top 20 Calinski Harabasz scores that are not already in the list.
    if len(candidate_ks_sorted) < 20:
        candidate_ks_sorted += [
            x[0]
            for x in sorted(
                zip(k_range, calinski_harabasz_scores), key=lambda x: x[1], reverse=True
            )[:20]
            if x[0] not in candidate_ks_sorted
        ][: 20 - len(candidate_ks_sorted)]

    first_k = candidate_ks_sorted[0]  # The best k is the first element in the list.
    remaining_ks = candidate_ks_sorted[
        1:
    ]  # The remaining ks are the rest of the elements in the list.

    # Find the top 10 remaining ks based on their literal values.
    top_10_remaining_ks = sorted(remaining_ks)[10:]

    # Move the top 10 remaining ks to the end of the list to penalize bigger numbers of clusters.
    remaining_ks = [
        x for x in remaining_ks if x not in top_10_remaining_ks
    ] + top_10_remaining_ks

    print(f"First k: {first_k}")
    print(f"Remaining ks: {remaining_ks}")
    print(
        f"Silhouette and Calinski Harabasz scores for each remaining k: {[(x, sil_scores[x-lower_n_clusters], calinski_harabasz_scores[x-lower_n_clusters]) for x in remaining_ks]}"
    )

    return first_k, remaining_ks


def find_kmeans(matrix, n_clusters):
    """
    Perform K-means clustering on the given matrix.

    Args:
        matrix (numpy.ndarray): The matrix to cluster.
        n_clusters (int): The number of clusters to create.

    Returns:
        sklearn.cluster.KMeans: The KMeans object trained on the matrix, which can be used to assign cluster labels.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        max_iter=3000,
        random_state=42,
    ).fit(matrix)
    return kmeans


def update_df_with_cluster_labels(kmeans, df):
    """
    Assign cluster labels to a dataframe using a KMeans object.

    Args:
        kmeans (sklearn.cluster.KMeans): The KMeans object to use for clustering.
        df (Pandas DataFrame): The DataFrame to which cluster labels will be assigned.

    Returns:
        Pandas DataFrame: The original DataFrame with an additional column ("Cluster") containing cluster labels.
    """
    labels = kmeans.labels_
    df["Cluster"] = labels
    return df


def estimate_num_tokens(model_name, text):
    """
    Estimate the number of GPT tokens for a given text and model.

    Args:
        model_name (str): The name of the model to use.
        text (str): The text to encode.

    Returns:
        The number of tokens in the encoded text.
    """
    enc = tiktoken.encoding_for_model(model_name)
    num_tokens = len(enc.encode(text))
    return num_tokens


def all_clusters_within_token_limit(clustered_df, model_name, max_num_tokens):
    """
    Check if the number of tokens for each cluster in a given DataFrame is less than or equal to the maximum number of tokens.

    Args:
        clustered_df (Pandas DataFrame): A DataFrame containing cluster information.
        model_name (str): The name of the model to use for encoding.
        max_num_tokens (int): The maximum number of tokens allowed in each cluster.

    Returns:
        bool: True if all clusters have a number of tokens less than or equal to max_num_tokens, False otherwise.
    """
    cluster_num_tokens = []

    for cluster in clustered_df["Cluster"].unique():
        cluster_sum_texts = "\n\n".join(
            clustered_df[clustered_df["Cluster"] == cluster]["text"].tolist()
        )
        cluster_num_tokens.append(estimate_num_tokens(model_name, cluster_sum_texts))

    return all([x <= max_num_tokens for x in cluster_num_tokens])


def solve_clustering(matrix, df, initial_num_splits, model_name, max_num_tokens):
    """
    Final selection for the number clusters to use before summarizing each cluster into a single paragraph.

    Args:
        matrix (numpy.ndarray): A matrix of data to cluster.
        df (Pandas DataFrame): The DataFrame to cluster.
        initial_num_splits (int): The initial number of splits for the clusters.
        model_name (str): The name of the model to use for encoding.
        max_num_tokens (int): The maximum number of tokens allowed in each cluster.

    Returns:
        tuple: A DataFrame of with updated cluster labels based on the ideal number of clusters.
    """
    # The initial number of splits being less than 2 means that the text is short enough to be summarized in a single paragraph.
    if initial_num_splits < 2:
        return update_df_with_cluster_labels(
            find_kmeans(matrix, initial_num_splits), df
        )

    # Initialize the DataFrame to store the clusters that are within the token limit and satisfies the selection process.
    # This DataFrame will be used to store the final selection of clusters and will be returned.
    cluster_solution_df = pd.DataFrame()

    # Find the number of clusters to select from based on the elbow method and a list of plausible numbers of clusters based on the silhouette score.
    first_k, backup_k = suggest_number_of_clusters(matrix, initial_num_splits)
    n_clusters_to_try = [first_k] + backup_k

    max_continuity_score = -1
    best_continuity_n_cluster = -1

    for n_clusters in n_clusters_to_try:
        print("Trying", n_clusters, "clusters")
        candidate_clustered_df = update_df_with_cluster_labels(
            find_kmeans(matrix, n_clusters), df
        )

        if all_clusters_within_token_limit(
            candidate_clustered_df, model_name, max_num_tokens
        ):
            continuity_score = 0
            for cluster in candidate_clustered_df["Cluster"].unique():
                diff_arr = np.diff(
                    sorted(
                        candidate_clustered_df[
                            candidate_clustered_df["Cluster"] == cluster
                        ]["id"]
                    )
                )
                continuity_score += sum(diff_arr == 1)

            print("Continuity score for", n_clusters, "clusters:", continuity_score)

            if continuity_score > max_continuity_score:
                max_continuity_score = continuity_score
                best_continuity_n_cluster = n_clusters
                print(
                    "Found new best continuity number of clusters:",
                    best_continuity_n_cluster,
                )
        else:
            print(n_clusters, "fails the token limit test")

    # Update the cluster solution DataFrame with the new best cluster solution
    cluster_solution_df = update_df_with_cluster_labels(
        find_kmeans(matrix, best_continuity_n_cluster), df
    )
    # Eliminate clusters with no text
    cluster_solution_df = cluster_solution_df[cluster_solution_df["text"] != "[]"]

    # Print the number of clusters with the highest continuity in id
    print(
        "Number of clusters with the highest continuity in id:",
        best_continuity_n_cluster,
    )
    print(
        "Final cluster solution labels:\n"
        + str(cluster_solution_df["Cluster"].unique())
    )

    return cluster_solution_df


def finalize_summary(grouped_summaries, document_type, goal):
    llm = ChatOpenAI(temperature=0)
    combine_prompt = """We have summarized a long {type} into multiple parts.

    Here's a set of the summarized parts of the long {type}:

    {text}

    (The order of the parts is not important)
    
    We want you to combine these parts into a single paragraph that is coherent and nuanced.

    Keep in mind that your paragraph will be part of a {goal}. You can refine, combine, or otherwise edit the original set for clarity and coherency.
    Do not add any new information that is not in the original set.
    
    Concisely Refined {goal} Paragraph:"""
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt, input_variables=["text", "type", "goal"]
    )
    summary_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT)

    summary_paragraphs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for summary in grouped_summaries:
            future = executor.submit(
                summary_chain.run, text=summary, type=document_type, goal=goal
            )
            futures.append(future)
        summary_paragraphs = [future.result() for future in futures]

    return summary_paragraphs
