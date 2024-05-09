from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
from functions import *
import numpy as np

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    end_goal: Optional[str] = Form(None),
):
    if not file:
        return {"error": "Please upload a file."}

    file_bytes = await file.read()

    # Obtaining the file's text content in chunks.
    text_chunks, pages_n, file_identifier = process_uploaded_file(file_bytes)
    if len(text_chunks) == 0:
        return {
            "error": "There was a problem reading the file. Please try a different file."
        }
    if pages_n > 50:
        return {
            "error": "This document is too long. Please try a different file with less than 50 pages.",
            "info": "This is a temporary limitation that will be removed in future updates.",
        }

    chunk_summaries = summarize_chunks(text_chunks, document_type or "text")

    df = create_df_with_embeddings(chunk_summaries)
    matrix = embeddings_to_numpy_array(df)

    _sum_texts = "\n\n".join(df["text"].tolist())
    num_tokens = estimate_num_tokens("gpt-3.5-turbo", _sum_texts)
    max_num_tokens = 2500

    num_splits = int(np.ceil(num_tokens / max_num_tokens))
    cluster_solution = solve_clustering(
        matrix, df, num_splits, "gpt-3.5-turbo", max_num_tokens
    )

    grouped_clusters = []
    for cluster in cluster_solution["Cluster"].unique():
        grouped_clusters.append(
            cluster_solution[cluster_solution["Cluster"] == cluster]["text"].values
        )

    summary_sections = finalize_summary(
        grouped_summaries=grouped_clusters,
        document_type=document_type or "text",
        goal=end_goal or "briefing",
    )

    return {"summary": summary_sections}
