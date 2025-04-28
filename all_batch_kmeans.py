import gc
import numpy as np
import faiss
import tqdm
from datasets import Dataset
from datetime import datetime
from loguru import logger
import os
import glob  # To find the saved centroid files

# --- Configuration ---
CENTROIDS_SAVE_DIR = (
    "./batch_centroids"  # Directory WHERE intermediate centroids WERE saved
)
N_CENTROIDS_FINAL = 8196  # Number of centroids for the final KMeans
N_ITER = 20  # KMeans iterations (can be same or different)
DIM = 768  # Feature dimension (MUST match Script 1)
FINAL_REPO_NAME_TEMPLATE = (
    "hungphongtrn/k-means-final-centroids-{today}"  # HF repo name
)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Saved Centroids ---
    logger.info(f"Loading intermediate centroids from: {CENTROIDS_SAVE_DIR}")
    centroid_files = sorted(
        glob.glob(os.path.join(CENTROIDS_SAVE_DIR, "batch_centroids_*.npy"))
    )  # Use sorted glob

    if not centroid_files:
        logger.error(
            f"No centroid files found in {CENTROIDS_SAVE_DIR}. Did Script 1 run successfully?"
        )
        exit()

    logger.info(f"Found {len(centroid_files)} centroid files.")

    all_centroids_list = []
    for f_path in tqdm(centroid_files, desc="Loading centroid files"):
        try:
            centroids = np.load(f_path)
            # Basic validation (optional but recommended)
            if centroids.ndim == 2 and centroids.shape[1] == DIM:
                all_centroids_list.append(centroids)
            else:
                logger.warning(
                    f"Skipping file {f_path}: Unexpected shape {centroids.shape} or dimension mismatch (Expected dim={DIM})."
                )
        except Exception as e:
            logger.error(f"Error loading centroid file {f_path}: {e}")

    if not all_centroids_list:
        logger.error("No valid centroids were loaded from the files. Exiting.")
        exit()

    # Combine all loaded centroids into one large array
    all_centroids_array = np.vstack(all_centroids_list)
    del all_centroids_list  # Free memory
    gc.collect()  # Explicit garbage collection

    logger.info(f"Combined centroids shape: {all_centroids_array.shape}")

    # Ensure we have enough collected centroids for the final KMeans
    if all_centroids_array.shape[0] < N_CENTROIDS_FINAL:
        logger.warning(
            f"Total loaded centroids ({all_centroids_array.shape[0]}) is less than final desired centroids ({N_CENTROIDS_FINAL}). Adjusting final K-Means 'k' to {all_centroids_array.shape[0]}."
        )
        N_CENTROIDS_FINAL = all_centroids_array.shape[0]  # Adjust k

    if N_CENTROIDS_FINAL == 0:
        logger.error("No valid centroids available to perform final KMeans. Exiting.")
        exit()

    # --- Final KMeans ---
    logger.info(
        f"Training final KMeans on {all_centroids_array.shape[0]} collected centroids to get {N_CENTROIDS_FINAL} final centroids..."
    )
    kmeans_final = faiss.Kmeans(d=DIM, k=N_CENTROIDS_FINAL, niter=N_ITER, verbose=True)
    # Ensure data is float32 for Faiss
    kmeans_final.train(all_centroids_array.astype(np.float32))

    # Cleanup the large combined array
    del all_centroids_array
    gc.collect()

    final_centroids = kmeans_final.centroids

    # Save centroids to file
    with open("centroids.npy", "wb") as f:
        np.save(f, final_centroids)

    if final_centroids is None:
        logger.error("Final KMeans training failed, no centroids generated. Exiting.")
        exit()

    logger.info(f"Final centroids calculated. Shape: {final_centroids.shape}")

    # --- Save and Push Final Centroids ---
    centroids_dataset = Dataset.from_dict(
        {"centroid_features": final_centroids.tolist()}
    )

    today = datetime.now().strftime("%Y%m%d")
    centroids_repo = FINAL_REPO_NAME_TEMPLATE.format(today=today)
    logger.info(
        f"Pushing final centroids dataset to Hugging Face Hub: {centroids_repo}"
    )

    try:
        centroids_dataset.push_to_hub(centroids_repo)
        logger.info("Final centroids pushed successfully.")
    except Exception as e:
        logger.error(
            f"Failed to push centroids to Hugging Face Hub: {e}", exc_info=True
        )

    logger.info("Script 2 finished.")
