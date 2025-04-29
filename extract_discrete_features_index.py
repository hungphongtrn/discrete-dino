import os
import numpy as np
from faiss import IndexFlatL2
from tqdm import tqdm
from datasets import load_dataset
from loguru import logger

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

SAVED_PATH = "./centroids.npy"
REPO_ID = "hungphongtrn/vqav2_extracted_features"
FINAL_BATCH_ID = 80  # Process batches 0 to 80


if __name__ == "__main__":
    # Check if the file exists
    if not os.path.exists(SAVED_PATH):
        logger.info(f"File {SAVED_PATH} does not exist.")
    else:
        # Load the centroids
        centroids = np.load(SAVED_PATH)
        logger.info(f"Centroids loaded successfully with shape: {centroids.shape}")

    num_clusters, input_dim = centroids.shape
    # Generate random data since no retraining data is provided
    index = IndexFlatL2(input_dim)
    index.reset()
    index.add(centroids)

    all_batches = {"text": [], "feature_indices": []}

    # Extract the index
    for i in tqdm(range(FINAL_BATCH_ID + 1), desc="Processing Batches"):
        logger.info(f"--- Processing Batch {i} ---")
        ds = None

        try:
            # Step 1a: Load features for the current batch
            logger.info(f"Loading features for batch {i}...")
            subset = f"batch_{i}"
            ds = load_dataset(REPO_ID, subset, split="train", num_proc=64)
            batch_features_list = ds["image_features"]

            if not batch_features_list:
                logger.info(f"Batch {i} is empty or has no features. Skipping.")
                continue

            batch_features = np.array(batch_features_list, dtype=np.float32)
            # Reshape to (total_patches, dim)
            if batch_features.ndim > 2:
                batch_features = batch_features.reshape(-1, batch_features.shape[-1])

            if batch_features.shape[0] == 0:
                logger.info(
                    f"Batch {i} resulted in 0 features after reshape. Skipping."
                )
                continue

            logger.info(f"Batch {i}: Features shape: {batch_features.shape}")

            # Step 1b: Check if enough data points for KMeans
            if batch_features.shape[0] < num_clusters:
                logger.info(
                    f"Batch {i} has only {batch_features.shape[0]} features, which is less than the required {num_clusters} for K-Means. Skipping K-Means for this batch."
                )
                continue

            # Step 2: Perform KMeans clustering
            _, indices = index.search(batch_features, 1)
            indices = indices.ravel()
            assert indices.shape[0] == batch_features.shape[0], (
                "indices and batch_features do not match in size"
            )

            all_batches["text"].extend(ds["text"])
            all_batches["feature_indices"].extend(indices.tolist())

        except Exception as e:
            logger.info(f"Error processing batch {i}: {e}")
