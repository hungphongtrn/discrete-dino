import gc
import numpy as np
import faiss
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
import os

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

REPO_ID = "hungphongtrn/vqav2_extracted_features"
FINAL_BATCH_ID = 80  # Process batches 0 to 80
N_CENTROIDS_PER_BATCH = 8196  # Target number of centroids for each batch's KMeans
N_ITER = 20  # KMeans iterations
DIM = 768  # Feature dimension
CENTROIDS_SAVE_DIR = "./batch_centroids"  # Directory to save intermediate centroids

if __name__ == "__main__":
    # Create the directory for saving centroids if it doesn't exist
    os.makedirs(CENTROIDS_SAVE_DIR, exist_ok=True)
    logger.info(f"Intermediate centroids will be saved to: {CENTROIDS_SAVE_DIR}")

    processed_batches = 0
    skipped_batches = 0

    logger.info("Starting per-batch KMeans processing...")
    for i in tqdm(range(FINAL_BATCH_ID + 1), desc="Processing Batches"):
        logger.info(f"--- Processing Batch {i} ---")
        batch_features = None
        kmeans_batch = None
        ds = None

        try:
            # Step 1a: Load features for the current batch
            logger.info(f"Loading features for batch {i}...")
            subset = f"batch_{i}"
            ds = load_dataset(REPO_ID, subset, split="train")
            batch_features_list = ds["image_features"]

            if not batch_features_list:
                logger.warning(f"Batch {i} is empty or has no features. Skipping.")
                skipped_batches += 1
                continue

            batch_features = np.array(batch_features_list, dtype=np.float32)
            # Reshape to (total_patches, dim)
            if batch_features.ndim > 2:
                batch_features = batch_features.reshape(-1, batch_features.shape[-1])

            if batch_features.shape[0] == 0:
                logger.warning(
                    f"Batch {i} resulted in 0 features after reshape. Skipping."
                )
                skipped_batches += 1
                continue

            logger.info(f"Batch {i}: Features shape: {batch_features.shape}")

            # Step 1b: Check if enough data points for KMeans
            if batch_features.shape[0] < N_CENTROIDS_PER_BATCH:
                logger.warning(
                    f"Batch {i} has only {batch_features.shape[0]} features, which is less than the required {N_CENTROIDS_PER_BATCH} for K-Means. Skipping K-Means for this batch."
                )
                skipped_batches += 1
                continue  # Skip KMeans for this batch

            # Step 1c: Train KMeans on the current batch
            logger.info(
                f"Training KMeans for batch {i} with {N_CENTROIDS_PER_BATCH} centroids..."
            )
            kmeans_batch = faiss.Kmeans(
                d=DIM, k=N_CENTROIDS_PER_BATCH, niter=N_ITER, verbose=True
            )
            kmeans_batch.train(batch_features)

            # Step 1d: Save centroids to disk
            if (
                kmeans_batch.centroids is not None
                and kmeans_batch.centroids.shape[0] == N_CENTROIDS_PER_BATCH
            ):
                save_path = os.path.join(
                    CENTROIDS_SAVE_DIR, f"batch_centroids_{i:04d}.npy"
                )  # Pad index for sorting
                np.save(save_path, kmeans_batch.centroids)
                logger.info(
                    f"Saved {kmeans_batch.centroids.shape[0]} centroids for batch {i} to {save_path}"
                )
                processed_batches += 1
            else:
                logger.warning(
                    f"KMeans training might have failed or produced unexpected centroid shape for batch {i}. Centroids not saved."
                )
                skipped_batches += 1

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}", exc_info=True)
            skipped_batches += 1
            # Decide if you want to stop on error or continue
