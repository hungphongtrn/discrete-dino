import gc
import numpy as np
import faiss
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
import os
import time  # Added for timing

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

REPO_ID = "hungphongtrn/vqav2_extracted_features"
FINAL_BATCH_ID = 80  # Process batches 0 to 80
N_CENTROIDS_PER_BATCH = 8196  # Target number of centroids for each *group's* KMeans
N_ITER = 20  # KMeans iterations
DIM = 768  # Feature dimension
CENTROIDS_SAVE_DIR = (
    "./batch_centroids"  # Directory to save intermediate centroids
)
NUM_BATCHES_PER_PROCESS = 40  # Number of batches to process together in one KMeans run
START_BATCH_ID = 15
NUM_CORES = os.cpu_count()  # Number of CPU cores available

if __name__ == "__main__":
    # Create the directory for saving centroids if it doesn't exist
    os.makedirs(CENTROIDS_SAVE_DIR, exist_ok=True)
    logger.info(
        f"Intermediate grouped centroids will be saved to: {CENTROIDS_SAVE_DIR}"
    )
    logger.info(f"Processing {NUM_BATCHES_PER_PROCESS} batches per KMeans run.")

    processed_groups = 0
    skipped_groups = 0
    total_individual_batches_processed = 0
    total_individual_batches_skipped = 0

    logger.info("Starting grouped per-batch KMeans processing...")

    # Iterate through batches in steps of NUM_BATCHES_PER_PROCESS
    for start_batch_idx in tqdm(
        range(START_BATCH_ID, FINAL_BATCH_ID + 1, NUM_BATCHES_PER_PROCESS),
        desc="Processing Batch Groups",
    ):
        group_start_time = time.time()
        end_batch_idx = min(
            start_batch_idx + NUM_BATCHES_PER_PROCESS - 1, FINAL_BATCH_ID
        )
        current_batch_indices = range(start_batch_idx, end_batch_idx + 1)

        logger.info(f"--- Processing Batch Group {start_batch_idx}-{end_batch_idx} ---")

        all_group_features_list = []
        group_successful_batches = 0
        group_skipped_batches = 0

        # --- Step 1: Load features for all batches in the current group ---
        logger.info(
            f"Loading features for batches {start_batch_idx} to {end_batch_idx}..."
        )
        for i in current_batch_indices:
            batch_features = None
            ds = None
            try:
                subset = f"batch_{i}"
                # Use streaming=True if memory becomes an issue, but requires different handling
                ds = load_dataset(
                    REPO_ID, subset, split="train", trust_remote_code=True, num_proc=NUM_CORES
                )  # Added trust_remote_code
                batch_features_list = ds["image_features"]

                if not batch_features_list:
                    logger.warning(
                        f"Batch {i} is empty or has no features. Skipping this batch."
                    )
                    total_individual_batches_skipped += 1
                    group_skipped_batches += 1
                    continue

                batch_features = np.array(batch_features_list, dtype=np.float32)
                # Reshape to (total_patches, dim)
                if batch_features.ndim > 2:
                    batch_features = batch_features.reshape(
                        -1, batch_features.shape[-1]
                    )

                if batch_features.shape[0] == 0:
                    logger.warning(
                        f"Batch {i} resulted in 0 features after reshape. Skipping this batch."
                    )
                    total_individual_batches_skipped += 1
                    group_skipped_batches += 1
                    continue

                logger.debug(
                    f"Batch {i}: Loaded features shape: {batch_features.shape}"
                )
                all_group_features_list.append(batch_features)
                group_successful_batches += 1
                total_individual_batches_processed += 1

            except Exception as e:
                logger.error(
                    f"Error loading or processing batch {i}: {e}", exc_info=False
                )  # exc_info=False for less verbose logs during runs
                total_individual_batches_skipped += 1
                group_skipped_batches += 1
            finally:
                # Clean up dataset object and intermediate array
                del ds
                del batch_features
                gc.collect()

        # --- Step 2: Aggregate features and run KMeans if data exists ---
        if not all_group_features_list:
            logger.warning(
                f"No features loaded for group {start_batch_idx}-{end_batch_idx}. Skipping KMeans for this group."
            )
            skipped_groups += 1
            continue  # Move to the next group

        logger.info(
            f"Aggregating features for group {start_batch_idx}-{end_batch_idx}..."
        )
        try:
            group_features = np.concatenate(all_group_features_list, axis=0)
            # Clear the list to free memory
            del all_group_features_list
            gc.collect()
            logger.info(
                f"Group {start_batch_idx}-{end_batch_idx}: Aggregated features shape: {group_features.shape}"
            )

            # Step 2b: Check if enough data points for KMeans
            if group_features.shape[0] < N_CENTROIDS_PER_BATCH:
                logger.warning(
                    f"Group {start_batch_idx}-{end_batch_idx} has only {group_features.shape[0]} combined features, "
                    f"which is less than the required {N_CENTROIDS_PER_BATCH} for K-Means. Skipping K-Means for this group."
                )
                skipped_groups += 1
                del group_features  # Clean up aggregated features
                gc.collect()
                continue  # Skip KMeans for this group

            # Step 2c: Train KMeans on the aggregated features
            logger.info(
                f"Training KMeans for group {start_batch_idx}-{end_batch_idx} with {N_CENTROIDS_PER_BATCH} centroids..."
            )
            kmeans_group = None
            try:
                kmeans_group = faiss.Kmeans(
                    d=DIM,
                    k=N_CENTROIDS_PER_BATCH,
                    niter=N_ITER,
                    verbose=True,
                    gpu=False,  # Set gpu=True if you have compatible GPUs and FAISS-GPU installed
                )
                kmeans_group.train(group_features)
            except Exception as kmeans_exc:
                logger.error(
                    f"Error during KMeans training for group {start_batch_idx}-{end_batch_idx}: {kmeans_exc}",
                    exc_info=True,
                )
                skipped_groups += 1
                del group_features
                if kmeans_group:
                    del kmeans_group
                gc.collect()
                continue  # Skip saving if training failed

            # Step 2d: Save centroids to disk
            if (
                kmeans_group is not None
                and kmeans_group.centroids is not None
                and kmeans_group.centroids.shape[0] == N_CENTROIDS_PER_BATCH
            ):
                # Pad indices for consistent sorting
                save_path = os.path.join(
                    CENTROIDS_SAVE_DIR,
                    f"batch_centroids_{start_batch_idx:04d}-{end_batch_idx:04d}.npy",
                )
                np.save(save_path, kmeans_group.centroids)
                logger.info(
                    f"Saved {kmeans_group.centroids.shape[0]} centroids for group {start_batch_idx}-{end_batch_idx} to {save_path}"
                )
                processed_groups += 1
            else:
                logger.warning(
                    f"KMeans training might have failed or produced unexpected centroid shape for group {start_batch_idx}-{end_batch_idx}. Centroids not saved."
                )
                skipped_groups += 1

            # Clean up group data and kmeans object
            del group_features
            del kmeans_group
            gc.collect()

        except Exception as e:
            logger.error(
                f"Error processing group {start_batch_idx}-{end_batch_idx} after loading: {e}",
                exc_info=True,
            )
            skipped_groups += 1
            # Ensure cleanup even if aggregation or other steps fail
            if "group_features" in locals() and group_features is not None:
                del group_features
            if (
                "all_group_features_list" in locals()
                and all_group_features_list is not None
            ):
                del all_group_features_list
            if "kmeans_group" in locals() and kmeans_group is not None:
                del kmeans_group
            gc.collect()
            # Decide if you want to stop on error or continue (currently continues)

        group_end_time = time.time()
        logger.info(
            f"Finished processing group {start_batch_idx}-{end_batch_idx}. Time taken: {group_end_time - group_start_time:.2f} seconds."
        )
        logger.info(
            f"Group Summary: {group_successful_batches} batches successfully processed, {group_skipped_batches} batches skipped within this group."
        )
        print("-" * 20)  # Add separator for readability

    logger.info("=" * 30)
    logger.info("KMeans Group Processing Finished.")
    logger.info(
        f"Successfully processed and saved centroids for {processed_groups} groups."
    )
    logger.info(
        f"Skipped KMeans for {skipped_groups} groups (due to errors, insufficient data, or no data)."
    )
    logger.info(
        f"Total individual batches contributing data: {total_individual_batches_processed}"
    )
    logger.info(
        f"Total individual batches skipped (empty/error): {total_individual_batches_skipped}"
    )
    logger.info("=" * 30)
