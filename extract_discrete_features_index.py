import os
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset, Dataset
from loguru import logger
import multiprocessing
from functools import partial
import time
import gc  # Garbage Collector interface

# --- Configuration ---
SAVED_PATH = "./centroids.npy"
REPO_ID = "hungphongtrn/vqav2_extracted_features"
FINAL_BATCH_ID = 80  # Process batches 0 to 80 (inclusive)
CENTROIDS_REPO = "hungphongtrn/k-means-final-centroids-20250428"
OUTPUT_REPO = "hungphongtrn/vqav2_extracted_features_index"
# Adjust based on your CPU cores and available RAM for loading phase.
# Loading is often I/O bound, so more workers might help up to a point.
NUM_LOAD_WORKERS = max(1, multiprocessing.cpu_count())
# --- End Configuration ---

# Configure logger for tqdm compatibility
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


def load_batch_data(batch_id, repo_id):
    """
    Loads texts and features for a single batch_id.
    Returns a dictionary {'texts': list, 'features': np.ndarray, 'batch_id': int} or None on error/skip.
    """
    subset = f"batch_{batch_id}"
    try:
        # logger.info(f"[Loader {os.getpid()}] Loading Batch {batch_id}...") # Optional: Log worker activity
        # Use num_proc=1 here as parallelism is handled by the Pool
        ds = load_dataset(repo_id, subset, split="train", num_proc=1)

        features_list = ds["image_features"]
        texts = ds["texts"]

        if not features_list or not texts:
            logger.warning(
                f"Batch {batch_id}: Contains no features or texts. Skipping."
            )
            return None

        # Convert features to numpy array immediately
        features_array = np.array(features_list, dtype=np.float32)

        return {"texts": texts, "features": features_array, "batch_id": batch_id}

    except FileNotFoundError:
        logger.warning(
            f"Batch {batch_id}: Data files not found (subset '{subset}' might not exist). Skipping."
        )
        return None
    except ValueError as ve:
        # Catch potential errors during np.array conversion if lists are jagged
        logger.error(
            f"Batch {batch_id}: Error converting features to array (likely inconsistent shapes): {ve}. Skipping."
        )
        return None
    except Exception as e:
        logger.error(
            f"Error loading batch {batch_id}: {e}", exc_info=True
        )  # Log full traceback
        return None


if __name__ == "__main__":
    main_start_time = time.time()

    # --- Phase 1: Load Centroids ---
    centroids_load_start = time.time()
    centroids = None
    if not os.path.exists(SAVED_PATH):
        logger.info(f"File {SAVED_PATH} does not exist. Downloading from Hub...")
        try:
            centroids_ds = load_dataset(CENTROIDS_REPO, split="train")
            centroids = np.array(centroids_ds["centroid_features"])
            np.save(SAVED_PATH, centroids)
            logger.info(
                f"Centroids downloaded and saved to {SAVED_PATH}. Shape: {centroids.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to load or save centroids from {CENTROIDS_REPO}: {e}")
            exit(1)
    else:
        centroids = np.load(SAVED_PATH)
        logger.info(
            f"Centroids loaded successfully from {SAVED_PATH}. Shape: {centroids.shape}"
        )

    if centroids is None or centroids.size == 0:
        logger.error("Centroids are empty. Exiting.")
        exit(1)

    num_clusters, input_dim = centroids.shape
    logger.info(
        f"Centroid loading took {time.time() - centroids_load_start:.2f} seconds."
    )
    gc.collect()  # Hint garbage collector

    # --- Phase 2: Parallel Data Loading ---
    load_start_time = time.time()
    logger.info(f"Starting parallel data loading using {NUM_LOAD_WORKERS} workers...")

    batch_ids_to_process = list(range(FINAL_BATCH_ID + 1))
    loaded_data = []  # Will store results from workers

    # Use 'spawn' for potentially better cross-platform compatibility if needed
    # multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(NUM_LOAD_WORKERS) as pool:
        # Use partial to fix the repo_id argument
        load_func = partial(load_batch_data, repo_id=REPO_ID)

        # Use imap_unordered to process results as they arrive
        results_iterator = pool.imap_unordered(load_func, batch_ids_to_process)

        for result in tqdm(
            results_iterator, total=len(batch_ids_to_process), desc="Loading Batches"
        ):
            if result:  # Only append successful loads
                loaded_data.append(result)

    logger.info(f"Finished loading data for {len(loaded_data)} batches.")
    logger.info(f"Data loading phase took {time.time() - load_start_time:.2f} seconds.")
    gc.collect()

    if not loaded_data:
        logger.error("No data was loaded successfully. Exiting.")
        exit(1)

    # --- Phase 3: Data Aggregation and Preparation ---
    aggregate_start_time = time.time()
    logger.info("Aggregating loaded data...")

    all_texts_repeated = []
    all_features_list = []
    total_patches = 0
    total_texts_items = 0

    # Sort loaded data by batch_id to maintain some order if desired (optional)
    loaded_data.sort(key=lambda x: x["batch_id"])

    for batch_data in tqdm(loaded_data, desc="Aggregating Batches"):
        texts = batch_data["texts"]
        features = batch_data["features"]
        batch_id = batch_data["batch_id"]  # For logging if needed

        if features.size == 0 or len(texts) == 0:
            logger.warning(f"Skipping aggregation for empty batch {batch_id}")
            continue

        original_num_items = len(texts)
        patches_per_item = 1  # Default for 2D features [items, dim]

        # Determine patches_per_item and reshape features to [total_patches, dim]
        if features.ndim == 3:  # Assume [items, patches, dim]
            if features.shape[0] != original_num_items:
                logger.warning(
                    f"Batch {batch_id}: Mismatch between texts count ({original_num_items}) and features first dim ({features.shape[0]}). Trying to proceed assuming last dim is feature dim."
                )
                # Heuristic: if second dim looks like patch count and third like dim
                if features.shape[2] == input_dim:
                    patches_per_item = features.shape[1]
                    # We might lose the connection to original texts if shape[0] is wrong. Be careful.
                else:
                    logger.error(
                        f"Batch {batch_id}: Cannot determine patches/dim for shape {features.shape}. Skipping."
                    )
                    continue
            elif features.shape[0] > 0:  # Standard case
                patches_per_item = features.shape[1]

            current_batch_features = features.reshape(-1, input_dim)

        elif features.ndim == 2:  # Assume [items, dim]
            if features.shape[0] != original_num_items:
                logger.error(
                    f"Batch {batch_id}: Mismatch between texts count ({original_num_items}) and features count ({features.shape[0]}) for 2D array. Skipping."
                )
                continue
            if features.shape[1] != input_dim:
                logger.error(
                    f"Batch {batch_id}: Feature dimension mismatch (Expected {input_dim}, Got {features.shape[1]}). Skipping."
                )
                continue
            current_batch_features = (
                features  # Already in [items, dim] format (patches_per_item=1)
            )
            patches_per_item = 1

        else:
            logger.warning(
                f"Batch {batch_id}: Unexpected features ndim {features.ndim}. Skipping."
            )
            continue

        if current_batch_features.shape[0] == 0:
            logger.info(
                f"Batch {batch_id} resulted in 0 features after reshape. Skipping."
            )
            continue

        # Repeat texts according to patches_per_item
        repeated_texts = [text for text in texts for _ in range(patches_per_item)]

        # Sanity check
        if len(repeated_texts) != current_batch_features.shape[0]:
            logger.error(
                f"Batch {batch_id}: Mismatch after text repetition! Texts: {len(repeated_texts)}, Features: {current_batch_features.shape[0]}. Patches_per_item={patches_per_item}, Original texts={original_num_items}. Skipping batch."
            )
            continue

        all_features_list.append(current_batch_features)
        all_texts_repeated.extend(repeated_texts)
        total_patches += current_batch_features.shape[0]
        total_texts_items += original_num_items

    if not all_features_list:
        logger.error("No features were aggregated. Exiting.")
        exit(1)

    # Concatenate all features into a single large NumPy array
    logger.info("Concatenating features...")
    try:
        final_features_array = np.concatenate(all_features_list, axis=0)
        # Clear the list to free memory
        del all_features_list
        gc.collect()
    except ValueError as e:
        logger.error(
            f"Error during concatenation, likely due to inconsistent feature dimensions: {e}"
        )
        # Optional: Add more detailed logging here to find the problematic batch shape
        exit(1)
    except MemoryError:
        logger.error(
            "Memory Error: Not enough RAM to concatenate all features into a single array."
        )
        logger.error(
            f"Required space approx: {total_patches * input_dim * 4 / (1024**3):.2f} GB"
        )
        exit(1)

    logger.info(
        f"Aggregation complete. Total patches: {final_features_array.shape[0]}, Total texts: {len(all_texts_repeated)}"
    )
    logger.info(
        f"Final features array shape: {final_features_array.shape}, dtype: {final_features_array.dtype}"
    )

    # Final check
    assert final_features_array.shape[0] == len(all_texts_repeated), (
        f"Final mismatch! Features: {final_features_array.shape[0]}, Texts: {len(all_texts_repeated)}"
    )
    assert final_features_array.shape[1] == input_dim, (
        f"Final feature dimension mismatch! Expected {input_dim}, Got {final_features_array.shape[1]}"
    )

    logger.info(
        f"Data aggregation phase took {time.time() - aggregate_start_time:.2f} seconds."
    )
    gc.collect()

    # --- Phase 4: Faiss Search ---
    search_start_time = time.time()
    logger.info("Initializing Faiss index...")
    index = faiss.IndexFlatL2(input_dim)
    index.add(centroids)
    logger.info(f"Faiss index populated with {index.ntotal} centroids.")

    logger.info(
        f"Performing Faiss search for {final_features_array.shape[0]} features..."
    )
    # Faiss expects float32
    if final_features_array.dtype != np.float32:
        logger.warning(
            f"Feature array dtype is {final_features_array.dtype}, converting to float32 for Faiss."
        )
        final_features_array = final_features_array.astype(np.float32)
        gc.collect()

    distances, indices = index.search(
        final_features_array, 1
    )  # Search for the 1 nearest neighbor
    indices = indices.ravel()  # Flatten the indices array [n, 1] -> [n]

    logger.info("Faiss search completed.")
    logger.info(
        f"Faiss search phase took {time.time() - search_start_time:.2f} seconds."
    )
    del final_features_array  # Free memory
    del index
    gc.collect()

    # --- Phase 5: Dataset Creation and Push ---
    dataset_start_time = time.time()
    logger.info("Creating final Hugging Face Dataset...")

    if len(all_texts_repeated) != len(indices):
        logger.error(
            f"FATAL: Mismatch between final texts ({len(all_texts_repeated)}) and indices ({len(indices)}) count!"
        )
        exit(1)

    try:
        final_data_dict = {
            "texts": all_texts_repeated,
            "feature_indices": indices.tolist(),
        }
        # Clear original list
        del all_texts_repeated
        del indices
        gc.collect()

        final_ds = Dataset.from_dict(final_data_dict)
        logger.info("Dataset created successfully.")
        logger.info(f"Dataset features: {final_ds.features}")
        logger.info(f"Number of rows: {len(final_ds)}")

        logger.info(f"Pushing dataset to hub: {OUTPUT_REPO}")
        # Adjust num_shards based on final dataset size if needed
        final_ds.push_to_hub(OUTPUT_REPO, split="train", num_shards=64)
        logger.info("Dataset successfully pushed to hub.")
    except Exception as e:
        logger.error(f"Failed to create or push dataset: {e}", exc_info=True)

    logger.info(
        f"Dataset creation and push phase took {time.time() - dataset_start_time:.2f} seconds."
    )
    logger.info(
        f"Total script execution time: {time.time() - main_start_time:.2f} seconds."
    )
