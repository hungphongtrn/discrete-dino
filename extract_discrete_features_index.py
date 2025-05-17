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
import shutil  # For directory cleanup

# --- Configuration ---
SAVED_PATH = "./centroids.npy"
REPO_ID = "hungphongtrn/vqav2_extracted_features"
FINAL_BATCH_ID = 80  # Process batches 0 to 80 (inclusive)
CENTROIDS_REPO = "hungphongtrn/k-means-final-centroids-20250428"
OUTPUT_REPO = "hungphongtrn/vqav2_extracted_features_index_chunked"  # Using a new repo name to avoid overwriting
# Adjust based on your CPU cores. Use more workers for loading within a chunk.
NUM_LOAD_WORKERS = max(1, multiprocessing.cpu_count())
# Number of batches to process simultaneously in one chunk
CHUNK_SIZE = 10
# Temporary directory to save intermediate chunk datasets
TEMP_DATA_DIR = "./temp_indexed_chunks"
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
        ds = load_dataset(
            repo_id, subset, split="train", num_proc=1, streaming=False
        )  # Ensure not streaming

        features_list = ds["image_features"]
        texts = ds["texts"]

        if not features_list or len(texts) != len(features_list):
            if len(texts) == 0 and len(features_list) == 0:
                logger.warning(
                    f"Batch {batch_id}: Contains no features AND no texts. Skipping."
                )
            else:
                logger.warning(
                    f"Batch {batch_id}: Mismatch in count (texts: {len(texts)}, features: {len(features_list)}) or lists are empty. Skipping."
                )
            return None

        # Attempt to convert features to numpy array. Handle potential jaggedness.
        try:
            features_array = np.array(features_list, dtype=np.float32)
        except ValueError as ve:
            logger.error(
                f"Batch {batch_id}: Error converting features to array (likely inconsistent shapes within batch): {ve}. Skipping."
            )
            return None

        return {"texts": texts, "features": features_array, "batch_id": batch_id}

    except FileNotFoundError:
        logger.warning(
            f"Batch {batch_id}: Data files not found (subset '{subset}' might not exist). Skipping."
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
    input_dim = None  # Define input_dim here
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
        try:
            centroids = np.load(SAVED_PATH)
            logger.info(
                f"Centroids loaded successfully from {SAVED_PATH}. Shape: {centroids.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to load centroids from {SAVED_PATH}: {e}")
            exit(1)

    if centroids is None or centroids.size == 0:
        logger.error("Centroids are empty. Exiting.")
        exit(1)

    num_clusters, input_dim = centroids.shape
    logger.info(f"Input feature dimension detected: {input_dim}")
    logger.info(
        f"Centroid loading took {time.time() - centroids_load_start:.2f} seconds."
    )
    gc.collect()  # Hint garbage collector

    # --- Phase 2: Initialize Faiss Index ---
    # Index can be initialized once with centroids as it's read-only
    faiss_init_start = time.time()
    logger.info("Initializing Faiss index with centroids...")
    try:
        index = faiss.IndexFlatL2(input_dim)
        index.add(centroids)  # Add centroids to the index
        logger.info(f"Faiss index populated with {index.ntotal} centroids.")
    except Exception as e:
        logger.error(f"Failed to initialize or populate Faiss index: {e}")
        exit(1)

    # Centroids are no longer needed in main memory after adding to index
    del centroids
    gc.collect()
    logger.info(
        f"Faiss index initialization took {time.time() - faiss_init_start:.2f} seconds."
    )

    # --- Phase 3: Process Data in Chunks ---
    process_chunks_start_time = time.time()
    logger.info(f"Starting data processing in chunks of {CHUNK_SIZE} batches.")

    # Create temporary directory for saving chunk results
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    logger.info(f"Saving intermediate chunk results to {TEMP_DATA_DIR}")

    batch_ids_to_process = list(range(FINAL_BATCH_ID + 1))
    num_batches = len(batch_ids_to_process)
    all_processed_chunk_paths = []  # Keep track of where chunks were saved

    chunk_start_index = 0
    chunk_idx = 0

    while chunk_start_index < num_batches:
        chunk_end_index = min(chunk_start_index + CHUNK_SIZE, num_batches)
        current_chunk_batch_ids = batch_ids_to_process[
            chunk_start_index:chunk_end_index
        ]

        logger.info(
            f"\n--- Processing Chunk {chunk_idx} (Batches {current_chunk_batch_ids[0]} to {current_chunk_batch_ids[-1]}) ---"
        )
        chunk_process_start_time = time.time()

        # --- Load Data for this Chunk (Parallel) ---
        chunk_load_start_time = time.time()
        loaded_chunk_data = []
        logger.info(
            f"Loading batches for chunk {chunk_idx} using {NUM_LOAD_WORKERS} workers..."
        )
        with multiprocessing.Pool(NUM_LOAD_WORKERS) as pool:
            load_func = partial(load_batch_data, repo_id=REPO_ID)
            results_iterator = pool.imap_unordered(load_func, current_chunk_batch_ids)
            for result in tqdm(
                results_iterator,
                total=len(current_chunk_batch_ids),
                desc=f"Loading Chunk {chunk_idx} Batches",
            ):
                if result:
                    loaded_chunk_data.append(result)

        logger.info(
            f"Finished loading data for {len(loaded_chunk_data)} batches in chunk {chunk_idx}."
        )
        logger.info(
            f"Chunk loading took {time.time() - chunk_load_start_time:.2f} seconds."
        )
        # Aggressive collection after pool finishes, before aggregation
        gc.collect()

        if not loaded_chunk_data:
            logger.warning(
                f"Chunk {chunk_idx} contained no successfully loaded data. Skipping processing for this chunk."
            )
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue  # Move to the next chunk

        # --- Aggregate Data for this Chunk ---
        chunk_aggregate_start_time = time.time()
        logger.info(f"Aggregating data for chunk {chunk_idx}...")
        all_texts_repeated_chunk = []
        all_features_list_chunk = []
        total_patches_chunk = 0
        total_texts_items_chunk = 0  # Original count per text item

        # Sort loaded data by batch_id for consistency, though not strictly necessary for aggregation order
        loaded_chunk_data.sort(key=lambda x: x["batch_id"])

        for batch_data in tqdm(
            loaded_chunk_data, desc=f"Aggregating Chunk {chunk_idx}"
        ):
            texts = batch_data["texts"]
            features = batch_data["features"]
            # batch_id = batch_data["batch_id"] # Use for logging if needed

            if features.size == 0 or len(texts) == 0 or features.shape[0] != len(texts):
                logger.warning(
                    f"Skipping aggregation for invalid data in loaded batch {batch_data['batch_id']}."
                )
                continue

            original_num_items = len(texts)
            current_batch_features = None
            patches_per_item = 1  # Default for 2D features

            # Determine patches_per_item and reshape features to [total_patches, dim]
            if features.ndim == 3:  # Assume [items, patches, dim]
                if features.shape[0] != original_num_items:
                    logger.error(
                        f"Batch {batch_data['batch_id']}: Mismatch between texts count ({original_num_items}) and features first dim ({features.shape[0]}) in 3D array. Skipping."
                    )
                    continue
                if features.shape[2] != input_dim:
                    logger.error(
                        f"Batch {batch_data['batch_id']}: Feature dimension mismatch for 3D array (Expected {input_dim}, Got {features.shape[2]}). Skipping."
                    )
                    continue

                patches_per_item = features.shape[1]
                current_batch_features = features.reshape(-1, input_dim)

            elif features.ndim == 2:  # Assume [items, dim]
                if features.shape[0] != original_num_items:
                    logger.error(
                        f"Batch {batch_data['batch_id']}: Mismatch between texts count ({original_num_items}) and features count ({features.shape[0]}) for 2D array. Skipping."
                    )
                    continue
                if features.shape[1] != input_dim:
                    logger.error(
                        f"Batch {batch_data['batch_id']}: Feature dimension mismatch for 2D array (Expected {input_dim}, Got {features.shape[1]}). Skipping."
                    )
                    continue
                current_batch_features = (
                    features  # Already in [items, dim] format (patches_per_item=1)
                )
                patches_per_item = 1

            else:
                logger.warning(
                    f"Batch {batch_data['batch_id']}: Unexpected features ndim {features.ndim}. Skipping."
                )
                continue

            if current_batch_features is None or current_batch_features.shape[0] == 0:
                logger.info(
                    f"Batch {batch_data['batch_id']} resulted in 0 valid features after processing. Skipping."
                )
                continue

            # Repeat texts according to patches_per_item
            repeated_texts = [text for text in texts for _ in range(patches_per_item)]

            # Sanity check after repeating
            if len(repeated_texts) != current_batch_features.shape[0]:
                logger.error(
                    f"Batch {batch_data['batch_id']}: Mismatch after text repetition! Texts: {len(repeated_texts)}, Features: {current_batch_features.shape[0]}. Patches_per_item={patches_per_item}, Original texts={original_num_items}. Skipping batch."
                )
                continue

            all_features_list_chunk.append(current_batch_features)
            all_texts_repeated_chunk.extend(repeated_texts)
            total_patches_chunk += current_batch_features.shape[0]
            total_texts_items_chunk += original_num_items  # Count original text items

        # Clear loaded_chunk_data list as contents are now in aggregated lists/arrays
        del loaded_chunk_data
        gc.collect()

        if not all_features_list_chunk:
            logger.warning(
                f"Chunk {chunk_idx} aggregation resulted in 0 features. Skipping processing for this chunk."
            )
            # Cleanup aggregated intermediate lists even if empty
            del all_features_list_chunk
            del all_texts_repeated_chunk
            gc.collect()
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue  # Move to the next chunk

        # Concatenate all features for the current chunk
        logger.info(f"Concatenating features for chunk {chunk_idx}...")
        try:
            final_features_array_chunk = np.concatenate(all_features_list_chunk, axis=0)
            # Clear the list to free memory immediately after concatenation
            del all_features_list_chunk
            gc.collect()
        except ValueError as e:
            logger.error(
                f"Error during concatenation for chunk {chunk_idx}, likely due to inconsistent feature dimensions: {e}. Skipping chunk.",
                exc_info=True,
            )
            # Cleanup
            del all_texts_repeated_chunk
            del all_features_list_chunk  # Ensure this is gone
            gc.collect()
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue  # Move to the next chunk
        except MemoryError:
            logger.error(
                f"Memory Error: Not enough RAM to concatenate features for chunk {chunk_idx}. This chunk might be too large or CHUNK_SIZE needs reducing."
            )
            logger.error(
                f"Attempted concatenation size approx: {total_patches_chunk * input_dim * 4 / (1024**3):.2f} GB"
            )
            # Cleanup
            del all_features_list_chunk
            del all_texts_repeated_chunk
            gc.collect()
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue  # Move to the next chunk

        logger.info(
            f"Chunk {chunk_idx} aggregation complete. Total patches: {final_features_array_chunk.shape[0]}."
        )
        logger.info(
            f"Chunk aggregation took {time.time() - chunk_aggregate_start_time:.2f} seconds."
        )
        gc.collect()  # Collect after aggregation

        # Final check for chunk data
        if final_features_array_chunk.shape[0] != len(all_texts_repeated_chunk):
            logger.error(
                f"FATAL: Mismatch between chunk aggregated features ({final_features_array_chunk.shape[0]}) and texts ({len(all_texts_repeated_chunk)}) for chunk {chunk_idx}! Skipping processing for this chunk."
            )
            # Cleanup
            del all_texts_repeated_chunk
            del final_features_array_chunk
            gc.collect()
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue  # Move to the next chunk

        # --- Faiss Search for this Chunk ---
        chunk_search_start_time = time.time()
        logger.info(
            f"Performing Faiss search for chunk {chunk_idx} ({final_features_array_chunk.shape[0]} features)..."
        )

        # Faiss expects float32
        if final_features_array_chunk.dtype != np.float32:
            logger.warning(
                f"Chunk {chunk_idx}: Feature array dtype is {final_features_array_chunk.dtype}, converting to float32 for Faiss."
            )
            final_features_array_chunk = final_features_array_chunk.astype(np.float32)
            gc.collect()  # Collect after type conversion if it created a copy

        try:
            # Use the pre-initialized global 'index'
            distances_chunk, indices_chunk = index.search(
                final_features_array_chunk, 1
            )  # Search for the 1 nearest neighbor
            indices_chunk = (
                indices_chunk.ravel()
            )  # Flatten the indices array [n, 1] -> [n]
            logger.info(f"Faiss search for chunk {chunk_idx} completed.")
        except Exception as e:
            logger.error(
                f"Error during Faiss search for chunk {chunk_idx}: {e}. Skipping chunk.",
                exc_info=True,
            )
            # Cleanup
            del all_texts_repeated_chunk
            del final_features_array_chunk
            gc.collect()
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue  # Move to the next chunk

        logger.info(
            f"Chunk search took {time.time() - chunk_search_start_time:.2f} seconds."
        )
        # Clear features array immediately after search
        del final_features_array_chunk
        del distances_chunk  # We don't need distances
        gc.collect()

        # --- Create and Save Chunk Dataset ---
        chunk_dataset_start_time = time.time()
        logger.info(f"Creating and saving dataset for chunk {chunk_idx}...")

        if len(all_texts_repeated_chunk) != len(indices_chunk):
            logger.error(
                f"FATAL: Mismatch between chunk texts ({len(all_texts_repeated_chunk)}) and indices ({len(indices_chunk)}) count after search for chunk {chunk_idx}! Skipping save."
            )
            # Cleanup
            del all_texts_repeated_chunk
            del indices_chunk
            gc.collect()
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue  # Move to the next chunk

        chunk_data_dict = {
            "texts": all_texts_repeated_chunk,
            "feature_indices": indices_chunk.tolist(),  # Convert numpy array to list for Dataset.from_dict
        }

        try:
            # Create dataset for the current chunk
            chunk_ds = Dataset.from_dict(chunk_data_dict)
            chunk_save_path = os.path.join(TEMP_DATA_DIR, f"chunk_{chunk_idx}")
            # Push to hub in case running of disk space
            chunk_ds.push_to_hub(OUTPUT_REPO, f"chunk_{chunk_idx}", split="train")
            # Save the chunk dataset to disk. This creates a directory structure.
            chunk_ds.save_to_disk(chunk_save_path)
            all_processed_chunk_paths.append(chunk_save_path)  # Record the path
            logger.info(
                f"Chunk {chunk_idx} dataset saved to {chunk_save_path}. Rows: {len(chunk_ds)}"
            )
        except Exception as e:
            logger.error(
                f"Failed to create or save chunk {chunk_idx} dataset to disk: {e}",
                exc_info=True,
            )
            # Cleanup
            del chunk_data_dict
            if "chunk_ds" in locals():
                del chunk_ds
            # Don't skip chunk index increment even on save failure, just log error
        finally:
            # --- Clean up memory for the current chunk ---
            del all_texts_repeated_chunk  # The list
            del indices_chunk  # The numpy array
            del chunk_data_dict  # The dictionary
            # No need to delete chunk_ds if save failed, it might not exist, handle in except block
            gc.collect()
            logger.info(f"Memory cleanup complete for chunk {chunk_idx}.")
            logger.info(
                f"Chunk {chunk_idx} processing took {time.time() - chunk_process_start_time:.2f} seconds."
            )

        # Move to the next chunk
        chunk_start_index = chunk_end_index
        chunk_idx += 1

    logger.info(
        f"\n--- All chunk processing completed in {time.time() - process_chunks_start_time:.2f} seconds. ---"
    )
    gc.collect()

    # --- Phase 4: Combine Chunk Datasets from Disk ---
    combine_start_time = time.time()
    logger.info(f"Loading combined dataset from {TEMP_DATA_DIR}...")

    if not os.listdir(TEMP_DATA_DIR):
        logger.error(f"No chunk data found in {TEMP_DATA_DIR}. Exiting.")
        exit(1)

    try:
        # load_dataset from a local directory created by save_to_disk automatically
        # discovers and combines the shards across the saved chunk directories.
        # Specify 'arrow' builder for local files.
        # Need to specify data_files pattern to find arrow files within subdirectories
        # Example path: ./temp_indexed_chunks/chunk_0/data/train-00000-of-NNNNN.arrow
        arrow_files_pattern = os.path.join(TEMP_DATA_DIR, "**", "data-*.arrow")
        logger.info(f"Searching for arrow files with pattern: {arrow_files_pattern}")

        final_combined_ds = load_dataset(
            "arrow", data_files=arrow_files_pattern, split="train"
        )

        logger.info("Combined dataset loaded successfully.")
        logger.info(f"Combined dataset features: {final_combined_ds.features}")
        logger.info(f"Combined number of rows: {len(final_combined_ds)}")
    except Exception as e:
        logger.error(
            f"Failed to load combined dataset from disk ({TEMP_DATA_DIR}): {e}",
            exc_info=True,
        )
        exit(1)

    logger.info(f"Combine phase took {time.time() - combine_start_time:.2f} seconds.")
    gc.collect()

    # --- Phase 5: Push Combined Dataset to Hub ---
    push_start_time = time.time()
    logger.info(f"Pushing combined dataset to hub: {OUTPUT_REPO}")

    try:
        # Decide on num_shards for the final push - based on total size
        # A common heuristic is ~1GB per shard, but let's pick a reasonable fixed number first
        # For a dataset of this potential size, 128 or 256 shards is reasonable.
        # We'll just use 128 as a default.
        recommended_shards = 128
        logger.info(f"Pushing with num_shards={recommended_shards}")
        final_combined_ds.push_to_hub(
            OUTPUT_REPO, "full", split="train", num_shards=recommended_shards
        )
        logger.info("Combined dataset successfully pushed to hub.")
    except Exception as e:
        logger.error(f"Failed to push combined dataset to hub: {e}", exc_info=True)

    logger.info(f"Push phase took {time.time() - push_start_time:.2f} seconds.")
    logger.info(
        f"Total script execution time: {time.time() - main_start_time:.2f} seconds."
    )

    # --- Optional: Cleanup Temporary Directory ---
    # Uncomment the following lines if you want to remove the temporary files after a successful run
    # logger.info(f"Cleaning up temporary directory: {TEMP_DATA_DIR}")
    # try:
    #     shutil.rmtree(TEMP_DATA_DIR)
    #     logger.info("Temporary directory removed.")
    # except Exception as e:
    #     logger.error(f"Failed to remove temporary directory {TEMP_DATA_DIR}: {e}")
