import os
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset, Dataset, Sequence, Value, load_from_disk
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
CHUNK_SIZE = 1 # Set to 1 for debugging
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
        # Keep original texts
        all_original_texts_chunk = []
        # Keep a list of the reshaped index arrays (one array [items, patches] per batch)
        all_reshaped_indices_chunk = []
        # Keep track of patches per item if it might vary (safer)
        # Or just store the start/end index of each batch's indices in the flattened list
        batch_patch_counts = []  # List of (original_num_items, patches_per_item) for each batch

        # Sort loaded data by batch_id
        loaded_chunk_data.sort(key=lambda x: x["batch_id"])

        aggregated_features_list_chunk = []  # List to hold flattened features [batch_total_patches, dim]

        for batch_data in tqdm(
            loaded_chunk_data,
            desc=f"Aggregating Data & Preparing Search for Chunk {chunk_idx}",
        ):
            texts = batch_data["texts"]  # Original texts
            features = batch_data["features"]  # Original features [items, patches, dim]
            batch_id = batch_data["batch_id"]

            if features.ndim != 3 or features.shape[2] != input_dim:
                logger.warning(
                    f"Batch {batch_id}: Expected 3D features [items, patches, {input_dim}], got shape {features.shape}. Skipping."
                )
                continue
            if features.shape[0] != len(texts):
                logger.warning(
                    f"Batch {batch_id}: Mismatch between texts ({len(texts)}) and features items ({features.shape[0]}). Skipping."
                )
                continue

            original_num_items = features.shape[0]
            patches_per_item = features.shape[1]
            batch_total_patches = original_num_items * patches_per_item

            if batch_total_patches == 0:
                logger.info(f"Batch {batch_id} has no patches. Skipping.")
                continue

            # Reshape features for search: [items, patches, dim] -> [total_patches, dim]
            features_for_search = features.reshape(-1, input_dim)

            # Add to the list for chunk concatenation
            aggregated_features_list_chunk.append(features_for_search)

            # Store original texts and metadata for reshaping indices later
            all_original_texts_chunk.extend(texts)
            batch_patch_counts.append((original_num_items, patches_per_item))

        # Clear loaded_chunk_data list
        del loaded_chunk_data
        gc.collect()

        # Concatenate features for the entire chunk
        if not aggregated_features_list_chunk:
            logger.warning(
                f"Chunk {chunk_idx} resulted in no valid features after processing batches. Skipping."
            )
            del all_original_texts_chunk
            del batch_patch_counts
            del aggregated_features_list_chunk
            gc.collect()
            # Skip to next chunk... (need continue logic)
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue

        logger.info(f"Concatenating features for chunk {chunk_idx}...")
        try:
            final_features_array_chunk = np.concatenate(
                aggregated_features_list_chunk, axis=0
            )
            del aggregated_features_list_chunk  # Free memory
            gc.collect()
        except Exception as e:
            logger.error(
                f"Error concatenating features for chunk {chunk_idx}: {e}. Skipping chunk."
            )
            del all_original_texts_chunk
            del batch_patch_counts
            del aggregated_features_list_chunk  # Ensure this is gone
            gc.collect()
            # Skip to next chunk...
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue

        total_patches_in_chunk = final_features_array_chunk.shape[0]
        total_original_items_in_chunk = len(all_original_texts_chunk)
        logger.info(
            f"Chunk {chunk_idx} aggregation complete. Total original items: {total_original_items_in_chunk}, Total patches: {total_patches_in_chunk}."
        )
        logger.info(
            f"Chunk aggregation took {time.time() - chunk_aggregate_start_time:.2f} seconds."
        )
        gc.collect()

        # --- Faiss Search for this Chunk ---
        chunk_search_start_time = time.time()
        logger.info(
            f"Performing Faiss search for chunk {chunk_idx} ({total_patches_in_chunk} features)..."
        )

        if final_features_array_chunk.dtype != np.float32:
            final_features_array_chunk = final_features_array_chunk.astype(np.float32)
            gc.collect()

        try:
            distances_chunk, indices_chunk_flat = index.search(
                final_features_array_chunk, 1
            )
            indices_chunk_flat = (
                indices_chunk_flat.ravel()
            )  # Shape [total_patches_in_chunk]
            logger.info(f"Faiss search for chunk {chunk_idx} completed.")
        except Exception as e:
            logger.error(
                f"Error during Faiss search for chunk {chunk_idx}: {e}. Skipping chunk."
            )
            del all_original_texts_chunk
            del batch_patch_counts
            del final_features_array_chunk
            del distances_chunk
            gc.collect()
            # Skip to next chunk...
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue

        logger.info(
            f"Chunk search took {time.time() - chunk_search_start_time:.2f} seconds."
        )
        # Clear features array and distances immediately after search
        del final_features_array_chunk
        del distances_chunk
        gc.collect()

        # --- Reshape Indices Back to Original Structure ---
        chunk_reshape_start_time = time.time()
        logger.info(f"Reshaping indices for chunk {chunk_idx}...")

        # Assuming constant patches_per_item for simplicity based on your description (261)
        # If patches_per_item can vary *per item*, this would need a more complex loop/logic.
        # Assuming features.shape[1] is consistent across items and batches:
        if not batch_patch_counts:
            logger.warning(
                f"No batch patch counts recorded for chunk {chunk_idx}. Cannot reshape indices. Skipping."
            )
            del all_original_texts_chunk
            del indices_chunk_flat
            gc.collect()
            # Skip to next chunk...
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue

        # Assuming patches_per_item is consistent across all data
        # A safer check: verify all patches_per_item in batch_patch_counts are the same
        unique_patch_counts = set([count for items, count in batch_patch_counts])
        if len(unique_patch_counts) != 1:
            logger.error(
                f"Chunk {chunk_idx}: Inconsistent patches_per_item detected: {unique_patch_counts}. Cannot reshape simply. Skipping."
            )
            del all_original_texts_chunk
            del batch_patch_counts
            del indices_chunk_flat
            gc.collect()
            # Skip to next chunk...
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue

        consistent_patches_per_item = unique_patch_counts.pop()

        if (
            indices_chunk_flat.shape[0]
            != total_original_items_in_chunk * consistent_patches_per_item
        ):
            logger.error(
                f"FATAL: Mismatch between total flat indices ({indices_chunk_flat.shape[0]}) and expected total patches ({total_original_items_in_chunk} * {consistent_patches_per_item}) for chunk {chunk_idx}! Skipping save."
            )
            del all_original_texts_chunk
            del batch_patch_counts
            del indices_chunk_flat
            gc.collect()
            # Skip to next chunk...
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue

        try:
            # Reshape flat indices [total_patches] -> [total_original_items, patches_per_item]
            reshaped_indices_chunk = indices_chunk_flat.reshape(
                total_original_items_in_chunk, consistent_patches_per_item
            )
            # Convert the numpy array of arrays/lists for the Dataset column
            reshaped_indices_list_of_lists = reshaped_indices_chunk.tolist()
            logger.info(f"Indices reshaped successfully for chunk {chunk_idx}.")

        except Exception as e:
            logger.error(
                f"Error reshaping indices for chunk {chunk_idx}: {e}. Skipping save."
            )
            del all_original_texts_chunk
            del batch_patch_counts
            del indices_chunk_flat
            if "reshaped_indices_chunk" in locals():
                del reshaped_indices_chunk
            gc.collect()
            # Skip to next chunk...
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue

        del indices_chunk_flat  # Free flat indices memory
        del reshaped_indices_chunk  # Free numpy reshaped array memory
        gc.collect()
        logger.info(
            f"Chunk index reshaping took {time.time() - chunk_reshape_start_time:.2f} seconds."
        )

        # --- Create and Save Chunk Dataset ---
        chunk_dataset_start_time = time.time()
        logger.info(f"Creating and saving dataset for chunk {chunk_idx}...")

        if len(all_original_texts_chunk) != len(reshaped_indices_list_of_lists):
            logger.error(
                f"FATAL: Mismatch between chunk original texts ({len(all_original_texts_chunk)}) and reshaped indices ({len(reshaped_indices_list_of_lists)}) count for chunk {chunk_idx}! Skipping save."
            )
            del all_original_texts_chunk
            del reshaped_indices_list_of_lists
            del batch_patch_counts
            gc.collect()
            # Don't increment chunk index, but log error and potentially exit or mark for retry
            # For now, follow the pattern of skipping the chunk
            chunk_start_index = chunk_end_index
            chunk_idx += 1
            continue  # Move to the next chunk

        chunk_data_dict = {
            "texts": all_original_texts_chunk,
            "feature_indices": reshaped_indices_list_of_lists,  # This is now list[list[int]]
        }

        try:
            chunk_ds = Dataset.from_dict(chunk_data_dict)

            # Define the features for the column containing lists of ints
            # This tells datasets what to expect for the "feature_indices" column
            chunk_ds = chunk_ds.cast_column("feature_indices", Sequence(Value("int32")))

            chunk_save_path = os.path.join(TEMP_DATA_DIR, f"chunk_{chunk_idx}")
            # Push to hub first
            chunk_ds.push_to_hub(OUTPUT_REPO, f"chunk_{chunk_idx}", split="train")
            # Save locally for final combination
            chunk_ds.save_to_disk(chunk_save_path)

            all_processed_chunk_paths.append(chunk_save_path)  # Record the path
            logger.info(
                f"Chunk {chunk_idx} dataset saved to {chunk_save_path}. Rows: {len(chunk_ds)}"
            )

        except Exception as e:
            logger.error(
                f"Failed to create or save chunk {chunk_idx} dataset: {e}",
                exc_info=True,
            )
            # Cleanup
            del chunk_data_dict
            if "chunk_ds" in locals():
                del chunk_ds
            # Error happened during save, still move to the next chunk
        finally:
            # --- Clean up memory for the current chunk ---
            del all_original_texts_chunk
            del reshaped_indices_list_of_lists
            del batch_patch_counts
            del chunk_data_dict
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
        # arrow_files_pattern = os.path.join(TEMP_DATA_DIR, "**", "data-*.arrow")
        # logger.info(f"Searching for arrow files with pattern: {arrow_files_pattern}")

        # final_combined_ds = load_dataset(
        #     "arrow", data_files=arrow_files_pattern, split="train"
        # )

        # Load each chunk dataset and combine them
        chunk_datasets = []
        for chunk_path in all_processed_chunk_paths:
            chunk_ds = load_from_disk(chunk_path)
            chunk_datasets.append(chunk_ds)

        # Combine all chunk datasets into one
        final_combined_ds = Dataset.from_dict(
            {
                "texts": [],
                "feature_indices": [],
            }
        )
        for chunk_ds in chunk_datasets:
            final_combined_ds = Dataset.concatenate(final_combined_ds, chunk_ds)
        # Cast the feature_indices column to the correct type
        final_combined_ds = final_combined_ds.cast_column(
            "feature_indices", Sequence(Value("int32"))
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
