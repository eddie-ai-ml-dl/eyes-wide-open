# src/utils/metadata_manager.py

import json
import os
import datetime

def load_metadata(filepath: str) -> list:
    """
    Loads metadata from a JSON file. Returns an empty list if file doesn't exist.

    Args:
        filepath (str): The path to the JSON metadata file.

    Returns:
        list: A list of dictionaries representing the metadata.
    """
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filepath}': {e}. Returning empty list.")
        return []
    except Exception as e:
        print(f"Error loading metadata from '{filepath}': {e}. Returning empty list.")
        return []

def save_metadata(data: list, filepath: str):
    """
    Saves metadata (a list of dictionaries) to a JSON file.

    Args:
        data (list): The list of dictionaries to save.
        filepath (str): The path to the output JSON metadata file.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving metadata to '{filepath}': {e}")

def update_metadata_entry(metadata: list, crop_id: str, updates: dict) -> list:
    """
    Updates a specific entry in the metadata list identified by crop_id.
    Adds a 'last_updated_timestamp' to the entry.
    Returns the updated metadata list (or original if crop_id not found).

    Args:
        metadata (list): The list of dictionaries (metadata).
        crop_id (str): The unique identifier for the crop to update.
        updates (dict): A dictionary of fields and new values to update.

    Returns:
        list: The metadata list with the specified entry updated.
    """
    found = False
    for entry in metadata:
        if entry.get("crop_id") == crop_id:
            entry.update(updates)
            entry['last_updated_timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
            found = True
            break
    if not found:
        print(f"Warning: Crop ID '{crop_id}' not found in metadata for update.")
    return metadata

# Add other metadata management functions here in the future
# e.g., filter_metadata, merge_metadata, etc.