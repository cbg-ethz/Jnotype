"""Utilities for downloading data sets over HTTPS."""
import hashlib
import pathlib
import requests

from typing import Optional, Union
from tqdm import tqdm

DATA_SETTINGS = {
    "path": "~/data-jnotype",
}


def download_file(
    address: str, filepath: Union[str, pathlib.Path], md5sum: Optional[str] = None
) -> None:
    """Downloads file from `address` and saves it to `filepath`.
    If `md5sum` is provided, checks the MD5 checksum of the downloaded file
    against this value.
    """
    response = requests.get(address, stream=True)

    # Check if the request was successful
    if response.status_code != 200:
        response.raise_for_status()

    # Initialize the progress bar
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    # Initialize MD5 hasher
    md5_hasher = hashlib.md5()

    # Open the destination file in write-binary mode
    with open(filepath, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            # Write chunk to file
            file.write(chunk)

            # Update the MD5 hasher with the chunk
            md5_hasher.update(chunk)

            # Update the progress bar
            progress_bar.update(len(chunk))

    # Finalize the progress bar
    progress_bar.close()

    # Calculate MD5 checksum
    calculated_md5sum = md5_hasher.hexdigest()

    # Compare checksums
    if md5sum is not None and calculated_md5sum != md5sum:
        raise ValueError(
            f"Checksum mismatch: {calculated_md5sum} (calculated) vs "
            f"{md5sum} (expected)"
        )
