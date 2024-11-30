from io import FileIO, TextIOWrapper
import os
import wget
import hashlib

# Directory to store downloaded resources
RESOURCE_DIR = os.path.join(os.path.expanduser("~"), ".locpro")
os.makedirs(RESOURCE_DIR, exist_ok=True)

# Dictionary containing resource metadata
RESOURCE_DICT = {
    "cafa4_min.pkl": {
        "url": "http://locpro.idrblab.cn/files/ekloc/resources/cafa4_min.pkl",
        "md5sum": "39bc2be2e7b07cf2ab6a16cb341ef1db",
    },
    "cafa4_max.pkl": {
        "url": "http://locpro.idrblab.cn/files/ekloc/resources/cafa4_max.pkl",
        "md5sum": "c78c2e0048fc34383b2db0dc0dbc7886",
    },
    "cafa4_del.csv": {
        "url": "http://locpro.idrblab.cn/files/ekloc/resources/cafa4_del.csv",
        "md5sum": "d00b71439084cb19b7d3d0d4fbbaa819",
    },
    "data_grid.pkl": {
        "url": "http://locpro.idrblab.cn/files/ekloc/resources/data_grid.pkl",
        "md5sum": "fb2d2d86a4bc21c6e60fac996b3a90d3",
    },
    "row_asses.pkl": {
        "url": "http://locpro.idrblab.cn/files/ekloc/resources/row_asses.pkl",
        "md5sum": "bf9bb1eda744a60c381d19b275ac6f33",
    },
}


def calculate_md5(file_path):
    """
    Compute the MD5 checksum for the given file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: MD5 checksum of the file.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):  # Read in 64KB chunks
            md5.update(chunk)
    return md5.hexdigest()


def validate_md5(file_path, expected_md5):
    """
    Validate the file's MD5 checksum against an expected value.

    Args:
        file_path (str): Path to the file.
        expected_md5 (str): Expected MD5 checksum.

    Returns:
        bool: True if checksum matches, False otherwise.
    """
    return calculate_md5(file_path) == expected_md5


def download_resource(name, overwrite = False):
    """
    Download a resource file if it doesn't exist or fails MD5 validation.

    Args:
        name (str): Name of the resource.
        overwrite (bool): Whether to overwrite the existing file.

    Returns:
        str: Path to the downloaded file.

    Raises:
        FileNotFoundError: If the resource name is invalid.
        RuntimeError: If the downloaded file fails MD5 validation.
    """
    if name not in RESOURCE_DICT:
        raise FileNotFoundError(f"Invalid resource name: {name}")
    resource = RESOURCE_DICT[name]
    file_path = os.path.join(RESOURCE_DIR, name)
    # Check if the file exists and is valid
    if os.path.exists(file_path) and not overwrite:
        if validate_md5(file_path, resource["md5sum"]):
            return file_path
        os.remove(file_path)  # Remove invalid file
        
    # Download the resource
    print(f"Downloading {name}...")
    wget.download(resource["url"], out=file_path)
    print(f"\nValidating MD5 checksum for {name}...")

    # Validate the downloaded file
    if not validate_md5(file_path, resource["md5sum"]):
        os.remove(file_path)
        raise RuntimeError(f"MD5 checksum validation failed for {name}")

    return file_path



def get_resource_path(name):
    """
    Retrieve the local path of a resource, downloading it if necessary.

    Args:
        name (str): Name of the resource.

    Returns:
        str: Path to the resource file.
    """
    return download_resource(name)

def open_binary(name):
    """
    Open a resource file in binary mode.

    Args:
        name (str): Name of the resource.

    Returns:
        FileIO: Binary file object.
    """
    return open(get_resource_path(name), "rb")