import os
import shutil
import traceback
import subprocess

from typing import Optional, Union

from datasets import load_from_disk, Dataset, DatasetDict


METACENTRUM_SCRATCH_PATH: Optional[str] = os.getenv("SCRATCHDIR")
DATASET_SCRATCH_PREFIX = "metacentrum_scratch"


def copy_data_dir_to_scratch(dir_path: str) -> Optional[str]:
    """Copy data directory to local scratch"""
    assert METACENTRUM_SCRATCH_PATH, "This function can only be used on Metacentrum"
    try:
        data_dir_scratch = os.path.join(METACENTRUM_SCRATCH_PATH, os.path.basename(dir_path))
        print(f"Copying data to local scratch: {data_dir_scratch}")
        shutil.copytree(dir_path, data_dir_scratch)
        return data_dir_scratch
    except Exception:
        print(f"ERROR:\n{traceback.format_exc()}")
        return None


def clear_scratch() -> bool:
    assert METACENTRUM_SCRATCH_PATH, "This function can only be used on Metacentrum"
    try:
        _clear_scratch_wild_card = os.path.join(METACENTRUM_SCRATCH_PATH, "*")
        subprocess.run(["rm", "-rf", _clear_scratch_wild_card], check=True)
        return True
    except Exception:
        print(f"ERROR:\n{traceback.format_exc()}")
        return False


def load_dataset_from_scratch(dataset_name: str) -> Union[Dataset, DatasetDict, None]:
    if dataset_name.startswith(DATASET_SCRATCH_PREFIX):
        _path_on_scratch = os.path.normpath(dataset_name).split(os.path.sep, maxsplit=1)[1]
        _dataset_scratch = os.path.join(METACENTRUM_SCRATCH_PATH, _path_on_scratch)
        return load_from_disk(_dataset_scratch)
    elif dataset_name.startswith(METACENTRUM_SCRATCH_PATH):
        return load_from_disk(dataset_name)
    else:
        return None