import pandas as pd
import numpy as np

from typing import Optional, Union

from datasets import Dataset


CSV_IMAGE_COLUMN = "image"
CSV_ID_COLUMN = "lesion_id"

MILK10k_LABELS = [
    "akiec",
    "bcc",
    "ben_oth",
    "bkl",
    "df",
    "inf",
    "mal_oth",
    "mel",
    "nv",
    "sccka",
    "vasc"
]


def format_isic_submission(
    predictions: Union[list[int], dict[str, int]],
    labels: list[str],
    dataset: Dataset,
    image_id_column_name: str,
    lesion_id_column_name: str,
    csv_submission_path: Optional[str] = None,
    drop_duplicates: bool = False
) -> pd.DataFrame:
    assert labels, "Labels must be provided to generate ISIC submission"
    labels = [name.upper() for name in labels]
    num_classes = len(labels)
    n_samples = len(predictions)
    preds_with_ids = isinstance(predictions, dict)

    one_hot = np.zeros((n_samples, num_classes), dtype=float)
    if preds_with_ids:
        sample_ids = list(predictions.keys())
        assert set(dataset[lesion_id_column_name]) == set(sample_ids)
        one_hot[np.arange(n_samples), list(predictions.values())] = 1.0
    else:
        sample_ids = list(dataset[lesion_id_column_name])
        one_hot[np.arange(n_samples), predictions] = 1.0

    df = pd.DataFrame(one_hot, columns=labels)
    df.insert(0, CSV_ID_COLUMN, sample_ids)
    # df.insert(1, CSV_IMAGE_COLUMN, image_ids)

    if drop_duplicates and not preds_with_ids:
        df = df.drop_duplicates(subset=[CSV_ID_COLUMN], keep="first")

    if csv_submission_path:
        # Optionally save as CSV file
        df.to_csv(
            csv_submission_path,
            index=False
        )
    return df