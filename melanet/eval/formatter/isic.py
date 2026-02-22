import pandas as pd
import numpy as np

from typing import Optional

from datasets import Dataset


CSV_IMAGE_COLUMN = "image"
CSV_ID_COLUMN = "lesion_id"


def format_isic_submission(
    predictions: list[int],
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
    sample_ids = list(dataset[lesion_id_column_name])
    image_ids = list(dataset[image_id_column_name])

    one_hot = np.zeros((n_samples, num_classes), dtype=float)
    one_hot[np.arange(n_samples), predictions] = 1.0

    df = pd.DataFrame(one_hot, columns=labels)
    df.insert(0, CSV_ID_COLUMN, sample_ids)
    # df.insert(1, CSV_IMAGE_COLUMN, image_ids)

    if drop_duplicates:
        df = df.drop_duplicates(subset=[CSV_ID_COLUMN], keep="first")

    if csv_submission_path:
        # Optionally save as CSV file
        df.to_csv(
            csv_submission_path,
            index=False
        )
    return df