import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

from matplotlib.axes import Axes
from PIL.Image import Image

from datasets import Dataset, ClassLabel


def draw_pie_chart_with_leader_lines(
    ax: Axes,
    data: list,
    labels: list,
    popout_percentage: bool = False,
    add_legend: bool = False
) -> Axes:
    wedges, *texts = ax.pie(
        data,
        autopct=None if popout_percentage else "%1.1f%%",
        pctdistance=0.75,
        wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )
    ax.axis("equal")

    x = np.asarray(data, np.float32)
    if x.ndim > 1:
        raise ValueError("x must be 1D")
    if np.any(x < 0):
        raise ValueError("Wedge sizes 'x' must be non negative values")
    sx = x.sum()
    perc = (x / sx) * 100

    # Calculate angles and positions
    angles = [(w.theta2 + w.theta1) / 2 for w in wedges]
    r = 1.0  # pie radius
    x = np.cos(np.deg2rad(angles))
    y = np.sin(np.deg2rad(angles))

    # Split into left/right halves
    left_idx = np.where(x < 0)[0]
    right_idx = np.where(x >= 0)[0]

    # Sort by y (top to bottom)
    left_idx = left_idx[np.argsort(y[left_idx])]
    right_idx = right_idx[np.argsort(y[right_idx])]

    def spread_positions(y_values, min_gap=0.08):
        """Ensure minimum vertical spacing between labels to avoid overlap."""
        new_y = y_values.copy()
        for i in range(1, len(new_y)):
            if new_y[i] - new_y[i - 1] < min_gap:
                new_y[i] = new_y[i - 1] + min_gap
        return new_y

    # Apply spacing
    y[left_idx] = spread_positions(y[left_idx])
    y[right_idx] = spread_positions(y[right_idx])

    # Draw labels and leader lines
    for i, (label, pct) in enumerate(zip(labels, perc)):
        # Start of leader line (on pie edge)
        start_x = r * np.cos(np.deg2rad(angles[i]))
        start_y = r * np.sin(np.deg2rad(angles[i]))

        # End of leader line (outside)
        end_x = 1.3 * np.sign(x[i])
        end_y = y[i]

        # Draw connecting line
        ax.plot([start_x, end_x * 0.9], [start_y, end_y], color="gray", lw=0.8)
        ax.plot([end_x * 0.9, end_x], [end_y, end_y], color="gray", lw=0.8)

        # Format label with percentage
        label_text = f"{label}"
        if popout_percentage:
            label_text += f" ({pct:.2f}%)"
        align = "left" if x[i] > 0 else "right"

        # Add text
        ax.text(end_x, end_y, label_text, ha=align, va="center", fontsize=9)

    if add_legend:
        ax.legend(wedges, labels, bbox_to_anchor=(1, 0.5), loc="center left")

    return ax


def sample_dataset_and_display(
    dataset: Dataset,
    image_column: str,
    n: int,
    class_label_column: Optional[str] = None,
    seed: Optional[int] = None,
    cols: Optional[int] = None,
    resize: Optional[tuple] = None,
    show_titles: bool = True,
    save_path: Optional[str] = None,
    **save_kwargs
) -> None:
    """
    From dataset randomly sample `n` images and display them in a grid.

    Args:
      dataset: dataset to sample.
      image_column: name of the column that contains the image. If None, attempts to auto-detect.
      n: number of images to sample
      class_label_column: column with labels
      seed: random seed for reproducibility
      cols: number of columns in the grid. If None, computed automatically (approx sqrt).
      resize: optional (w,h) to resize thumbnails for consistent display
      show_titles: whether to display a small title with index & possible label
    """
    # Shuffle and pick n indices (efficient and safe)
    n = min(n, len(dataset))

    sampled = dataset.shuffle(seed=seed).select(range(n))
    indices = sampled._indices.to_pydict()["indices"]

    # Build grid size
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig_w = cols * 3
    fig_h = rows * 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for ax in axes:
        ax.axis("off")

    for i, item in enumerate(sampled):
        pil_img: Image = item[image_column]
        label = item[class_label_column] if class_label_column else None
        # raw = item[image_column]
        # pil_img = _to_pil(raw)
        if pil_img is None:
            # draw a placeholder
            placeholder = Image.new("RGB", (256, 256), color=(220, 220, 220))
            axes[i].imshow(placeholder)
            axes[i].text(0.5, 0.5, "No image", ha="center", va="center")
        else:
            if resize is not None:
                pil_img = pil_img.thumbnail(resize, resample=Image.LANCZOS)
            axes[i].imshow(pil_img)
        if show_titles:
            if label is not None:
                if isinstance(sampled.features[class_label_column], ClassLabel):
                    label = sampled.features[class_label_column].int2str(label)
                title = f"{label} (ID: {indices[i]})"
            else:
                title = f"{indices[i]}"
            axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")

    # hide any leftover axes
    for j in range(n, rows * cols):
        axes[j].axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, **save_kwargs)
    plt.show()