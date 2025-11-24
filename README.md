# MelaNet

---

## Image-based Skin Cancer Recognition

The work was carried out as part of a master's thesis (2025/2026) at the Faculty of Applied Sciences, University of West Bohemia in Pilsen, in the field of Artificial Intelligence and Automation.

---

## Requirements

- Create virtual enviroment (optional but useful, example provided using [Anaconda](https://anaconda.org/anaconda/conda) package manager)
    ```bash
    conda create -n melanet python=3.12.8 -y
    ```
    Command for creation of virtual enviroment called `melanet` with Python version `3.12.8` that automatically accepts any confirmations during installation.

- To activate created Anaconda enviroment run
    ```bash
    conda activate melanet
    ```

- Install required dependencies for model training and evaluations (recommended using virtual enviroments)
    ```bash
    pip install -r melanet/requirements.txt
    ```
    Command for installing required dependencies defined in `melanet/requirements.txt` file.