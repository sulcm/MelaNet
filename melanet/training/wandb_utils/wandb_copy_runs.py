# Implementation based on WandB Issue #5726 - Missing feature to copy runs between projects
# GitHub Link: https://github.com/wandb/wandb/issues/5726

import wandb
import json
import math
import numpy as np

from wandb.apis.public.runs import Run

from tqdm import tqdm
from argparse import ArgumentParser


def is_none_like(value) -> bool:
    # Check actual None
    if value is None:
        return True

    # Check numeric NaN
    if isinstance(value, (int, float)) and math.isnan(value):
        return True

    # Check string representations
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nil", "n/a", "nan"}:
        return True

    # Check numpy NaN-like values
    try:
        if np.isnan(value):
            return True
    except TypeError:
        pass

    return False


def main(args):
    if args.names is not None:
        if args.runs is None:
            raise ValueError("Renaming copied runs not supported when copying whole project.")
        assert len(args.names) == len(args.runs), "Number of new names must equal number of run IDs"

    args.dst_entity = args.dst_entity if args.dst_entity is not None else args.src_entity
    args.dst_project = args.dst_project if args.dst_project is not None else args.src_project
    same_project = args.src_entity == args.dst_entity and args.src_project == args.dst_project
    name_append = "-copy" if same_project and args.names is None else ""

    # Initialize the wandb API
    if args.force_login:
        # Set your API key
        wandb.login()
    api = wandb.Api()

    # Get the runs from the source project
    runs: list[Run] = api.runs(f"{args.src_entity}/{args.src_project}")
    # Iterate through the runs and copy them to the destination project
    for run in tqdm(runs, desc="Copying runs"):
        if args.runs is not None and run.id not in args.runs:
            continue

        config  = run.config
        if isinstance(config, str):
            config = json.loads(config)
        if args.prj_name_as_tag:
            config["_model_type_tag"] = args.src_project

        name = run.name if args.names is None else args.names[args.runs.index(run.id)]

        # Create a new run in the destination project
        # Log the history to the new run
        with wandb.init(
            project=args.dst_project,
            entity=args.dst_entity,
            config=config,
            name=name + name_append,
            tags=[args.src_project] if args.prj_name_as_tag else None,
            resume="allow"
        ) as new_run:
            history_rows = run.history(samples=run.lastHistoryStep + 1)
            system = run.history(samples=run.lastHistoryStep + 1, stream="system")
            history_rows = history_rows.join(system, rsuffix="_system")
            for index, row in history_rows.iterrows():
                new_run.log({
                    k: v
                    for k, v in row.to_dict().items()
                    if not is_none_like(v)
                })

            # Upload the files to the new run
            files = run.files()
            for file in files:
                file.download(replace=True)
                new_run.save(file.name, policy="now")


if __name__ == "__main__":
    parser = ArgumentParser(description="Copies one or all of the runs in a wandb project to another.")
    parser.add_argument("-se", "--src-entity", type=str, help="Source wandb entity name.")
    parser.add_argument("-sp", "--src-project", type=str, help="Name of the wandb projecet.")
    parser.add_argument("-de", "--dst-entity", type=str, help="Destination wandb entity name.")
    parser.add_argument("-dp", "--dst-project", type=str, help="Name of destination wandb project.")
    parser.add_argument("-r", "--runs", nargs="*", type=str, default=None, help="List of run IDs to copy. If None will copy all in project.")
    parser.add_argument("-n", "--names", nargs="*", type=str, default=None, help="List of new names for copied runs (optional).")
    parser.add_argument("-pnt", "--prj-name-as-tag", action="store_true", default=False, help="Copied runs will have tag same as `src-project` name.")
    parser.add_argument("-fl", "--force-login", action="store_true", default=False, help="Force `wandb.login()` before init.")

    main(parser.parse_args())