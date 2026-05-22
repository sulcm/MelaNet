# Implementation based on WandB Issue #5726 - Missing feature to copy runs between projects
# GitHub Link: https://github.com/wandb/wandb/issues/5726

import wandb
import json

from tqdm import tqdm
from argparse import ArgumentParser


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
    runs = api.runs(f"{args.src_entity}/{args.src_project}")
    # Iterate through the runs and copy them to the destination project
    for run in tqdm(runs):
        if args.runs is not None and run.id not in args.runs:
            continue

        # Get the run history and files
        history = run.history(samples=run.lastHistoryStep + 1)
        system = run.history(samples=run.lastHistoryStep + 1, stream="system")
        history = history.join(system, rsuffix="_system")
        files = run.files()

        config  = run.config
        if isinstance(config, str):
            config = json.loads(config)

        name = run.name if args.names is None else args.names[args.runs.index(run.id)]

        # Create a new run in the destination project
        # Log the history to the new run
        new_run = wandb.init(
            project=args.dst_project,
            entity=args.dst_entity,
            config=config,
            name=name + name_append,
            resume="allow"
        )

        # Recreate logs in new run
        for index, row in history.iterrows():
            new_run.log(row.to_dict())

        # Upload the files to the new run
        for file in files:
            file.download(replace=True)
            new_run.save(file.name, policy="now")

        # Finish the new run
        new_run.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Copies one or all of the runs in a wandb project to another.")
    parser.add_argument("-se", "--src-entity", type=str, help="Source wandb entity name.")
    parser.add_argument("-sp", "--src-project", type=str, help="Name of the wandb projecet.")
    parser.add_argument("-de", "--dst-entity", type=str, help="Destination wandb entity name.")
    parser.add_argument("-dp", "--dst-project", type=str, help="Name of destination wandb project.")
    parser.add_argument("-r", "--runs", nargs="*", type=str, default=None, help="List of run IDs to copy. If None will copy all in project.")
    parser.add_argument("-n", "--names", nargs="*", type=str, default=None, help="List of new names for copied runs (optional).")
    parser.add_argument("-fl", "--force-login", action="store_true", default=False, help="Force `wandb.login()` before init.")

    main(parser.parse_args())