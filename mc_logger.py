import os
import wandb
from pathlib import Path

# Do not upload git patch or .py files on W&B server
os.environ["WANDB_IGNORE_GLOBS"] = "*.patch,*.py,*/*.py,*/*/*.py"
wandb.login(key="78ac4f3117cc874807a7e334d35512dfcd196d0e")

class classproperty(object):
    """ Helper to get a class property decorator."""

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class Logger(object):
    """ """

    def __init__(
        self, project_name, entity, bucket_path, local_log_dir, group_id, args
    ):
        """
        Logger class wrapper.
        Update results to wandb sever.
        Create a unique folder to store the training logs.
        Provide a method to synchronise to a GCP bucket.
        
        Args:
            project_name : project anme
            entity : name of the team. All members of this team will see the logs.
            root_bucket_path: bucket path where to save the results
            root_log_dir: local directory where to save the results.
            config : config to log.
            experiment_id (str) : experiment id if we want to resume an experiment
        """
        base_path = Path(local_log_dir).name
        assert (
            base_path == group_id
        ), "The local directory basename should be equal to the experiment id."

        bucket_path = os.path.join(bucket_path, str(base_path))
        
        wandb.init(
            project=project_name,
            entity=entity,
            dir=local_log_dir,
            config=args,
            save_code=True,
            group=group_id,
            id = group_id + "_" + wandb.util.generate_id()
        )

        # save local and bucket paths in wandb config for easy access
        wandb.config.update({"bucket_path": bucket_path, "local_path": local_log_dir})

    @classmethod
    def log(cls, log_dict, step=None):
        """
        Add log information.
        """
        
        if wandb.run is not None:
            wandb.log(log_dict, step=step)

    @classmethod
    def sync(cls):
        """
        Synchronize local log dir and GCP bucket
        """
        if wandb.run is not None:
            os.system(
                "gsutil -m rsync -r {} {}".format(
                    wandb.config.local_path, wandb.config.bucket_path
                )
            )
            print("Sync successful")

    @classproperty
    def log_dir(cls):
        """
        Returns where results are saved
        """
        if wandb.run is not None:
            return wandb.run.dir

    @classproperty
    def config(cls):
        """
        Returns config saved with wandb
        """
        if wandb.run is not None:
            return wandb.config
