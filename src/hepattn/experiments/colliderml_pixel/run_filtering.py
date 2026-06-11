import comet_ml  # noqa: F401
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.colliderml_pixel.data import ColliderMLPixelFilterDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class ColliderMLPixelFilter(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

    def log_custom_metrics(self, preds, targets, stage):
        task = self.model.tasks[0]
        target_field = task.target_field       # "on_valid_particle"
        input_object = task.input_object       # "sihit"
        expected_key = f"{input_object}_{target_field}"  # "sihit_on_valid_particle"

        pred = preds["final"]["sihit_filter"][expected_key]
        true = targets[expected_key]

        tp = (pred & true).sum()
        tn = ((~pred) & (~true)).sum()

        metrics = {
            "nh_total_pre": float(pred.shape[1]),
            "nh_total_post": float(pred.sum()),
            "nh_pred_true": pred.float().sum(),
            "nh_pred_false": (~pred).float().sum(),
            "nh_valid_pre": true.float().sum(),
            "nh_valid_post": (pred & true).float().sum(),
            "nh_noise_pre": (~true).float().sum(),
            "nh_noise_post": (pred & ~true).float().sum(),
            "acc": (pred == true).half().mean(),
            "valid_recall": tp / true.sum(),
            "valid_precision": tp / pred.sum(),
            "noise_recall": tn / (~true).sum(),
            "noise_precision": tn / (~pred).sum(),
            "num_particles": targets["particle_valid"].float().sum(),
        }

        for metric_name, metric_value in metrics.items():
            self.log(f"{stage}/{metric_name}", metric_value, sync_dist=True, batch_size=1)


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ColliderMLPixelFilter,
        datamodule_class=ColliderMLPixelFilterDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
