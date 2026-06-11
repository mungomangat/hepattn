import comet_ml  # noqa: F401
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.colliderml_pixel.data import ColliderMLPixelDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class ColliderMLPixelTracker(ModelWrapper):
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
        query_mask = targets.get("query_mask")
        if query_mask is not None:
            query_mask = query_mask.bool()

        for layer_name, layer_preds in preds.items():
            if "track_sihit_valid" not in layer_preds:
                continue
            mask = layer_preds["track_sihit_valid"]["track_sihit_valid"]
            if mask is not None:
                num_valid = mask.sum(-1).float()
                frac_valid = num_valid / mask.shape[-1]
                if query_mask is None:
                    self.log(f"{stage}/{layer_name}_avg_num_valid_sihits", torch.mean(num_valid), sync_dist=True)
                    self.log(f"{stage}/{layer_name}_avg_frac_valid_sihits", torch.mean(frac_valid), sync_dist=True)
                else:
                    valid_num = num_valid[query_mask]
                    valid_frac = frac_valid[query_mask]
                    if valid_num.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_num_valid_sihits", valid_num.mean(), sync_dist=True)
                    if valid_frac.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_frac_valid_sihits", valid_frac.mean(), sync_dist=True)

        preds = preds["final"]

        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        if query_mask is not None:
            pred_valid = pred_valid & query_mask

        pred_hit_masks = preds["track_sihit_valid"]["track_sihit_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets["particle_sihit_valid"] & true_valid.unsqueeze(-1)

        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)
        hit_p = pred_hit_masks.sum(-1)
        hit_t = true_hit_masks.sum(-1)

        for wp in [0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid
            effs = (hit_tp / (hit_t + 1e-8) >= wp) & both_valid
            purs = (hit_tp / (hit_p + 1e-8) >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / (true_valid.float().sum(-1) + 1e-8)
            roi_purs = purs.float().sum(-1) / (pred_valid.float().sum(-1) + 1e-8)

            self.log(f"{stage}/p{wp}_eff", roi_effs.nanmean(), sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", roi_purs.nanmean(), sync_dist=True)

        pred_num = pred_valid.sum(-1)
        true_num = true_valid.sum(-1)

        nh_per_true = hit_t.float()[true_valid].mean()
        nh_per_pred = hit_p.float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", nh_per_true, sync_dist=True)
        self.log(f"{stage}/nh_per_track", nh_per_pred, sync_dist=True)
        self.log(f"{stage}/num_tracks", pred_num.float().mean(), sync_dist=True)
        self.log(f"{stage}/num_particles", true_num.float().mean(), sync_dist=True)

        num_sihits_total = float(pred_hit_masks.shape[-1])
        num_sihits_valid = float(true_hit_masks.sum())
        self.log(f"{stage}/num_sihits", num_sihits_total, sync_dist=True)
        self.log(f"{stage}/num_sihits_valid", num_sihits_valid, sync_dist=True)
        self.log(f"{stage}/num_sihits_noise", num_sihits_total - num_sihits_valid, sync_dist=True)


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ColliderMLPixelTracker,
        datamodule_class=ColliderMLPixelDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
