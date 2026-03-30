import comet_ml  # noqa: F401
import numpy as np
import torch
from lightning.pytorch.cli import ArgsType
from torch import Tensor, nn
from pathlib import Path

from hepattn.experiments.trackml.data import TrackMLDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class TrackMLTracker(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
        self._first_hit_eta_edges = np.linspace(-5.0, 5.0, 81)
        self._first_hit_r_edges = np.linspace(0.0, 1.5, 81)
        self._first_hit_stats: dict[str, dict[str, torch.Tensor]] = {}

    def _reset_first_hit_stats(self, stage: str) -> None:
        eta_bins = len(self._first_hit_eta_edges) - 1
        r_bins = len(self._first_hit_r_edges) - 1
        self._first_hit_stats[stage] = {
            "eta_tp": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_fn": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_fp": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_true": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_pred": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_r_tp": torch.zeros((eta_bins, r_bins), dtype=torch.float64),
            "eta_r_fn": torch.zeros((eta_bins, r_bins), dtype=torch.float64),
            "eta_r_fp": torch.zeros((eta_bins, r_bins), dtype=torch.float64),
        }

    def _accumulate_first_hit_hists(
        self,
        stage: str,
        eta_vals: np.ndarray,
        r_vals: np.ndarray,
        tp_mask: np.ndarray,
        fn_mask: np.ndarray,
        fp_mask: np.ndarray,
        true_mask: np.ndarray,
        pred_mask: np.ndarray,
    ) -> None:
        stats = self._first_hit_stats[stage]
        eta_edges = self._first_hit_eta_edges
        r_edges = self._first_hit_r_edges

        stats["eta_tp"] += torch.from_numpy(np.histogram(eta_vals[tp_mask], bins=eta_edges)[0]).to(dtype=torch.float64)
        stats["eta_fn"] += torch.from_numpy(np.histogram(eta_vals[fn_mask], bins=eta_edges)[0]).to(dtype=torch.float64)
        stats["eta_fp"] += torch.from_numpy(np.histogram(eta_vals[fp_mask], bins=eta_edges)[0]).to(dtype=torch.float64)
        stats["eta_true"] += torch.from_numpy(np.histogram(eta_vals[true_mask], bins=eta_edges)[0]).to(dtype=torch.float64)
        stats["eta_pred"] += torch.from_numpy(np.histogram(eta_vals[pred_mask], bins=eta_edges)[0]).to(dtype=torch.float64)

        stats["eta_r_tp"] += torch.from_numpy(np.histogram2d(eta_vals[tp_mask], r_vals[tp_mask], bins=[eta_edges, r_edges])[0]).to(
            dtype=torch.float64
        )
        stats["eta_r_fn"] += torch.from_numpy(np.histogram2d(eta_vals[fn_mask], r_vals[fn_mask], bins=[eta_edges, r_edges])[0]).to(
            dtype=torch.float64
        )
        stats["eta_r_fp"] += torch.from_numpy(np.histogram2d(eta_vals[fp_mask], r_vals[fp_mask], bins=[eta_edges, r_edges])[0]).to(
            dtype=torch.float64
        )

    def _update_first_hit_stats(self, inputs: dict[str, Tensor], preds: dict, targets: dict[str, Tensor], stage: str) -> None:
        stage_stats = self._first_hit_stats.get(stage)
        if stage_stats is None:
            return

        encoder_preds = preds.get("encoder", {})
        query_init_preds = encoder_preds.get("query_init")
        if query_init_preds is None:
            return

        pred_first = query_init_preds.get("hit_is_first")
        true_first = targets.get("hit_is_first")
        hit_eta = inputs.get("hit_eta")
        hit_r = inputs.get("hit_r")
        if pred_first is None or true_first is None or hit_eta is None or hit_r is None:
            return

        pred_first = pred_first.bool()
        true_first = true_first.bool()
        hit_valid = targets.get("hit_valid")
        valid = hit_valid.bool() if hit_valid is not None else torch.ones_like(true_first, dtype=torch.bool)

        tp = pred_first & true_first & valid
        fn = (~pred_first) & true_first & valid
        fp = pred_first & (~true_first) & valid
        true_mask = true_first & valid
        pred_mask = pred_first & valid

        eta_vals = hit_eta.detach().to(torch.float32).cpu().numpy().reshape(-1)
        r_vals = hit_r.detach().to(torch.float32).cpu().numpy().reshape(-1)

        self._accumulate_first_hit_hists(
            stage=stage,
            eta_vals=eta_vals,
            r_vals=r_vals,
            tp_mask=tp.detach().cpu().numpy().reshape(-1),
            fn_mask=fn.detach().cpu().numpy().reshape(-1),
            fp_mask=fp.detach().cpu().numpy().reshape(-1),
            true_mask=true_mask.detach().cpu().numpy().reshape(-1),
            pred_mask=pred_mask.detach().cpu().numpy().reshape(-1),
        )

    def _gather_hist(self, hist: torch.Tensor) -> torch.Tensor:
        gathered = self.all_gather(hist.to(self.device))
        if gathered.dim() == hist.dim():
            return gathered.detach().cpu()
        return gathered.sum(dim=0).detach().cpu()

    def _save_first_hit_plots(self, stage: str, stats: dict[str, torch.Tensor]) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        out_dir = Path(self.trainer.default_root_dir) / "first_hit_diagnostics" / f"{stage}_epoch{int(self.current_epoch):03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        eta_edges = self._first_hit_eta_edges
        r_edges = self._first_hit_r_edges
        eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
        eta_width = eta_edges[1] - eta_edges[0]

        eta_tp = stats["eta_tp"].numpy()
        eta_fn = stats["eta_fn"].numpy()
        eta_fp = stats["eta_fp"].numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(eta_centers, eta_tp, where="mid", label="TP (correct first-hit predictions)")
        ax.step(eta_centers, eta_fn, where="mid", label="FN (missed true first hits)")
        ax.step(eta_centers, eta_fp, where="mid", label="FP (incorrect first-hit predictions)")
        ax.set_xlabel("eta")
        ax.set_ylabel("count")
        ax.set_title(f"{stage}: first-hit prediction outcomes vs eta")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "first_hit_outcomes_eta.png", dpi=180)
        plt.close(fig)

        eta_true = stats["eta_true"].numpy()
        eta_pred = stats["eta_pred"].numpy()
        eta_recall = np.divide(eta_tp, eta_true, out=np.zeros_like(eta_tp, dtype=float), where=eta_true > 0)
        eta_precision = np.divide(eta_tp, eta_pred, out=np.zeros_like(eta_tp, dtype=float), where=eta_pred > 0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(eta_centers - 0.5 * eta_width, eta_recall, width=eta_width, label="recall", alpha=0.7)
        ax.bar(eta_centers + 0.5 * eta_width, eta_precision, width=eta_width, label="precision", alpha=0.7)
        ax.set_xlabel("eta")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{stage}: first-hit recall/precision vs eta")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "first_hit_recall_precision_eta.png", dpi=180)
        plt.close(fig)

        extent = [eta_edges[0], eta_edges[-1], r_edges[0], r_edges[-1]]
        eta_r_maps = [
            ("TP (correct)", stats["eta_r_tp"].numpy(), "Greens"),
            ("FN (missed)", stats["eta_r_fn"].numpy(), "Blues"),
            ("FP (incorrect)", stats["eta_r_fp"].numpy(), "Reds"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        for ax, (title, arr, cmap) in zip(axes, eta_r_maps):
            im = ax.imshow(arr.T, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap=cmap)
            ax.set_xlabel("eta")
            ax.set_ylabel("r [m]")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"{stage}: first-hit prediction outcomes in eta-r")
        fig.savefig(out_dir / "first_hit_outcomes_eta_r.png", dpi=180)
        plt.close(fig)

    def _finalize_first_hit_stats(self, stage: str) -> None:
        stats = self._first_hit_stats.get(stage)
        if stats is None:
            return

        gathered = {k: self._gather_hist(v) for k, v in stats.items()}
        tp_total = gathered["eta_tp"].sum()
        fn_total = gathered["eta_fn"].sum()
        fp_total = gathered["eta_fp"].sum()
        true_total = gathered["eta_true"].sum()
        pred_total = gathered["eta_pred"].sum()

        recall = tp_total / true_total if true_total > 0 else 0.0
        precision = tp_total / pred_total if pred_total > 0 else 0.0
        miss_rate = fn_total / true_total if true_total > 0 else 0.0
        false_discovery = fp_total / pred_total if pred_total > 0 else 0.0

        self.log(f"{stage}/first_hit_tp", float(tp_total), sync_dist=True)
        self.log(f"{stage}/first_hit_fn", float(fn_total), sync_dist=True)
        self.log(f"{stage}/first_hit_fp", float(fp_total), sync_dist=True)
        self.log(f"{stage}/first_hit_recall", float(recall), sync_dist=True)
        self.log(f"{stage}/first_hit_precision", float(precision), sync_dist=True)
        self.log(f"{stage}/first_hit_miss_rate", float(miss_rate), sync_dist=True)
        self.log(f"{stage}/first_hit_false_discovery", float(false_discovery), sync_dist=True)

        if self.trainer.is_global_zero:
            self._save_first_hit_plots(stage, gathered)

    def log_custom_metrics(self, preds, targets, stage):
        query_mask = targets.get("query_mask")
        if query_mask is not None:
            query_mask = query_mask.bool()

        # log intermediate layer mask predictions
        for layer_name, layer_preds in preds.items():
            # Skip layers that don't have track_hit_valid task (e.g., encoder layer)
            if "track_hit_valid" not in layer_preds:
                continue
            mask = layer_preds["track_hit_valid"]["track_hit_valid"]
            if mask is not None:
                num_valid = mask.sum(-1).float()
                frac_valid = num_valid / mask.shape[-1]

                if query_mask is None:
                    self.log(f"{stage}/{layer_name}_avg_num_valid_hits", torch.mean(num_valid), sync_dist=True)
                    self.log(f"{stage}/{layer_name}_avg_frac_valid_hits", torch.mean(frac_valid), sync_dist=True)
                else:
                    valid_num_valid = num_valid[query_mask]
                    valid_frac_valid = frac_valid[query_mask]
                    if valid_num_valid.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_num_valid_hits", valid_num_valid.mean(), sync_dist=True)
                    if valid_frac_valid.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_frac_valid_hits", valid_frac_valid.mean(), sync_dist=True)

        # Just log predictions from the final layer
        preds = preds["final"]

        # First log metrics that depend on outputs from multiple tasks
        # TODO: Make the task names configurable or match task names automatically
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        if query_mask is not None:
            pred_valid = pred_valid & query_mask

        # Set the masks of any track slots that are not used as null
        pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets["particle_hit_valid"] & true_valid.unsqueeze(-1)

        # Calculate the true/false positive rates between the predicted and true masks
        # Number of hits that were correctly assigned to the track
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)

        # Number of predicted hits on the track
        hit_p = pred_hit_masks.sum(-1)

        # True number of hits on the track
        hit_t = true_hit_masks.sum(-1)

        # Calculate the efficiency and purity at differnt matching working points
        for wp in [0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid

            effs = (hit_tp / hit_t >= wp) & both_valid
            purs = (hit_tp / hit_p >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()

            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

        pred_num = pred_valid.sum(-1)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()), sync_dist=True)

        self.log(f"{stage}/num_tracks", torch.mean(pred_num.float()), sync_dist=True)
        true_num = true_valid.sum(-1)
        self.log(f"{stage}/num_particles", torch.mean(true_num.float()), sync_dist=True)

        num_hits_total = float(pred_hit_masks.shape[-1])
        num_hits_valid = float(true_hit_masks.sum())
        num_hits_noise = num_hits_total - num_hits_valid
        self.log(f"{stage}/num_hits", num_hits_total, sync_dist=True)
        self.log(f"{stage}/num_hits_valid", num_hits_valid, sync_dist=True)
        self.log(f"{stage}/num_hits_noise", num_hits_noise, sync_dist=True)

    def on_validation_epoch_start(self) -> None:
        self._reset_first_hit_stats("val")

    def on_validation_epoch_end(self) -> None:
        self._finalize_first_hit_stats("val")

    def on_test_epoch_start(self) -> None:
        self._reset_first_hit_stats("test")

    def on_test_epoch_end(self) -> None:
        self._finalize_first_hit_stats("test")

    def validation_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> dict[str, Tensor]:
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute losses then aggregate and log them
        outputs, targets, losses = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")
        self._update_first_hit_stats(inputs, preds, targets, stage="val")

        return {"loss": total_loss, "attn_mask_outputs": outputs}

    def test_step(
        self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """TrackML override: include post-loss targets for prediction writing.

        The post-loss targets are sorter-aligned when a sorter is configured, which keeps
        saved targets and saved predictions on the same hit ordering in eval files.
        """
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        preds = self.model.predict(outputs)
        self._update_first_hit_stats(inputs, preds, targets, stage="test")
        return outputs, preds, losses, targets


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLTracker,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
