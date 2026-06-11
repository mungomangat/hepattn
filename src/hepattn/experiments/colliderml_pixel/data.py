from __future__ import annotations

from typing import TYPE_CHECKING

import awkward as ak
import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader

from hepattn.experiments.colliderml.data import ColliderMLDataset

if TYPE_CHECKING:
    from pathlib import Path


class ColliderMLPixelDataset(ColliderMLDataset):
    """ColliderML dataset filtered to pixel detector hits for charged particle tracking.

    Inherits the parquet loading infrastructure from ColliderMLDataset and overrides
    load_event to: filter sihits to pixel detector IDs, optionally apply a trained hit
    filter from HDF5, apply a minimum pixel hit count cut, pad the particle dimension,
    and produce the dense particle_sihit_valid matrix and sihit_is_first target required
    by the MaskFormer tracking model.
    """

    def __init__(
        self,
        dirpath: str,
        num_events: int = -1,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        particle_min_num_hits: int = 3,
        event_max_num_particles: int = 1500,
        strict_max_objects: bool = False,
        pixel_detector_ids: list[int] | None = None,
        event_type: str = "ttbar",
        hit_eval_path: str | None = None,
        hit_filter_threshold: float = 0.1,
        debug: bool = False,
    ):
        super().__init__(
            dirpath=dirpath,
            num_events=num_events,
            particle_min_pt=particle_min_pt,
            particle_max_abs_eta=particle_max_abs_eta,
            particle_include_charged=True,
            particle_include_neutral=False,
            return_calohits=False,
            return_tracks=False,
            event_type=event_type,
            debug=debug,
        )
        self.pixel_detector_ids = pixel_detector_ids
        self.particle_min_num_hits = particle_min_num_hits
        self.event_max_num_particles = event_max_num_particles
        self.strict_max_objects = strict_max_objects
        self.hit_eval_path = hit_eval_path
        self.hit_filter_threshold = hit_filter_threshold

        if self.hit_eval_path:
            rank_zero_info(f"Using hit eval dataset {self.hit_eval_path}")

    def load_event(self, sample_id: int) -> tuple[dict, dict]:
        shard_idx, event_idx = self.sample_index[sample_id]
        particle_path = self.particle_shard_paths[shard_idx]
        sihit_path = self._get_collection_path("tracker_hits", particle_path)

        particles = self._read_event_from_file(particle_path, event_idx)
        sihits = self._read_event_from_file(sihit_path, event_idx)

        # Filter sihits to pixel detector subdetectors only
        if self.pixel_detector_ids is not None:
            detector_ids = np.asarray(ak.to_numpy(sihits["detector"]), dtype=np.int64)
            pixel_mask = np.isin(detector_ids, self.pixel_detector_ids)
            sihits = ak.zip({f: sihits[f][pixel_mask] for f in sihits.fields if isinstance(sihits[f], ak.Array)})

        # Apply trained hit filter predictions to reduce sihit count
        if self.hit_eval_path is not None:
            with h5py.File(self.hit_eval_path, "r") as f:
                key = f"{sample_id}/preds/final/sihit_filter/sihit_on_valid_particle_prob"
                assert key in f, f"Key {key} not found in {self.hit_eval_path}"
                filter_probs = f[key][0]  # shape (1, N_hits) → (N_hits,)
                filter_mask = filter_probs >= self.hit_filter_threshold
                sihits = ak.zip({f: sihits[f][filter_mask] for f in sihits.fields if isinstance(sihits[f], ak.Array)})

        # Build particle targets and apply kinematic cuts
        targets = self._build_particle_targets(particles)
        particle_valid = self._build_particle_kinematic_mask(targets)
        self._apply_mask_to_prefixed_keys(targets, "particle_", particle_valid)
        targets["particle_valid"] = self._valid_mask_like(targets["particle_pt"])

        # Build sihit inputs (scales xyz to metres, adds r/eta/phi)
        inputs = self._build_sihit_inputs(sihits)
        targets["sihit_valid"] = inputs["sihit_valid"]

        # Build particle–sihit CSR and apply minimum pixel hit count cut
        particle_sihit_indptr, particle_sihit_indices, _, particle_num_sihits = self._build_particle_sihit_fields(
            targets["particle_particle_id"],
            inputs["sihit_particle_id"],
        )

        hit_cut_mask = particle_num_sihits >= self.particle_min_num_hits
        self._apply_mask_to_prefixed_keys(targets, "particle_", hit_cut_mask)
        targets["particle_valid"] = self._valid_mask_like(targets["particle_pt"])

        # Rebuild CSR after hit cut
        particle_sihit_indptr, particle_sihit_indices, _, particle_num_sihits = self._build_particle_sihit_fields(
            targets["particle_particle_id"],
            inputs["sihit_particle_id"],
        )

        num_particles = int(targets["particle_particle_id"].shape[0])
        num_sihits = int(inputs["sihit_x"].shape[0])

        # Truncate to event_max_num_particles by highest pT if necessary
        if num_particles > self.event_max_num_particles:
            if self.strict_max_objects:
                msg = f"Event has {num_particles} particles, exceeding max {self.event_max_num_particles}"
                raise ValueError(msg)
            keep_mask = torch.zeros(num_particles, dtype=torch.bool)
            top_k = targets["particle_pt"].topk(self.event_max_num_particles).indices
            keep_mask[top_k] = True
            self._apply_mask_to_prefixed_keys(targets, "particle_", keep_mask)
            num_particles = self.event_max_num_particles
            # Rebuild CSR after truncation
            particle_sihit_indptr, particle_sihit_indices, _, particle_num_sihits = self._build_particle_sihit_fields(
                targets["particle_particle_id"],
                inputs["sihit_particle_id"],
            )

        # Compute sihit_is_first: for each particle, mark its innermost pixel hit (min r)
        indptr_np = particle_sihit_indptr.numpy()
        indices_np = particle_sihit_indices.numpy()
        r_np = inputs["sihit_r"].numpy()

        sihit_is_first = np.zeros(num_sihits, dtype=bool)
        for i in range(num_particles):
            s, e = int(indptr_np[i]), int(indptr_np[i + 1])
            if s >= e:
                continue
            hit_ids = indices_np[s:e]
            sihit_is_first[hit_ids[r_np[hit_ids].argmin()]] = True
        targets["sihit_is_first"] = torch.from_numpy(sihit_is_first)

        # Build dense particle_sihit_valid for the MaskFormer task [max_particles, num_sihits]
        particle_sihit_valid = np.zeros((self.event_max_num_particles, num_sihits), dtype=bool)
        for i in range(num_particles):
            s, e = int(indptr_np[i]), int(indptr_np[i + 1])
            if s < e:
                particle_sihit_valid[i, indices_np[s:e]] = True
        targets["particle_sihit_valid"] = torch.from_numpy(particle_sihit_valid)

        # Pad particle targets to event_max_num_particles
        num_padding = self.event_max_num_particles - num_particles
        targets["particle_valid"] = torch.cat([
            torch.ones(num_particles, dtype=torch.bool),
            torch.zeros(num_padding, dtype=torch.bool),
        ])

        skip_keys = {"particle_valid", "particle_sihit_valid"}
        for key in list(targets.keys()):
            if not key.startswith("particle_") or key in skip_keys:
                continue
            val = targets[key]
            if val.shape[0] != num_particles:
                continue
            pad_shape = (num_padding, *val.shape[1:])
            if val.dtype == torch.bool:
                pad_val = torch.zeros(pad_shape, dtype=torch.bool)
            elif val.dtype in (torch.int64, torch.int32):
                pad_val = torch.full(pad_shape, -999, dtype=val.dtype)
            else:
                pad_val = torch.full(pad_shape, float("nan"), dtype=val.dtype)
            targets[key] = torch.cat([val, pad_val])

        targets["sample_id"] = torch.tensor(sample_id, dtype=torch.int64)
        return inputs, targets

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        if idx < 0:
            idx += len(self.sample_ids)
        if idx < 0 or idx >= len(self.sample_ids):
            msg = f"Dataset index {idx} out of range for length {len(self.sample_ids)}"
            raise IndexError(msg)

        sample_id = self.sample_ids[idx]
        inputs, targets = self.load_event(sample_id)

        inputs_out = {k: v.unsqueeze(0) for k, v in inputs.items()}
        targets_out = {k: v.unsqueeze(0) for k, v in targets.items()}
        return inputs_out, targets_out


class ColliderMLPixelDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = True,
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        if stage in {"fit", "test"}:
            self.train_dset = ColliderMLPixelDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                hit_eval_path=self.hit_eval_train,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dset = ColliderMLPixelDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                hit_eval_path=self.hit_eval_val,
                **self.kwargs,
            )

        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test directory specified, see --data.test_dir"
            self.test_dset = ColliderMLPixelDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                hit_eval_path=self.hit_eval_test,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def _get_dataloader(self, dataset: ColliderMLPixelDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dset, shuffle=False)


class ColliderMLPixelFilterDataset(ColliderMLDataset):
    """ColliderML pixel dataset for training the hit filter.

    Produces per-sihit binary targets: sihit_on_valid_particle and sihit_is_first.
    No padding or dense matrices — the model processes variable-length sequences.
    """

    def __init__(
        self,
        dirpath: str,
        num_events: int = -1,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        particle_min_num_hits: int = 3,
        pixel_detector_ids: list[int] | None = None,
        event_type: str = "ttbar",
        debug: bool = False,
    ):
        super().__init__(
            dirpath=dirpath,
            num_events=num_events,
            particle_min_pt=particle_min_pt,
            particle_max_abs_eta=particle_max_abs_eta,
            particle_include_charged=True,
            particle_include_neutral=False,
            return_calohits=False,
            return_tracks=False,
            event_type=event_type,
            debug=debug,
        )
        self.pixel_detector_ids = pixel_detector_ids
        self.particle_min_num_hits = particle_min_num_hits

    def load_event(self, sample_id: int) -> tuple[dict, dict]:
        shard_idx, event_idx = self.sample_index[sample_id]
        particle_path = self.particle_shard_paths[shard_idx]
        sihit_path = self._get_collection_path("tracker_hits", particle_path)

        particles = self._read_event_from_file(particle_path, event_idx)
        sihits = self._read_event_from_file(sihit_path, event_idx)

        # Filter sihits to pixel detector subdetectors only
        if self.pixel_detector_ids is not None:
            detector_ids = np.asarray(ak.to_numpy(sihits["detector"]), dtype=np.int64)
            pixel_mask = np.isin(detector_ids, self.pixel_detector_ids)
            sihits = ak.zip({f: sihits[f][pixel_mask] for f in sihits.fields if isinstance(sihits[f], ak.Array)})

        # Build particle targets and apply kinematic cuts (charged only)
        targets = self._build_particle_targets(particles)
        particle_valid = self._build_particle_kinematic_mask(targets)
        self._apply_mask_to_prefixed_keys(targets, "particle_", particle_valid)
        targets["particle_valid"] = self._valid_mask_like(targets["particle_pt"])

        # Build sihit inputs (scales xyz to metres, adds r/eta/phi)
        inputs = self._build_sihit_inputs(sihits)
        targets["sihit_valid"] = inputs["sihit_valid"]

        # Build particle–sihit CSR and apply minimum pixel hit count cut
        particle_sihit_indptr, particle_sihit_indices, _, particle_num_sihits = self._build_particle_sihit_fields(
            targets["particle_particle_id"],
            inputs["sihit_particle_id"],
        )

        hit_cut_mask = particle_num_sihits >= self.particle_min_num_hits
        self._apply_mask_to_prefixed_keys(targets, "particle_", hit_cut_mask)
        targets["particle_valid"] = self._valid_mask_like(targets["particle_pt"])

        # Rebuild CSR after hit cut
        particle_sihit_indptr, particle_sihit_indices, _, _ = self._build_particle_sihit_fields(
            targets["particle_particle_id"],
            inputs["sihit_particle_id"],
        )

        num_particles = int(targets["particle_particle_id"].shape[0])
        num_sihits = int(inputs["sihit_x"].shape[0])

        indptr_np = particle_sihit_indptr.numpy()
        indices_np = particle_sihit_indices.numpy()
        r_np = inputs["sihit_r"].numpy()

        # sihit_on_valid_particle: True if this sihit is linked to any valid particle in the CSR
        sihit_on_valid_particle = np.zeros(num_sihits, dtype=bool)
        if indices_np.size > 0:
            sihit_on_valid_particle[indices_np] = True
        targets["sihit_on_valid_particle"] = torch.from_numpy(sihit_on_valid_particle)

        # sihit_is_first: innermost pixel hit per particle (min r)
        sihit_is_first = np.zeros(num_sihits, dtype=bool)
        for i in range(num_particles):
            s, e = int(indptr_np[i]), int(indptr_np[i + 1])
            if s >= e:
                continue
            hit_ids = indices_np[s:e]
            sihit_is_first[hit_ids[r_np[hit_ids].argmin()]] = True
        targets["sihit_is_first"] = torch.from_numpy(sihit_is_first)

        targets["sample_id"] = torch.tensor(sample_id, dtype=torch.int64)
        return inputs, targets

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        if idx < 0:
            idx += len(self.sample_ids)
        if idx < 0 or idx >= len(self.sample_ids):
            msg = f"Dataset index {idx} out of range for length {len(self.sample_ids)}"
            raise IndexError(msg)

        sample_id = self.sample_ids[idx]
        inputs, targets = self.load_event(sample_id)

        inputs_out = {k: v.unsqueeze(0) for k, v in inputs.items()}
        targets_out = {k: v.unsqueeze(0) for k, v in targets.items()}
        return inputs_out, targets_out


class ColliderMLPixelFilterDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        if stage in {"fit", "test"}:
            self.train_dset = ColliderMLPixelFilterDataset(dirpath=self.train_dir, num_events=self.num_train, **self.kwargs)

        if stage == "fit":
            self.val_dset = ColliderMLPixelFilterDataset(dirpath=self.val_dir, num_events=self.num_val, **self.kwargs)

        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test directory specified, see --data.test_dir"
            self.test_dset = ColliderMLPixelFilterDataset(dirpath=self.test_dir, num_events=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def _get_dataloader(self, dataset: ColliderMLPixelFilterDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dset, shuffle=False)
