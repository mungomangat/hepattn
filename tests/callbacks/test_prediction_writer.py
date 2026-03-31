import torch

from hepattn.callbacks.prediction_writer import PredictionWriter


class WriteSampleSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, sample_id, inputs, targets, outputs, preds, losses, idx):
        self.calls.append({
            "sample_id": sample_id,
            "inputs": inputs,
            "targets": targets,
            "outputs": outputs,
            "preds": preds,
            "losses": losses,
            "idx": idx,
        })


def _make_writer() -> PredictionWriter:
    return PredictionWriter(
        write_inputs=True,
        write_outputs=True,
        write_preds=True,
        write_targets=True,
        write_losses=True,
    )


def test_on_test_batch_end_uses_aligned_targets_when_returned() -> None:
    writer = _make_writer()
    spy = WriteSampleSpy()
    writer.write_sample = spy  # type: ignore[method-assign]

    inputs = {"hit_eta": torch.tensor([[0.1, 0.2]])}
    batch_targets = {
        "sample_id": torch.tensor([11]),
        "hit_is_first": torch.tensor([[False, True]]),
    }
    aligned_targets = {
        "sample_id": torch.tensor([22]),
        "hit_is_first": torch.tensor([[True, False]]),
    }
    outputs, preds, losses = {}, {}, {}

    writer.on_test_batch_end(
        trainer=None,
        pl_module=None,
        test_step_outputs=(outputs, preds, losses, aligned_targets),
        batch=(inputs, batch_targets),
        batch_idx=0,
    )

    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["sample_id"].item() == 22
    assert call["targets"] is aligned_targets


def test_on_test_batch_end_falls_back_to_batch_targets_for_legacy_output() -> None:
    writer = _make_writer()
    spy = WriteSampleSpy()
    writer.write_sample = spy  # type: ignore[method-assign]

    inputs = {"hit_eta": torch.tensor([[0.1, 0.2]])}
    batch_targets = {
        "sample_id": torch.tensor([11]),
        "hit_is_first": torch.tensor([[False, True]]),
    }
    outputs, preds, losses = {}, {}, {}

    writer.on_test_batch_end(
        trainer=None,
        pl_module=None,
        test_step_outputs=(outputs, preds, losses),
        batch=(inputs, batch_targets),
        batch_idx=0,
    )

    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["sample_id"].item() == 11
    assert call["targets"] is batch_targets
