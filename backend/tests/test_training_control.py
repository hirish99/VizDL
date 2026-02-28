"""Tests for training pause/resume/stop and checkpoint functionality."""
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.engine.training_control import TrainingController, TrainingState
from app.engine.checkpoint import save_checkpoint, load_checkpoint
from app.nodes.training import TrainingLoopNode


class TestTrainingControllerStates:
    def test_initial_state_running(self):
        ctrl = TrainingController()
        assert ctrl.state == TrainingState.RUNNING

    def test_pause(self):
        ctrl = TrainingController()
        ctrl.pause()
        assert ctrl.state == TrainingState.PAUSED

    def test_resume(self):
        ctrl = TrainingController()
        ctrl.pause()
        ctrl.resume()
        assert ctrl.state == TrainingState.RUNNING

    def test_stop(self):
        ctrl = TrainingController()
        ctrl.stop()
        assert ctrl.state == TrainingState.STOPPED

    def test_stop_from_paused(self):
        ctrl = TrainingController()
        ctrl.pause()
        ctrl.stop()
        assert ctrl.state == TrainingState.STOPPED

    def test_pause_only_from_running(self):
        ctrl = TrainingController()
        ctrl.stop()
        ctrl.pause()  # should not change state from STOPPED
        assert ctrl.state == TrainingState.STOPPED

    def test_resume_only_from_paused(self):
        ctrl = TrainingController()
        ctrl.resume()  # already running, should stay
        assert ctrl.state == TrainingState.RUNNING

    def test_check_returns_running(self):
        ctrl = TrainingController()
        assert ctrl.check() == TrainingState.RUNNING

    def test_check_returns_stopped(self):
        ctrl = TrainingController()
        ctrl.stop()
        assert ctrl.check() == TrainingState.STOPPED


class TestTrainingControllerBlocking:
    def test_check_blocks_when_paused(self):
        ctrl = TrainingController()
        ctrl.pause()

        result = [None]

        def worker():
            result[0] = ctrl.check()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=0.2)
        assert t.is_alive(), "check() should block when paused"

        ctrl.resume()
        t.join(timeout=1.0)
        assert not t.is_alive()
        assert result[0] == TrainingState.RUNNING

    def test_stop_unblocks_paused_thread(self):
        ctrl = TrainingController()
        ctrl.pause()

        result = [None]

        def worker():
            result[0] = ctrl.check()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=0.2)
        assert t.is_alive()

        ctrl.stop()
        t.join(timeout=1.0)
        assert not t.is_alive()
        assert result[0] == TrainingState.STOPPED


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = {"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.6], "epoch": [1, 2]}
        path = tmp_path / "checkpoint.pt"

        save_checkpoint(path, model, optimizer, 2, history)
        assert path.exists()

        model2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        epoch, loaded_history = load_checkpoint(path, model2, optimizer2)

        assert epoch == 2
        assert loaded_history == history
        # Model weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1.cpu(), p2.cpu())

    def test_creates_parent_dirs(self, tmp_path):
        model = nn.Sequential(nn.Linear(4, 1))
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "nested" / "dir" / "checkpoint.pt"

        save_checkpoint(path, model, optimizer, 0, {"train_loss": []})
        assert path.exists()


class TestTrainingLoopWithController:
    def _make_loaders(self):
        torch.manual_seed(42)
        X = torch.randn(100, 4)
        y = torch.randn(100, 1)
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=32), None

    def _make_model_and_optimizer(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, optimizer

    def test_normal_training_no_controller(self):
        """Training works without a controller (backward compatible)."""
        model, optimizer = self._make_model_and_optimizer()
        train_loader, val_loader = self._make_loaders()
        node = TrainingLoopNode()
        result = node.execute(
            model=model, optimizer=optimizer, loss_fn=nn.MSELoss(),
            train_loader=train_loader, val_loader=val_loader, epochs=3,
        )[0]
        assert len(result["history"]["train_loss"]) == 3
        assert not result["stopped_early"]

    def test_stop_after_first_epoch(self):
        """Controller stops training after 1 epoch."""
        model, optimizer = self._make_model_and_optimizer()
        train_loader, val_loader = self._make_loaders()
        ctrl = TrainingController()

        progress = []

        def on_progress(msg):
            progress.append(msg)
            if msg.get("type") == "training_progress" and msg.get("epoch") == 1:
                ctrl.stop()

        node = TrainingLoopNode()
        result = node.execute(
            model=model, optimizer=optimizer, loss_fn=nn.MSELoss(),
            train_loader=train_loader, epochs=5,
            progress_callback=on_progress,
            training_controller=ctrl,
        )[0]

        assert result["stopped_early"]
        assert len(result["history"]["train_loss"]) == 1

    def test_pause_and_resume(self):
        """Controller pauses then resumes training."""
        model, optimizer = self._make_model_and_optimizer()
        train_loader, _ = self._make_loaders()
        ctrl = TrainingController()

        epochs_seen = []

        def on_progress(msg):
            if msg.get("type") == "training_progress":
                epochs_seen.append(msg["epoch"])
                if msg["epoch"] == 2:
                    ctrl.pause()

        # Run training in a thread since pause will block
        result_holder = [None]

        def run_training():
            node = TrainingLoopNode()
            result_holder[0] = node.execute(
                model=model, optimizer=optimizer, loss_fn=nn.MSELoss(),
                train_loader=train_loader, epochs=5,
                progress_callback=on_progress,
                training_controller=ctrl,
            )[0]

        t = threading.Thread(target=run_training)
        t.start()

        # Wait for pause
        time.sleep(1.0)
        assert ctrl.state == TrainingState.PAUSED

        # Resume
        ctrl.resume()
        t.join(timeout=10.0)
        assert not t.is_alive()

        result = result_holder[0]
        assert result is not None
        assert len(result["history"]["train_loss"]) == 5
        assert not result["stopped_early"]

    def test_checkpoint_on_stop(self, tmp_path):
        """Checkpoint is saved when training is stopped."""
        model, optimizer = self._make_model_and_optimizer()
        train_loader, _ = self._make_loaders()
        ctrl = TrainingController()
        ckpt_path = tmp_path / "checkpoint.pt"

        def on_progress(msg):
            if msg.get("type") == "training_progress" and msg.get("epoch") == 2:
                ctrl.stop()

        node = TrainingLoopNode()
        result = node.execute(
            model=model, optimizer=optimizer, loss_fn=nn.MSELoss(),
            train_loader=train_loader, epochs=10,
            progress_callback=on_progress,
            training_controller=ctrl,
            checkpoint_path=ckpt_path,
        )[0]

        assert result["stopped_early"]
        assert ckpt_path.exists()

    def test_resume_from_checkpoint(self, tmp_path):
        """Training resumes from a checkpoint at the correct epoch."""
        model, optimizer = self._make_model_and_optimizer()
        train_loader, _ = self._make_loaders()
        ckpt_path = tmp_path / "checkpoint.pt"

        # Save a checkpoint at epoch 3
        history = {"train_loss": [1.0, 0.8, 0.6], "val_loss": [None, None, None], "epoch": [1, 2, 3]}
        save_checkpoint(ckpt_path, model, optimizer, 3, history)

        node = TrainingLoopNode()
        result = node.execute(
            model=model, optimizer=optimizer, loss_fn=nn.MSELoss(),
            train_loader=train_loader, epochs=5,
            checkpoint_path=ckpt_path,
        )[0]

        # Should have 5 total epochs (3 from checkpoint + 2 new)
        assert len(result["history"]["train_loss"]) == 5
        assert result["history"]["epoch"] == [1, 2, 3, 4, 5]
