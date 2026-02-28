"""Thread-safe training control for pause/resume/stop."""
import threading
from enum import Enum


class TrainingState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class TrainingController:
    """Thread-safe signal for controlling a training loop from another thread."""

    def __init__(self):
        self._state = TrainingState.RUNNING
        self._lock = threading.Lock()
        self._resume_event = threading.Event()
        self._resume_event.set()  # starts unblocked (running)

    @property
    def state(self) -> TrainingState:
        with self._lock:
            return self._state

    def pause(self):
        with self._lock:
            if self._state == TrainingState.RUNNING:
                self._state = TrainingState.PAUSED
                self._resume_event.clear()

    def resume(self):
        with self._lock:
            if self._state == TrainingState.PAUSED:
                self._state = TrainingState.RUNNING
                self._resume_event.set()

    def stop(self):
        with self._lock:
            self._state = TrainingState.STOPPED
            self._resume_event.set()  # unblock if paused so thread can exit

    def check(self) -> TrainingState:
        """Call at the top of each epoch. Blocks while paused, returns state."""
        self._resume_event.wait()
        return self.state
