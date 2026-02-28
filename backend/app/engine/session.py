"""Execution session manager: tracks active training runs and their controllers."""
from .training_control import TrainingController


class ExecutionSession:
    def __init__(self, execution_id: str, session_id: str):
        self.execution_id = execution_id
        self.session_id = session_id
        self.controller = TrainingController()


_sessions: dict[str, ExecutionSession] = {}


def create_session(execution_id: str, session_id: str) -> ExecutionSession:
    session = ExecutionSession(execution_id, session_id)
    _sessions[execution_id] = session
    return session


def get_session(execution_id: str) -> ExecutionSession | None:
    return _sessions.get(execution_id)


def remove_session(execution_id: str) -> None:
    _sessions.pop(execution_id, None)
