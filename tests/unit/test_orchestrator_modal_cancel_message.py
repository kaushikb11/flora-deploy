from modal import exception as mx

from flora_deploy.modal_app.orchestrator_app import modal_errors, status_from_get

ERR = modal_errors()


def _raise(exc):
    def get():
        raise exc
    return get


def test_terminated_call_remote_error_is_cancelled():
    st = status_from_get("fc-1", _raise(mx.RemoteError(
        "Function call was cancelled by user or a failure.")), ERR)
    assert st.state == "cancelled"


def test_other_remote_errors_stay_failed():
    st = status_from_get("fc-1", _raise(mx.RemoteError("ValueError: boom")), ERR)
    assert st.state == "failed"
