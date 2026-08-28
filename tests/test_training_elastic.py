import signal

from training.elastic import PreemptionCoordinator


def test_preemption_coordinator_tracks_and_restores_handlers():
    coordinator = PreemptionCoordinator()
    previous = signal.getsignal(signal.SIGTERM)
    coordinator.install()
    try:
        coordinator.request()
        assert coordinator.should_stop()
    finally:
        coordinator.restore()
    assert signal.getsignal(signal.SIGTERM) == previous
