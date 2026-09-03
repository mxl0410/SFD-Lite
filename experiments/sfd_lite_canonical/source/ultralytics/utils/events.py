# AGPL-3.0 License - https://ultralytics.com/license

"""Telemetry compatibility shim for the public SFD-Lite release.

The upstream analytics client is intentionally disabled in this repository.
Keeping a no-op callable preserves the public events interface without
embedding analytics credentials or transmitting usage information.
"""


class Events:
    """Provide the upstream events interface with telemetry disabled."""

    enabled = False

    def __call__(self, cfg, device=None, backend=None) -> None:
        """Ignore analytics events in the public research-code release."""
        return None


events = Events()
