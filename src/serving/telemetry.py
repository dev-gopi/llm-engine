"""Optional OpenTelemetry tracing hooks that fail closed only when enabled."""

from __future__ import annotations

from contextlib import contextmanager


def configure_opentelemetry(
    *, enabled: bool, service_name: str, endpoint: str | None = None
):
    if not enabled:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenTelemetry SDK is required when tracing is enabled"
        ) from exc
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OTLP exporter package is required for GOPI_OTEL_ENDPOINT"
            ) from exc
        exporter = OTLPSpanExporter(endpoint=endpoint)
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


@contextmanager
def maybe_span(tracer, name: str, **attributes):
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        yield span


__all__ = ["configure_opentelemetry", "maybe_span"]
