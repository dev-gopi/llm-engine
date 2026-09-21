"""Suite-wide test execution invariants."""


def pytest_collection_modifyitems(items):
    """Run synchronous ASGI TestClient cases before async event-loop cases.

    Starlette's synchronous client owns an AnyIO portal and cannot safely be
    initialized after pytest-asyncio has closed the main event loop.  The test
    cases are independent, so this ordering keeps both integration styles
    deterministic without weakening their assertions.
    """
    items.sort(key=lambda item: item.get_closest_marker("asyncio") is not None)
