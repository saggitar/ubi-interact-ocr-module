import pytest

pytest_plugins = ['ubii.node.pytest']


@pytest.fixture(scope='session', autouse=True)
def not_always_verbose():
    import ubii.node.pytest
    old_val = ubii.node.pytest.ALWAYS_VERBOSE
    ubii.node.pytest.ALWAYS_VERBOSE = False
    yield
    ubii.node.pytest.ALWAYS_VERBOSE = old_val
