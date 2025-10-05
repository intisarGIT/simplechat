import importlib


def test_import_app():
    mod = importlib.import_module('app')
    assert hasattr(mod, 'build_ui')


if __name__ == '__main__':
    test_import_app()
    print('import test passed')
