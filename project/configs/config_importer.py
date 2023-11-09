import importlib.util

def import_config(filename):
    if filename is None: return None

    # Remove the '.py' extension from the filename
    module_name = filename.replace('.py', '')

    # Create a module spec from the filename
    spec = importlib.util.spec_from_file_location(module_name, filename)

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    config = getattr(module, 'run_config', None)
    if config is not None: return config

    config = getattr(module, 'sweep_config', None)
    if config is not None: return config

    raise ValueError('No valid config found.')