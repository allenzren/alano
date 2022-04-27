import yaml

# usage:
# configDict = load_config(filePath)
# config_env = configDict['environment']
# config_training = configDict['training']
# config_arch_performance = configDict['arch_performance']
# config_arch_backup = configDict['arch_backup']
# config_update_performance = configDict['update_performance']
# config_update_backup = configDict['update_backup']

# objects = [ config_env, config_training,
#             config_arch_performance, config_arch_backup,
#             config_update_performance, config_update_backup]
# dump_config(filePath, objects)


class Struct:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def dict2object(dictionary, key):
    return Struct(**dictionary[key])


def load_config(filePath):
    with open(filePath) as f:
        data = yaml.safe_load(f)
    configDict = {}
    for key, value in data.items():
        configDict[key] = Struct(**value)
    return configDict


def dump_config(filePath, objects, keys):
    data = {}
    for key, object in zip(keys, objects):
        data[key] = object.__dict__
    with open(filePath, "w") as f:
        yaml.dump(data, f)
