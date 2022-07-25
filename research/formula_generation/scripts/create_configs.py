import yaml
from uuid import uuid4


def load_default_config():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def write_config(config):
    with open('configs/' + uuid4().hex + '.yaml', 'w') as f:
        f.write(yaml.dump(config))


# --------------------

config = load_default_config()

for shift in [[-10, 10], [-5, 5], [-2, 2], [-1, 1]]:
    config['augmentation']['shift'] = shift
    write_config(config)

# --------------------

config = load_default_config()

for add in [[-0.1, 0.1], [-0.05, 0.05], [-0.01, 0.01]]:
    config['augmentation']['add'] = add
    write_config(config)

# --------------------

config = load_default_config()

for noise in [0.1, 0.01, 0.001, 1e-4]:
    config['augmentation']['random_noise'] = noise
    write_config(config)

# --------------------

config = load_default_config()

for scale in [[0.85, 1.2], [0.7, 1.3], [0.5, 1.5], [0.9, 1.1], [0.95, 1.05]]:
    config['augmentation']['scale'] = scale
    write_config(config)


# --------------------

config = load_default_config()

for dropping_aug in [[0.05, 0.1], [0.05, 0.25], [0.05, 0.5], [0.01, 0.1], [0.01, 0.25], [0.01, 0.5]]:
    config['augmentation']['dropping_aug'] = dropping_aug
    write_config(config)

