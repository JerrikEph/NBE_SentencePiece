import configparser
import traceback
import json


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    """

    """General"""
    revision = 'lm-basic'
    filepath = './'
    wordpath = './'

    optimizer = 'adam'

    max_epochs = 50
    pre_trained = False
    batch_sz = 64
    partial_update_until_epoch = 0
    vocab_sz = 4135

    embed_size = 300
    hidden_size = 200

    lr = 0.0001
    decay_steps = 1000
    decay_rate = 0.9

    dropout = 0.2
    reg = 0.

    def __init__(self):
        self.attr_list = [i for i in list(Config.__dict__.keys()) if
                          not callable(getattr(self, i)) and not i.startswith("__")]

    def printall(self):
        for attr in self.attr_list:
            print(attr, getattr(self, attr), type(getattr(self, attr)))

    def saveConfig(self, filePath):

        cfg = configparser.ConfigParser()
        cfg['General'] = {}
        gen_sec = cfg['General']
        for attr in self.attr_list:
            try:
                gen_sec[attr] = json.dumps(getattr(self, attr))
            except Exception as e:
                traceback.print_exc()
                raise ValueError('something wrong in “%s” entry' % attr)

        with open(filePath, 'w') as fd:
            cfg.write(fd)

    def loadConfig(self, filePath):

        cfg = configparser.ConfigParser()
        cfg.read(filePath)
        gen_sec = cfg['General']
        for attr in self.attr_list:
            try:
                val = json.loads(gen_sec[attr])
                assert type(val) == type(getattr(self, attr)), \
                    'type not match, expect %s got %s' % \
                    (type(getattr(self, attr)), type(val))

                setattr(self, attr, val)
            except Exception as e:
                traceback.print_exc()
                raise ValueError('something wrong in “%s” entry' % attr)

        with open(filePath, 'w') as fd:
            cfg.write(fd)