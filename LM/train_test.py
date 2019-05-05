import argparse, sys, os, time, logging, threading, traceback, subprocess
import numpy as np
import tensorflow as tf
import _pickle as pkl
from .Config import Config
from .model import model
from .utils import utils
from .LMdataset import Dataset

_REVISION = 'lm-basic'

parser = argparse.ArgumentParser(description="training options")

parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
parser.add_argument('--restore-ckpt', action='store_true', dest='restore_ckpt', default=False)
parser.add_argument('--retain-gpu', action='store_true', dest='retain_gpu', default=False)

parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

args = parser.parse_args()

DEBUG = args.debug_enable
if not DEBUG:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def debug(s):
    if DEBUG:
        print(s)
    pass

class Train:

    def __init__(self, args):

        gpu_lock = threading.Lock()
        gpu_lock.acquire()
        def retain_gpu():
            if args.retain_gpu:
                with tf.Session():
                    gpu_lock.acquire()
            else:
                pass

        lockThread = threading.Thread(target=retain_gpu)
        lockThread.start()
        try:
            self.args = args
            config = Config()

            self.args = args
            self.weight_path = args.weight_path

            if args.load_config == False:
                config.saveConfig(self.weight_path + '/config')
                print('default configuration generated, please specify --load-config and run again.')
                gpu_lock.release()
                lockThread.join()
                sys.exit()
            else:
                if os.path.exists(self.weight_path + '/config'):
                    config.loadConfig(self.weight_path + '/config')
                else:
                    raise ValueError('No config file in %s' % self.weight_path)

            if config.revision != _REVISION:
                raise ValueError('revision dont match: %s over %s' % (config.revision, _REVISION))

            self.config = config

        except Exception as e:
            traceback.print_exc()
            gpu_lock.release()
            lockThread.join()
            exit()

        gpu_lock.release()
        lockThread.join()

    def train_run(self):
        _process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
        b_process = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], shell=False, stdout=subprocess.PIPE)
        _git_head_hash = _process.communicate()[0].strip()
        _git_head_branch = b_process.communicate()[0].strip()

        logging.info('Training start with code version: %s - %s'% (_git_head_hash, _git_head_branch))
        logging.info("Parameter count is: %d" % self.model.param_cnt)

        ds_fn = Dataset(filepath=self.config.filepath, wordpath=self.config.wordpath,
                        batch_size=self.config.batch_sz, epoch=self.config.max_epochs)
        my_model = model(self.config)

        est_model = tf.estimator.Estimator(model_fn=my_model, params={})
        est_model.train(input_fn=ds_fn)

        logging.info("Training complete")


    def main_run(self):

        if not os.path.exists(self.args.weight_path):
            os.makedirs(self.args.weight_path)
        logFile = self.args.weight_path + '/run.log'

        if self.args.train_test == "train":

            try:
                os.remove(logFile)
            except OSError:
                pass
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            debug('_main_run_')
            self.train_run()
        else:
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)

    @staticmethod
    def save_loss_accu(fileName, train_loss, valid_loss,
                       test_loss, valid_accu, test_accu, epoch):
        with open(fileName, 'a') as fd:
            fd.write('%3d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' %
                     (epoch, train_loss, valid_loss,
                      test_loss, valid_accu, test_accu))

    @staticmethod
    def remove_file(fileName):
        if os.path.exists(fileName):
            os.remove(fileName)

if __name__ == '__main__':
    trainer = Train(args)
    trainer.main_run()

