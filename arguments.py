
import torch.nn.functional as F
import torch
import json
import time

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)

class Arguments:

    def __init__(self, logger):
        self.logger = logger
        self.debugOrRun = 'results'
        self.dataset = 'cifar100' # 'cifar100', 'lfw', 'mnist', 'celebA', 'stl10'
        self.set_imidx = 10085 # int or 000000
        self.net = 'resnet20-4' # 'lenet', 'resnet20-4', 'resnet34', 'vgg11'
        self.net_mt_diff = True
        self.batch_size = 1
        self.model_path = './model'
        self.root_path = '.'

        self.inv_loss = 'sim'

        self.lr = 0.1 # 0.1 for Adam, 1 for LBFGS
        self.optim = 'Adam'

        # self.lr = 1  # 0.1 for Adam, 1 for LBFGS
        # self.optim = 'LBFGS'

        self.iteration = 1501
        self.scheduler = False

        self.use_game = True
        self.earlystop = 1e-9
        self.save_final_img = False

        self.num_dummy = 1  # batch size
        self.num_exp = 1

        self.defense_method = 'none'  # 'none', 'soteria', 'noise', 'clipping', 'sparsification'
        self.noise_std = 0.0001 # for noise defense
        self.max_grad_norm_clipping = 4.0 # for clip defense
        self.sparsification_defense_sparsity = 90 # for sparsification_defense

        self.methods = ['mDLG']  # mDLG, mDLG_mt
        self.diff_task_agg = 'random' # 'single', 'random', 'game'
        self.num_servers = 2
        self.int_time = int(time.time())
        self.log_interval = 5
        self.log_metrics_interval = 100
        self.tv = 1e-2
        self.eval_metrics = ['mse'] # ['mse', 'lpips', 'psnr', 'ssim']

        self.train_data_loader_pickle_path = "data_loaders/cifar100/train_data_loader.pickle"
        self.test_data_loader_pickle_path = "data_loaders/cifar100/test_data_loader.pickle"


    def get_logger(self):
        return self.logger

    def get_dataset(self):
        return self.dataset

    def get_eval_metrics(self):
        return self.eval_metrics

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path

    def get_num_dummy(self):
        return self.num_dummy

    def set_test_data_loader_pickle_path(self, path):
        self.test_data_loader_pickle_path = path
    def get_default_model_folder_path(self):
        return self.model_path

    def get_root_path(self):
        return self.root_path

    def get_debugOrRun(self):
        return self.debugOrRun

    def get_lr(self):
        return self.lr

    def get_earlystop(self):
        return self.earlystop

    def get_iteration(self):
        return self.iteration

    def get_num_exp(self):
        return self.num_exp

    def get_methods(self):
        return self.methods

    def get_start_index_str(self):
        return self.start_index_str

    def get_log_interval(self):
        return self.log_interval

    def get_net(self):
        return self.net

    def get_net_mt_diff(self):
        return self.net_mt_diff

    def get_imidx(self):
        return self.set_imidx

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Iteration: {}\n".format(self.iteration) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Model Path (Relative): {}\n".format(self.model_path) + \
               "Methods: {}\n".format(self.methods) + \
               "Number Exp: {}\n".format(self.num_exp) + \
               "Dataset: {}\n".format(self.dataset) + \
               "Number Server: {}\n".format(self.num_servers) + \
               "Set Imidx: {}\n".format(self.set_imidx) + \
               "Batch Size: {}\n".format(self.num_dummy) + \
               "Log Interval: {}\n".format(self.log_interval)

