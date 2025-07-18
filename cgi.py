from loguru import logger
import arguments
import os
from utils.methods import mdlg, mdlg_mt
import torch
from utils import files
from utils.data_download import load_data
from utils.net_utils import intialize_nets
from utils.save import save_results


def run():
    args = arguments.Arguments(logger)

    # Initialize logger
    log_files, str_time = files.files(args)
    handler = logger.add(log_files[0], enqueue=True)

    dataset = args.get_dataset()
    root_path = args.get_root_path()
    data_path = os.path.join(root_path, 'data').replace('\\', '/')
    save_path = os.path.join(root_path, args.get_debugOrRun()+'/compare_%s' % dataset).replace('\\', '/')
    # eval_res_path = os.path.join(root_path, '/compare_%s' % dataset).replace('\\', '/')
    lr = args.get_lr()
    num_dummy = args.get_num_dummy()
    Iteration = args.get_iteration()
    num_exp = args.get_num_exp()
    methods = args.get_methods()
    log_interval = args.get_log_interval()
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    args.log()
    args.logger.info('dataset is #{}:', dataset)
    args.logger.info('lr is #{}', lr)
    args.logger.info('log interval is #{}', log_interval)
    args.logger.info('root path is #{}:', root_path)
    args.logger.info('data_path is #{}:', data_path)
    args.logger.info('save_path is #{}:', save_path)

    tt, tp, num_classes, alter_num_classes, channel, hidden, dst, input_size, idx_shuffle, mean_std = load_data(dataset = dataset, root_path = root_path, data_path = data_path, save_path = save_path)

    ''' train DLG and iDLG and mDLG and DLGAdam'''
    for idx_net in range(num_exp):

        args.logger.info('running #{}|#{} experiment', idx_net, num_exp)

        '''train on different methods'''
        for method in methods: #
            args.logger.info('#{}, Try to generate #{} images', method, num_dummy)

            if method == 'mDLG':
                nets = intialize_nets(method = method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                            alter_num_classes=alter_num_classes, input_size=input_size)
                for net in nets:
                    net = net.to(device)
                imidx_list, final_iter, final_img ,results = mdlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time)
                save_results(results, root_path + '/' + method + '_' + str(imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(args.num_servers) + '_' + str_time + '_' + args.defense_method + '.csv', args)
            if method == 'mDLG_mt':
                nets = intialize_nets(method = method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                            alter_num_classes=alter_num_classes, input_size=input_size)
                for i in range(len(nets)):
                    nets[i] = nets[i].to(device)
                    args.logger.info('Size of net #{} is #{}',i, len(nets[i].state_dict()))
                imidx_list, final_iter, final_img, results = mdlg_mt(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time)
                method = method + '_' + args.diff_task_agg
                save_results(results, root_path + '/' + method + '_' + str(imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(args.num_servers) + '_' + str_time + '_' + args.defense_method + '.csv', args)

        args.logger.info('imidx_list: #{}', imidx_list)
    logger.remove(handler)


if __name__ == '__main__':
    run()

