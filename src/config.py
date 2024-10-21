import argparse
import argparse  
import time
import os
import logging
import torch


parser = argparse.ArgumentParser()
  
parser.add_argument('--mode',default='default',help="default / train / eval")  

parser.add_argument('--workers', type=int, default=4,  
                    help='number of data loading workers, you had better put it '  
                          '4 times of your gpu')  

parser.add_argument('--batch_size', type=int, default=40, help='input batch size, default=40')  
parser.add_argument('--pretrained', type=str, default='hfl/chinese-bert-wwm', help='pretrained model path')  

parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for, default=20')  

parser.add_argument('--lr', type=float, default=3e-5, help='select the learning rate, default=3e-5')  

parser.add_argument('--seed', type=int, default=118, help="random seed")  

parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(), help='enables cuda')  
parser.add_argument('--checkpoint_path',type=str,default='',  
                    help='Path to load a previous trained model if not empty (default empty)')  
# parser.add_argument('--output',action='store_true',default=True,help="shows output")  
parser.add_argument('--log',action='store_true',default=False,help="save log")  
parser.add_argument('--debug',action='store_true',default=False,help="debug mode")  
parser.add_argument('--out_dir',default=os.path.realpath('out'),help="output path")  
parser.add_argument('--save_freq',type=int ,default=0,help="save frequence of model, default is save per 2 epochs")  
parser.add_argument('--id',default=time.strftime("%d%H%M%S"),help="id of train")  
config = parser.parse_args()
out_dir = os.path.join(config.out_dir,f'{config.id}-{config.mode}')

log_level = logging.DEBUG if config.debug else logging.INFO

if(config.log or config.save_freq):
    if(not os.path.isdir(out_dir)): os.mkdir(out_dir)

    logging.basicConfig(level=log_level,
                        filename=os.path.join(out_dir, 'log'),
                        format="%(asctime)s %(filename)s [%(levelname)s] %(message)s",
                        datefmt = '%y-%m-%d %H:%M:%S'
                        )

    logger = logging.getLogger()
    console = logging.StreamHandler()  
    console.setLevel(log_level)
    logger.addHandler(console)
else:
    logging.basicConfig(level=log_level,
                        format="[%(asctime)s] %(message)s",
                        datefmt = '%H:%M:%S'
                        )

# if opt.output:
if True:
    logging.info(f'id: {config.id}')  
    logging.info(f'save log: {config.log}')  
    logging.info(f'mode: {config.mode}')  
    logging.info(f'pretrained model: {config.pretrained}')  
    logging.info(f'num_workers: {config.workers}')  
    logging.info(f'batch_size: {config.batch_size}')  
    logging.info(f'epochs (niters) : {config.niter}')  
    logging.info(f'learning rate : {config.lr}')  
    logging.info(f'manual_seed: {config.seed}')  
    logging.info(f'cuda enable: {config.cuda}')  
    logging.info(f'checkpoint_path: {config.checkpoint_path}')  
