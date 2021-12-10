import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, DictConfig

from trainer import build_trainer

@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = build_trainer(args)

    if args.mode == 'train':
        train(args, trainer)
    elif args.mode == 'eval':
        evaluate(trainer)

    trainer.writer.close()

def train(args, trainer):
    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.val(epoch)

def evaluate(trainer):
    trainer.val(0)

if __name__ == "__main__":
    main()
