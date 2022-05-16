from trainer.trainer import Trainer
from trainer.bayesian_trainer import BayesianTrainer
from trainer.phiseg_trainer import PhiSegTrainer

def build_trainer(args):
    if args.model.name in ['vae', 'boemd']:
        return BayesianTrainer(args)
    elif args.model.name in ['phiseg', 'prob-unet']:
        return PhiSegTrainer(args)
    
    return Trainer(args)