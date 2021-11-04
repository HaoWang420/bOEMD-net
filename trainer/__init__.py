from trainer.trainer import Trainer
from trainer.bayesian_trainer import BayesianTrainer

def build_trainer(args):
    if args.model.name in ['vae', 'boemd']:
        return BayesianTrainer(args)
    
    return Trainer(args)