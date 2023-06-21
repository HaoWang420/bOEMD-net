from trainer.trainer import Trainer
from trainer.bayesian_trainer import BayesianTrainer
from trainer.phiseg_trainer import PhiSegTrainer
from trainer.oeod_trainer import OEODTrainer

def build_trainer(args):
    if args.model.name in ['vae', 'boemd', 'boeod']:
        return BayesianTrainer(args)
    elif args.model.name in ['phiseg', 'prob-unet']:
        return PhiSegTrainer(args)
    elif args.model.name in ['bOEOD-unet']:
        return OEODTrainer(args)
    
    return Trainer(args)