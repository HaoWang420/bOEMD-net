from trainer.bayesian_trainer import BayesianTrainer

class OEODTrainer(BayesianTrainer):
    def predict_iter(self, image, target):
        n, c, w, h = image.shape
        nsamples = self.args.model.num_samples

        # N num_samples 1 H W
        image = image[:, None, ...].repeat(1, nsamples, 1, 1, 1)
        image = image.reshape([-1, c, h, w])

        pred = self.model(image)

        # N*num_sampels nclass H W
        pred = pred.reshape([n, nsamples, h, w])

        return pred