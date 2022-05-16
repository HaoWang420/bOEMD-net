from trainer.trainer import *
from utils import metrics

class BayesianTrainer(Trainer):

    def training(self, epoch):
        train_loss = 0.0
        kl_loss = 0.0

        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            output, kl, loss = self.forward_iter(image, target, epoch, i)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            kl_loss += kl.item()

            tbar.set_description('Train loss: %.4f' % (train_loss / (i + 1)))
            # tbar.set_description("Train kl loss: %.4f" % (kl_loss / (i + 1)))

            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar("train/total_kl_loss_iter", kl.item(), i + num_img_tr * epoch)

        # Show 10 * 3 inference results each epoch
        global_step = i + num_img_tr * epoch
        self.summary.visualize(self.writer, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss / i, epoch)
        self.writer.add_scalar("train/total_kl_loss_epoch", kl_loss / i, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss))
        print("KL: %.4f" % (kl_loss))
    
    def forward_iter(self, image, target, epoch, step):
        kl = 0.
        beta = metrics.get_beta(step, len(self.train_loader), self.args.loss.beta_type, epoch, self.args.epochs)
        # print("beta, ", beta)
        output, kl = self.model(image)

        loss = self.criterion(output, target, kl, beta, self.train_length)
        
        return output, kl.mean(), loss.mean()

    def predict_iter(self, image, target):
        n, c, w, h = image.shape
        nsamples = self.args.model.num_samples

        image = image[:, None, ...].repeat(1, nsamples, 1, 1, 1)
        image = image.reshape([-1, c, h, w])

        pred = self.model(image)

        pred = pred.reshape([n, nsamples, -1, h, w])
        pred = torch.mean(pred, dim=1)

        return pred