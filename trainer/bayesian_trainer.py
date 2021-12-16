from trainer.trainer import *

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

        output, kl = self.model(image)

        loss = self.criterion(output, target, kl, beta, self.train_length)
        
        return output, kl.mean(), loss.mean()
    

    # multi-sample evaluation
    def val(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        for i, sample in enumerate(tbar):
            if self.args.dataset == 'lidc-syn-rand':
                image, target = sample['image'], sample['labels']
            else:
                image, target = sample['image'], sample['label']
            n, c, w, h = target.shape
            if self.args.cuda:
                image= image.cuda()
            # Keep the image shape as one
            assert image.shape[0] == 1
            if self.args.dataset == 'lidc-syn-rand':
                image = image.repeat(self.args.model.num_sample * 3, 1, 1, 1)
            else:
                image = image.repeat(self.args.model.num_sample, 1, 1, 1)

            with torch.no_grad():
                predictions= self.model(image)

            if self.args.dataset == 'lidc-syn-rand':
                predictions = predictions.reshape((self.num_sample, 3, predictions.shape[2], predictions.shape[3]))

            mean_out = torch.mean(predictions, dim=0, keepdim=True).cpu().numpy()
            target = target.data.cpu().numpy()
            # print("target shape", target.shape)
            self.evaluator.add_batch(target, mean_out)

        results = self.evaluator.compute()

        for metric in results:
            self.writer.add_scalar(metric, results[metric], epoch)

        for metric in results:
            print(f"{metric} {results[metric]}")

        is_best = True
        self.best_pred = results['qubiq']
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)
