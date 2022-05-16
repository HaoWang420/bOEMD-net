from trainer.trainer import *

class PhiSegTrainer(Trainer):

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
        # self.summary.visualize(self.writer, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss / i, epoch)
        self.writer.add_scalar("train/total_kl_loss_epoch", kl_loss / i, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss))
        print("KL: %.4f" % (kl_loss))
    
    def forward_iter(self, image, target, epoch, step):
        output, loss, kl = self.model.forward(image, target)
        # output = self.model.module.accumulate_output(output)
        output = None
        
        return output, kl.mean(), loss.mean()
    

    # multi-sample evaluation
    def val(self, epoch, test=False):
        self.model.eval()
        self.evaluator.reset()

        if test:
            tbar = tqdm(self.test_loader, desc='\r')
        else:
            tbar = tqdm(self.val_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['labels']

            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()

            with torch.no_grad():
                pred = self.predict_iter(image, target)

            target = target.data.cpu().numpy()
            pred = pred.data.cpu().numpy()

            self.evaluator.add_batch(target, pred)

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


    def predict_iter(self, image, target=None):
        n, c, w, h = image.shape
        nsamples = self.args.model.num_samples
        # N num_samples 1 H W
        image = image[:, None, ...].repeat(1, nsamples, 1, 1, 1)
        image = image.reshape([-1, c, h, w])

        pred_list = self.model.forward(image, None)
        pred = self.model.module.accumulate_output(pred_list)

        # N*num_sampels nclass H W
        pred = torch.argmax(pred, dim=1)
        pred = pred.reshape([n, nsamples, h, w])

        return pred
