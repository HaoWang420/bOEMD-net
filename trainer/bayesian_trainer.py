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
        beta = metrics.get_beta(step, len(self.train_loader), self.beta_type, epoch, self.num_epoch)

        output, kl = self.model(image)
        # print("check for kl", output, kl)
        loss = self.criterion(output, target, kl, beta, self.train_length)
        
        return output, kl, loss
    
    # TODO
    # multi-sample evaluation

    def get_weight_SNR(self):
        weight_SNR_vec = []

        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):

                W_mu = module.W_mu.data
                W_p = module.W_rho.data
                sig_W = 1e-6 + F.softplus(W_p, beta=1, threshold=20)

                b_mu = module.bias_mu.data
                b_p = module.bias_rho.data
                sig_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)

                W_snr = (torch.abs(W_mu) / sig_W)
                b_snr = (torch.abs(b_mu) / sig_b)

                for weight_SNR in W_snr.cpu().view(-1):
                    weight_SNR_vec.append(weight_SNR)

                for weight_SNR in b_snr.cpu().view(-1):
                    weight_SNR_vec.append(weight_SNR)

        return np.array(weight_SNR_vec)

    def sample_weights(self, W_mu, b_mu, W_p, b_p):

        eps_W = W_mu.data.new(W_mu.size()).normal_()
        # sample parameters
        std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
        W = W_mu + 1 * std_w * eps_W

        if b_mu is not None:
            std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)
            eps_b = b_mu.data.new(b_mu.size()).normal_()
            b = b_mu + 1 * std_b * eps_b
        else:
            b = None

        return W, b

    def get_weight_KLD(self, Nsamples=20):
        weight_KLD_vec = []

        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):

                W_mu = module.W_mu.data
                W_p = module.W_rho.data

                b_mu = module.bias_mu.data
                b_p = module.bias_rho.data

                std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)

                KL_W = W_mu.new(W_mu.size()).zero_()
                KL_b = b_mu.new(b_mu.size()).zero_()
                for i in range(Nsamples):
                    W, b = self.sample_weights(W_mu=W_mu, b_mu=b_mu, W_p=W_p, b_p=b_p)
                    # Note that this will currently not work with slab and spike prior
                    KL_W += metrics.isotropic_gauss_loglike(W, W_mu, std_w, do_sum=False) - module.likelihood.loglike(W,
                                                                                                                      do_sum=False)
                    KL_b += metrics.isotropic_gauss_loglike(b, b_mu, std_b, do_sum=False) - module.likelihood.loglike(b,
                                                                                                                      do_sum=False)

                KL_W /= Nsamples
                KL_b /= Nsamples

                for weight_KLD in KL_W.cpu().view(-1):
                    weight_KLD_vec.append(weight_KLD)

                for weight_KLD in KL_b.cpu().view(-1):
                    weight_KLD_vec.append(weight_KLD)

        return np.array(weight_KLD_vec)