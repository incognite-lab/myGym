import cv2
import csv
import glob, imageio
import numpy as np
import os, sys, argparse
import torch
from myGym.vae.model import VAE
import matplotlib.pyplot as plt
import time, random
import myGym.vae.vis_helpers as h
import multiprocessing as mp
from datetime import datetime
from myGym.vae import sample
from configparser import ConfigParser
import pkg_resources
from myGym.vae.vis_helpers import load_checkpoint

SEED = 1111
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
SAVE_DIR = "./trained_models/vae_{}/".format(datetime.now().strftime("%m_%d_%H_%M_%S"))
os.makedirs(SAVE_DIR)

class MMVAETrainer:
    def __init__(self, model, train_loader, test_loader, offscreen, imsize, viz_fqc, beta):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.img_loss = []
        self.kld = []
        self.offscreen = offscreen
        self.imsize = imsize
        self.viz_fqx = viz_fqc
        self.beta = beta
        self.loss_meter = Logger()

    def train(self, epoch):
        self.model.train()

        for batch_idx, image in enumerate(self.train_loader):
            if epoch <  int(config.get('training params', 'annealing_epochs')):
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(int(config.get('training params', 'annealing_epochs')) * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0
            image = torch.autograd.Variable(image[0])

            # refresh the optimizer
            optimizer.zero_grad()

            # pass data through self.model
            recon_image, mu, logvar = self.model(image)
            image_loss, KLD = self.elbo_loss(recon_image, image, mu, logvar, annealing_factor=annealing_factor)

            if not self.offscreen:
                # process the images for visualisation
                gt_i = self.tensor2img(image[0])
                viz = np.hstack((gt_i, self.tensor2img(recon_image[0])))
                viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
                cv2.imshow("GT | Image ", viz/255)
                cv2.waitKey(1)

            # compute ELBO for each data combo
            loss_m = torch.mean(image_loss)
            kld_m = torch.mean(KLD)
            progress_d = {"Epoch": epoch, "Train Img Loss": loss_m.item(), "Train KLD":kld_m.item()}
            self.loss_meter.update_train(progress_d)

            # compute and take gradient step
            image_loss.cuda().backward()
            optimizer.step()

            if batch_idx % int(config.get('training params', 'log_interval')) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, batch_idx * len(image), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss_m, annealing_factor))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_m))

    def plot_save(self, what, title, xlabel, ylabel, filename):
        plt.plot(what)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(SAVE_DIR, filename))
        plt.clf()

    @torch.no_grad()
    def test(self, epoch):
        self.model.eval()
        i_loss = []
        kl_loss = []
        for batch_idx, image in enumerate(self.test_loader):
            with torch.no_grad():
                image = torch.autograd.Variable(image[0])
            recon_image, mu, logvar = self.model(image)
            image_loss, KLD = self.elbo_loss(recon_image, image, mu, logvar, annealing_factor=1)
            recons = self.tensor2img(recon_image)[0:9]
            i_loss.append(image_loss)
            kl_loss.append(KLD)
            if epoch % self.viz_fqx == 0:
                orig = self.tensor2img(image[0:9])
                o = orig[0]
                r = recons[0]
                for x in range(8):
                    r = np.hstack((r, recons[x+1]))
                    o = np.hstack((o, orig[x+1]))
                viz_r = np.vstack((o, r))
                cv2.imwrite(os.path.join(SAVE_DIR, "{}_image_reconstruction.png".format(epoch)), viz_r)
        image_loss = torch.stack(i_loss)
        KLD = torch.stack(kl_loss)
        loss_m = torch.mean(image_loss)
        kld_m = torch.mean(KLD)
        progress_d = {"Test Img Loss": loss_m.item(), "Test KLD": kld_m.item()}
        self.loss_meter.update(progress_d)
        print('====> Test Loss: {:.4f}'.format(loss_m.item()))
        return loss_m

    def tensor2img(self, img):
        img = img * 255
        if len(img.shape) > 3:
            img = img.reshape(img.shape[0], img.shape[2], img.shape[3], img.shape[1])
        else:
            img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
        img = np.asarray(img.cpu().detach())
        return img

    def elbo_loss(self, recon_image, image, mu, logvar, annealing_factor=1):
        """ELBO loss function."""
        image_loss = 0

        if not torch.isnan(recon_image).any():
            for i in range(recon_image.shape[0]):
                img_loss = torch.nn.functional.binary_cross_entropy(recon_image[i].cuda().view(-1, 3 * self.imsize * self.imsize),
                                               image[i].cuda().view(-1, 3 * self.imsize * self.imsize), reduction="sum")
                image_loss += img_loss
        else:
            print("NaN in image reconstruction")
            image_loss = 500000
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        KLD_L = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        KLD = torch.sum(KLD_L, dim=0)
        ELBO = torch.mean(image_loss + self.beta * annealing_factor * KLD)
        return ELBO, KLD

    def linear_annealing(self, init, fin, step, annealing_steps):
        """Linear annealing of a parameter."""
        if annealing_steps == 0:
            return fin
        assert fin > init
        delta = fin - init
        annealed = min(init + delta * step / annealing_steps, fin)
        return annealed


class Logger(object):
    """Saves training progress into csv"""
    def __init__(self):
        self.fields = ["Epoch", "Train Img Loss", "Train KLD", "Test Img Loss", "Test KLD"]
        self.reset()

    def reset(self):
        with open(os.path.join(SAVE_DIR, "loss.csv"), mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writeheader()

    def update_train(self, val_d):
        self.dic = val_d

    def update(self, val_d):
        with open(os.path.join(SAVE_DIR, "loss.csv"), mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writerow({**self.dic, **val_d})
        self.dic = {}

def load_images(path, size, imsize):
    print("Loading data...")
    types = ('*.png', '*.jpg', '*.jpeg')
    images = []
    for files in types:
        images.extend(glob.glob(os.path.join(path, files)))
    dataset = np.zeros((size, imsize, imsize, 3), dtype=np.float)
    for i, image_path in enumerate(sorted(images)):
            image = imageio.imread(image_path)
            # image = reshape_image(image, self.imsize)
            image = cv2.resize(image, (imsize, imsize))
            if i >= size:
                break
            dataset[i, :] = image/255
    print("Dataset of shape {} loaded".format(dataset.shape))
    return dataset

def load_images_all(path, imsize):
    print("Loading data...")
    types = ('*.png', '*.jpg', '*.jpeg')
    images = []
    for files in types:
        images.extend(glob.glob(os.path.join(path, files)))
    dataset = np.zeros((len(images), imsize, imsize, 3), dtype=np.float)
    for i, image_path in enumerate(images):
            image = imageio.imread(image_path)
            # image = reshape_image(image, self.imsize)
            image = cv2.resize(image, (imsize, imsize))
            dataset[i, :] = image/255
    print("Dataset of shape {} loaded".format(dataset.shape))
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--offscreen', action='store_true', default=False,
                        help='offscreen for remote training')
    parser.add_argument('--continue-train', type=str, default=False,
                        help='model to continue training')
    parser.add_argument('--config', type=str, default='config.ini',
                        help='vae config')
    args = parser.parse_args()
    config = ConfigParser()
    config.read(pkg_resources.resource_filename("myGym", '/vae/{}'.format(args.config)))

    with open(os.path.join(SAVE_DIR,'config.ini'), 'w') as f:
         config.write(f)

    n_epochs = int(config.get('training params', 'n_epochs'))
    use_cuda = (config.get('training params', 'use_cuda'))
    viz_frequency = int(config.get('training params', 'viz_every_n_epochs'))
    test_data_percent = float(config.get('training params', 'test_data_percentage'))
    n_latents = int(config.get('model params', 'n_latents'))
    batch_size = int(config.get('model params', 'batch_size'))
    lr = float(config.get('model params', 'lr'))
    imsize = int(config.get('model params', 'img_size'))
    beta = float(config.get('model params', 'beta'))
    use_cuda = True if use_cuda == "True" else False

    # load dataset
    images = load_images_all(str(config.get('training params', 'dataset_path')), imsize)

    # partition dataset
    data_size = images.shape[0]
    images_all = images.reshape((data_size, 3, imsize, imsize))
    images = torch.Tensor(images_all[:int(data_size*(1-test_data_percent))])
    images_test = torch.Tensor(images_all[int(data_size*(1-test_data_percent)):])

    # load data
    t_dataset = torch.utils.data.TensorDataset(images)
    v_dataset = torch.utils.data.TensorDataset(images_test)
    train_loader = torch.utils.data.DataLoader(t_dataset, batch_size=batch_size, shuffle=True, num_workers=mp.cpu_count())
    test_loader = torch.utils.data.DataLoader(v_dataset, batch_size=batch_size)
    N_mini_batches = len(train_loader)


    if not args.continue_train:
        model = VAE(n_latents, batch_size=batch_size, training=True, imsize=imsize, use_cuda=use_cuda)
    else:
        model = load_checkpoint(args.continue_train, use_cuda=use_cuda, training=True)
        print("Loaded model {}".format(args.continue_train))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_cuda:
        model.cuda()

    trainer = MMVAETrainer(model, train_loader, test_loader, offscreen=args.offscreen, imsize=imsize,
                           viz_fqc=viz_frequency, beta=beta)
    best_loss = sys.maxsize
    start = time.time()

    # train loop
    for epoch in range(1, n_epochs + 1):
        trainer.train(epoch)
        loss = trainer.test(epoch)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        # save the best model and current model
        h.save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': n_latents,
            'batch_size': batch_size,
            'optimizer': optimizer.state_dict(),
            'imsize': imsize
        }, is_best, folder=SAVE_DIR)

    end = time.time()
    print("Training duration: {} minutes".format((end - start)/60))
    sample.make_visualizations(SAVE_DIR, model)
