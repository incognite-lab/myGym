from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import torch, random
import cv2
from myGym.vae.vis_helpers import load_checkpoint
from myGym.vae.visualize import Visualizer
import os, glob, imageio


def fetch_sample_images(path, n, imsize):
    images = glob.glob(os.path.join(path, "*.png"))
    dataset = np.zeros((len(images), imsize, imsize, 3), dtype=np.float)
    samples = np.zeros((n, imsize, imsize, 3))
    for i, image_path in enumerate(images):
        image = imageio.imread(image_path)
        # image = reshape_image(image, self.imsize)
        image = cv2.resize(image, (imsize, imsize))
        dataset[i, :] = image
    print("Dataset of shape {} loaded".format(dataset.shape))
    for x in range(n):
        samples[x] = dataset[np.random.choice(np.arange(dataset.shape[0]))]
    return samples


def make_visualizations(savedir, model, n_per_latent=15):
    model.eval()
    viz = Visualizer(model, savedir, vocab=vocab)
    viz.traversals(data=None, n_per_latent=n_per_latent, n_latents=model.n_latents, is_reorder_latents=False)


def encode_images(model, images):
    imgs_input = []
    for img in images:
        im = torch.tensor(img).type(torch.FloatTensor)
        im = im.reshape(img.shape[2], img.shape[0], img.shape[0]).unsqueeze(0) / 255
        imgs_input = torch.cat((imgs_input, im), dim=0) if torch.is_tensor(imgs_input) else im
    latent_z = model.infer(image=imgs_input)[0]
    return latent_z


def decode_images(model, latent_z):
    decoded = model.image_decoder(latent_z)
    dec_img = process_decoded(model, decoded)
    return dec_img


def process_decoded(model, decoded):
    dec_img = []
    for im in decoded:
        img = im.reshape(model.imsize, model.imsize, 3)
        img = np.asarray((img * 255).detach().cpu(), dtype="uint8")
        dec_img.append(img)
    return dec_img

def list_of_dists(source, target):
    dists = []
    for t in target:
        dists.append(np.linalg.norm(np.asarray(source) - np.asarray(t)))
    return dists

def test_latent_linearity(model, length):
    model.eval()
    samples = []
    for x in range(length):
       samples.append(torch.tensor([random.uniform(-1, 1) for _ in range(model.n_latents)]))
    ordered_samples = samples[0].unsqueeze(0)
    cut_samples = samples[1:]
    dists = list_of_dists(samples[0], cut_samples)
    order = sorted(range(len(dists)), key=lambda k: dists[k])
    for x in order:
        ordered_samples = torch.cat((ordered_samples, cut_samples[x].unsqueeze(0)))
    decoded = decode_images(model, ordered_samples.cuda())
    dec_img = []
    dists = [0] + dists
    for ix, im in enumerate(decoded):
        img = im.reshape(model.imsize, model.imsize, 3)
        img = cv2.putText(img, str(round(sorted(dists)[ix],2)), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, 0)
        img = cv2.copyMakeBorder(img, 4, 0, 1, 4, cv2.BORDER_CONSTANT, value=[0,0,0])
        dec_img.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    image = np.hstack(([d for d in dec_img]))
    return image

def make_linear_grid(savedir, model, size=[10,15]):
    images = []
    for x in range(size[0]):
        images.append(test_latent_linearity(model, size[1]))
    image = np.vstack(([d for d in images]))
    filename = os.path.join(savedir, "linearity_test.png")
    cv2.imwrite(filename, image)
    print("Image saved into {}".format(filename))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/gabi/myGym/myGym/trained_models/vae_sidearm_beta10/model_best.pth.tar', help='path to trained model file')
    parser.add_argument('--n-samples', type=int, default=64, 
                        help='Number of images and texts to sample [default: 64]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    model, vocab = load_checkpoint(args.model_path, use_cuda=args.cuda)
    if args.cuda:
        model.cuda()
    make_linear_grid(os.path.dirname(args.model_path), model)
    make_visualizations(os.path.dirname(args.model_path), model)

