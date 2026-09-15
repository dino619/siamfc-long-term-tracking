from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
try:
    from got10k.trackers import Tracker
except ImportError:
    # V nalogi uporabljamo samo osnovni vmesnik razreda Tracker.
    # Ta nadomestni razred omogoča zagon tudi brez paketa got10k.
    class Tracker(object):
        def __init__(self, name, is_deterministic=True):
            self.name = name
            self.is_deterministic = is_deterministic

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms


__all__ = ['TrackerSiamFC', 'TrackerSiamFCLongTerm']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # Parametri za dolgotrajno sledenje. Privzete vrednosti so izbrane
            # tako, da je zagon na CPU še izvedljiv, poskusi pa jih lahko
            # prepišejo iz ukazne vrstice.
            'redetect_threshold': 4.0,
            'redetect_samples': 96,
            'redetect_batch_size': 32,
            'redetect_sampling': 'uniform',
            'redetect_sigma': 80.0,
            'redetect_sigma_growth': 1.15,
            'redetect_uniform_ratio': 0.25,
            'redetect_seed': 0,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 16,  # 32
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)
    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        max_resp = max(0, response.max())
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box, max_resp
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels


class TrackerSiamFCLongTerm(TrackerSiamFC):
    # Dolgotrajni sledilnik namerno deduje osnovni SiamFC. Tako ostane
    # kratkotrajna logika nespremenjena, dodamo pa samo zaznavo izgube in
    # ponovno detekcijo.

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFCLongTerm, self).__init__(net_path=net_path, **kwargs)
        # Generator z znanim semenom omogoča ponovljive poskuse. To je pomembno,
        # ker naključno vzorčenje lahko spremeni rezultat med zagoni.
        self.rng = np.random.RandomState(self.cfg.redetect_seed)
        self.lost = False
        self.frames_lost = 0
        self.last_confident_center = None
        self.last_sample_boxes = []

    @torch.no_grad()
    def init(self, img, box):
        super(TrackerSiamFCLongTerm, self).init(img, box)
        self.lost = False
        self.frames_lost = 0
        self.last_confident_center = self.center.copy()
        self.last_sample_boxes = []

    @torch.no_grad()
    def update(self, img):
        self.last_sample_boxes = []

        if not self.lost:
            # Najprej poskusimo običajni SiamFC korak. Pred tem shranimo stanje,
            # ker ga moramo povrniti, če je odziv prenizek. S tem preprečimo,
            # da bi slab lokalni premik postal nova osnova za naslednje slike.
            state = self._save_tracking_state()
            box, score = super(TrackerSiamFCLongTerm, self).update(img)

            if score >= self.cfg.redetect_threshold:
                # Visok vrh korelacijskega odziva pomeni, da je lokalno iskanje
                # še zanesljivo. Ta položaj zato shranimo kot zadnji zaupanja
                # vreden položaj za morebitno kasnejšo Gaussovo vzorčenje.
                self.last_confident_center = self.center.copy()
                self.frames_lost = 0
                return box, score

            # Če je zaupanje prenizko, lokalnega premika ne sprejmemo. Sledilnik
            # preide v izgubljeno stanje in začne iskati po širšem delu slike.
            self._restore_tracking_state(state)
            self.lost = True
            self.frames_lost = 1
        else:
            self.frames_lost += 1

        box, score, found = self._redetect(img)
        if found:
            # Ponovno detekcijo sprejmemo samo, če najboljši vzorec preseže isti
            # prag kot običajno sledenje. Tako ima prehod iz izgubljenega stanja
            # enako merilo zaupanja kot prehod v izgubljeno stanje.
            self.lost = False
            self.frames_lost = 0
            self.last_confident_center = self.center.copy()

        return box, score

    def _save_tracking_state(self):
        return {
            'center': self.center.copy(),
            'target_sz': self.target_sz.copy(),
            'z_sz': self.z_sz,
            'x_sz': self.x_sz}

    def _restore_tracking_state(self, state):
        self.center = state['center']
        self.target_sz = state['target_sz']
        self.z_sz = state['z_sz']
        self.x_sz = state['x_sz']

    def _redetect(self, img):
        # V izgubljenem stanju ne predpostavljamo več majhnega premika tarče.
        # Namesto tega ocenimo več kandidatov in izberemo tistega z največjim
        # korelacijskim odzivom.
        centers = self._sample_centers(img)
        scores, locs = self._score_centers(img, centers)

        best_id = int(np.argmax(scores))
        best_score = float(scores[best_id])
        best_center = self._refine_sample_center(centers[best_id], locs[best_id])
        found = best_score >= self.cfg.redetect_threshold

        if found:
            self.center = best_center

        box_center = best_center if found else centers[best_id]
        box = self._box_from_center(box_center)
        return box, best_score, found

    def _sample_centers(self, img):
        height, width = img.shape[:2]
        n = max(1, int(self.cfg.redetect_samples))

        if self.cfg.redetect_sampling == 'gaussian':
            # Gaussovo vzorčenje daje prednost okolici zadnjega zanesljivega
            # položaja, ker se tarča pogosto vrne blizu mesta, kjer je izginila.
            # Del uniformnih vzorcev ohranimo, da sledilnik še vedno lahko najde
            # tarčo, če se pojavi drugje.
            uniform_n = int(round(n * self.cfg.redetect_uniform_ratio))
            gaussian_n = n - uniform_n
            center = self.last_confident_center
            if center is None:
                center = np.array([height / 2.0, width / 2.0], dtype=np.float32)
            # Daljše izgubljeno stanje poveča negotovost, zato širimo standardni
            # odklon in postopoma pregledujemo večji del slike.
            sigma = self.cfg.redetect_sigma * \
                (self.cfg.redetect_sigma_growth ** max(0, self.frames_lost - 1))
            gaussian = self.rng.normal(
                loc=center, scale=sigma, size=(gaussian_n, 2))
            uniform = self._sample_uniform_centers(height, width, uniform_n)
            centers = np.vstack((gaussian, uniform)) if uniform_n else gaussian
        else:
            centers = self._sample_uniform_centers(height, width, n)

        anchors = [self.last_confident_center, self.center]
        anchors = [a for a in anchors if a is not None]
        if anchors:
            # Sidrne točke vedno dodamo med kandidate. Naključni vzorci jih lahko
            # zgrešijo, zadnji znani položaj pa je poceni in pogosto uporaben test.
            centers = np.vstack((np.array(anchors), centers))

        # Vzorce omejimo na sliko. Robne vzorce še vedno ocenimo, saj lahko
        # tarča ponovno vstopi skozi rob kadra.
        centers[:, 0] = np.clip(centers[:, 0], 0, height - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, width - 1)
        self.last_sample_boxes = [self._box_from_center(c) for c in centers]
        return centers.astype(np.float32)

    def _sample_uniform_centers(self, height, width, n):
        if n <= 0:
            return np.empty((0, 2), dtype=np.float32)

        ys = self.rng.uniform(0, height - 1, size=n)
        xs = self.rng.uniform(0, width - 1, size=n)
        return np.stack((ys, xs), axis=1).astype(np.float32)

    def _score_centers(self, img, centers):
        self.net.eval()
        all_scores = []
        all_locs = []
        batch_size = max(1, int(self.cfg.redetect_batch_size))

        for start in range(0, len(centers), batch_size):
            # Kandidate ocenjujemo v paketih, ker je to hitreje kot po en vzorec,
            # hkrati pa ne porabi toliko pomnilnika kot en velik paket.
            batch_centers = centers[start:start + batch_size]
            patches = [ops.crop_and_resize(
                img, center, self.x_sz,
                out_size=self.cfg.instance_sz,
                border_value=self.avg_color) for center in batch_centers]
            x = np.stack(patches, axis=0)
            x = torch.from_numpy(x).to(
                self.device).permute(0, 3, 1, 2).float()

            x = self.net.backbone(x)
            responses = self.net.head(self.kernel, x)
            responses = responses.squeeze(1).cpu().numpy()

            for response in responses:
                response = cv2.resize(
                    response, (self.upscale_sz, self.upscale_sz),
                    interpolation=cv2.INTER_CUBIC)
                all_scores.append(max(0, response.max()))
                all_locs.append(np.unravel_index(response.argmax(), response.shape))

        return np.array(all_scores), np.array(all_locs)

    def _refine_sample_center(self, sample_center, loc):
        # Vzorec poda grobo lokacijo, vrh odziva znotraj izreza pa popravi center.
        # Tako ponovno detekcijo poravnamo enako kot običajni SiamFC korak.
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz / self.cfg.instance_sz
        return sample_center + disp_in_image

    def _box_from_center(self, center):
        return np.array([
            center[1] + 1 - (self.target_sz[1] - 1) / 2,
            center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
