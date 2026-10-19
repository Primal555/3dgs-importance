"""Opt-in, training-only unconditional PatchGAN for rendered RGB.

G is updated before D. D then uses detached, pre-G-update cached views; no
second scene decode/render and no renderer/LPIPS graphs survive into D training.
"""
import math
import time

import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask


def adversarial_settings(weight=.001, discriminator_lr=1e-4, discriminator_betas=(.5, .999)):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('GAN weight must be finite and nonnegative')
    if not math.isfinite(discriminator_lr) or discriminator_lr <= 0:
        raise ValueError('discriminator learning rate must be finite and positive')
    if len(discriminator_betas)!=2 or any(not math.isfinite(v) or not 0<=v<1 for v in discriminator_betas):
        raise ValueError('discriminator Adam betas must be finite in [0,1)')
    return dict(version='render_patchgan_v1', gan_weight=float(weight),
                discriminator_lr=float(discriminator_lr), discriminator_betas=list(discriminator_betas),
                discriminator='unconditional 70x70 PatchGAN; spectral-normalized conv 64/128/256/512/1',
                spectral_initialization='20 CPU power iterations per convolution, no data or optimizer updates',
                update_order='G then D; one each per render update; D uses detached pre-G images',
                gan_weight_origin='tunable engineering starting point; not a published NTSCC numeric setting')


class PatchDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for index,(a,b,stride) in enumerate(zip((3,64,128,256,512),(64,128,256,512,1),(2,2,2,1,1))):
            layers.append(spectral_norm(torch.nn.Conv2d(a,b,4,stride,1)))
            if index<4:layers.append(torch.nn.LeakyReLU(.2))
        self.layers=torch.nn.Sequential(*layers)

    def forward(self, image):
        return self.layers(image)


class AdversarialImageObjective(torch.nn.Module):
    kind='mse_gan'

    def __init__(self, weight, gan_weight, device, *, discriminator_lr=1e-4,
                 discriminator_betas=(.5,.999)):
        super().__init__()
        self.settings=adversarial_settings(gan_weight,discriminator_lr,discriminator_betas)
        self.perceptual=PerceptualImageObjective('mse_lpips',weight,device)
        self.weight=self.perceptual.weight
        self.gan_weight=float(gan_weight)
        # D is initialized on CPU; adding this arm must not perturb view/noise RNG.
        with torch.random.fork_rng(devices=[]):
            self.discriminator=PatchDiscriminator()
            # The first G forward uses eval mode. PyTorch's initial random u/v
            # are NOT yet a singular-vector estimate; calibrate before freezing
            # them or the initial D logits/gradients can explode or saturate.
            with torch.no_grad():
                for layer in self.discriminator.modules():
                    if hasattr(layer,'weight_orig'):
                        matrix=layer.weight_orig.flatten(1)
                        for _ in range(20):
                            layer.weight_v.copy_(F.normalize(matrix.t() @ layer.weight_u,dim=0,eps=1e-12))
                            layer.weight_u.copy_(F.normalize(matrix @ layer.weight_v,dim=0,eps=1e-12))
            self.discriminator.to(device)
        self.optimizer=torch.optim.Adam(self.discriminator.parameters(),lr=discriminator_lr,
                                        betas=tuple(discriminator_betas))
        self.updates=0
        self.begin_generator()

    def protocol(self):
        protocol=self.perceptual.protocol()
        protocol.update(kind=self.kind, formula='MSE + weight * LPIPS + gan_weight * softplus(-D(fake))',
                        adversarial=self.settings,
                        discriminator_input='unpaired full-resolution displayed RGB mapped to [-1,1]; no latent/camera condition',
                        discriminator_reference='source PLY rendering, never training photos',
                        limitations='not NTSCC cGAN replication; no multi-view consistency guarantee; textures may be hallucinated')
        return protocol

    def begin_generator(self):
        # eval ALSO freezes spectral norm buffers, not only the trainable tensors.
        self.discriminator.eval().requires_grad_(False)
        self.optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _rgb(image):
        if image.ndim!=3 or image.shape[0]!=3 or min(image.shape[-2:])<32:
            raise ValueError('PatchGAN requires nonempty 3xHxW RGB with height/width >= 32')
        if not torch.isfinite(image).all():raise ValueError('PatchGAN RGB must be finite')
        return image.clamp(0,1).float().mul(2).sub(1)[None]

    def forward(self, image, target):
        # Defensive: no caller may accidentally train D buffers through a G loss.
        self.begin_generator()
        fake=self._rgb(image)
        self._rgb(target)
        loss,terms=self.perceptual(image,target)
        adversarial=F.softplus(-self.discriminator(fake)).mean()
        weighted=self.gan_weight*adversarial
        loss=loss+weighted
        if not torch.isfinite(loss):raise RuntimeError('nonfinite adversarial generator loss')
        terms.update(image_lpips_weighted=terms['image_aux_weighted'],image_gan_loss=adversarial,
                     image_gan_weighted=weighted,image_aux_weighted=terms['image_aux_weighted']+weighted)
        return loss,terms

    def update_discriminator(self, pairs):
        if not pairs:raise ValueError('discriminator requires cached training views')
        device=next(self.discriminator.parameters()).device
        # Validate ALL views before changing spectral buffers or applying Adam.
        for fake,real in pairs:
            if fake.shape!=real.shape:raise ValueError('discriminator real/fake RGB shape mismatch')
            self._rgb(fake);self._rgb(real)
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        before=[p.detach().clone() for p in self.discriminator.parameters()]
        self.discriminator.train().requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        losses=[];real_logits=[];fake_logits=[]
        try:
            for fake,real in pairs:
                # Consume one view/branch at a time to bound activation memory.
                for image,is_real in ((real,True),(fake,False)):
                    logits=self.discriminator(self._rgb(image.detach().to(device)))
                    term=F.softplus(-logits if is_real else logits).mean()
                    if not torch.isfinite(term):raise RuntimeError('nonfinite discriminator loss')
                    (term*(.5/len(pairs))).backward()
                    losses.append(float(term.detach())*.5/len(pairs))
                    (real_logits if is_real else fake_logits).append(float(logits.detach().mean()))
            gradients=[p.grad.detach().norm() for p in self.discriminator.parameters() if p.grad is not None]
            norm=torch.stack(gradients).norm()
            if not torch.isfinite(norm):raise RuntimeError('nonfinite discriminator gradient')
            self.optimizer.step()
            changes=torch.stack([(p.detach()-old).norm() for p,old in zip(self.discriminator.parameters(),before)]).norm()
            if not torch.isfinite(changes):raise RuntimeError('nonfinite discriminator update')
            self.updates+=1
            if device.type=='cuda':torch.cuda.synchronize(device)
            return dict(discriminator_loss=sum(losses),discriminator_real_logit=sum(real_logits)/len(pairs),
                        discriminator_fake_logit=sum(fake_logits)/len(pairs),discriminator_grad_norm=float(norm),
                        discriminator_update_norm=float(changes),discriminator_updates=self.updates,
                        discriminator_seconds=time.perf_counter()-start)
        finally:
            self.begin_generator()

    def training_state(self):
        return dict(protocol=self.protocol(),discriminator=self.discriminator.state_dict(),
                    optimizer=self.optimizer.state_dict(),updates=self.updates)

    def restore_training_state(self, state):
        if not isinstance(state,dict) or not all(k in state for k in ('protocol','discriminator','optimizer','updates')):
            raise ValueError('exact adversarial resume requires discriminator and Adam state')
        if state['protocol']!=self.protocol():raise ValueError('adversarial protocol mismatch')
        if type(state['updates']) is not int or state['updates']<0:raise ValueError('invalid adversarial update count')
        self.discriminator.load_state_dict(state['discriminator'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates=state['updates']
        self.begin_generator()


class AdversarialMultiViewRenderTask(PerceptualMultiViewRenderTask):
    def __init__(self, cameras, reference, degree, white_background, objective):
        super().__init__(cameras,reference,degree,white_background,objective)
        self.pairs=[]

    def _terms(self, scene):
        from .rendering import render
        self.components=[];self.pairs=[]
        self.objective.begin_generator()
        for camera in self.cameras:
            image=render(scene,camera,self.degree,self.white_background)
            target=self.reference.get(camera,scene.device)
            loss,terms=self.objective(image,target)
            # CPU copies only, NEVER keep the renderer or VGG graph for D.
            self.pairs.append((image.detach().clamp(0,1).cpu(),target.detach().clamp(0,1).cpu()))
            self.components.append({k:float(v.detach()) for k,v in terms.items()})
            yield loss

    def finish_discriminator(self):
        try:
            return self.objective.update_discriminator(self.pairs)
        finally:
            self.pairs.clear()
