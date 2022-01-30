from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from sparselearning.snip import SNIP
import numpy as np
import math

def add_sparse_args(parser):
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix sparse connectivity during training. Default: False.')
    parser.add_argument('--sparse_init', type=str, default='ERK', help='sparse initialization')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    # hyperparameters for NTK
    parser.add_argument('--reini', type=str, default='', help='reinitialization mode: NTK, isometry')
    parser.add_argument('--LAMBDA_KER_DIST', type=float, default=1.0e-3, help='scaling parameter for kernel distance.')
    parser.add_argument('--LR_NTK', type=float, default=5e-4, help='learning rate for NTK adam.')
    parser.add_argument('--L2_NTK', type=float, default=1e-8, help='weight decay for NTK adam.')
    parser.add_argument('--epochs_NTK', type=int, default=20, help='number of training epoch for NTK adam.')
    # hyperparameters for pruning at initialization
    parser.add_argument('--mu', type=float, default=0.1, help='hyperparameters for l1 authogonal pruning at initialization')


def get_model_params(model):
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params

def set_model_params(model, model_parameters):
    model.load_state_dict(model_parameters)


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, args=False, train_loader=False):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.train_loader = train_loader
        self.args = args
        self.LAMBDA_KER_DIST = self.args.LAMBDA_KER_DIST
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.reini = args.reini
        self.mu = args.mu

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.name2nonzeros = {}
        self.death_rate = death_rate
        self.steps = 0

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix: self.prune_every_k_steps = None
        else: self.prune_every_k_steps = self.args.update_frequency

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
        self.density = density
        if mode == 'global_magnitude':
            print('initialize by global magnitude')
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * self.density)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = ((torch.abs(weight)) >= acceptable_score).float()

        elif mode == 'l1_authogonal':
            masks = self.l1_authogonal(mu=self.mu, density=density)
            for mask, name in zip(masks, self.masks):
                assert (mask.shape == self.masks[name].shape)
                self.masks[name][:] = mask

        elif mode == 'snip':
            print('initialize by snip')
            snip_masks = SNIP(self.module, self.density, self.train_loader, self.device)
            for snip_mask, name in zip(snip_masks, self.masks):
                assert (snip_mask.shape == self.masks[name].shape)
                self.masks[name][:] = snip_mask

        elif mode == 'uniform_plus':
            print('initialize by uniform+')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            total_sparse_params = total_params * self.density

            # remove the first layer
            total_sparse_params = total_sparse_params - self.masks['conv.weight'].numel()
            self.masks.pop('conv.weight')

            if self.density < 0.2:
                total_sparse_params = total_sparse_params - self.masks['fc.weight'].numel() * 0.2
                self.density = float(total_sparse_params / total_params)

                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        if name != 'fc.weight':
                            self.masks[name][:] = (torch.rand(weight.shape) < self.density).float().data.cuda()
                        else:
                            self.masks[name][:] = (torch.rand(weight.shape) < 0.2).float().data.cuda()
            else:
                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        self.masks[name][:] = (torch.rand(weight.shape) < self.density).float().data.cuda()

        elif mode == 'uniform':
            print('initialize by uniform')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < self.density).float().data.cuda() #lsw
                    # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                    self.baseline_nonzero += weight.numel()*density

        elif mode == 'modifide_ERK':

            print('initialize by fixed_ERK')

            total_params = 0

            self.baseline_nonzero = 0

            for name, weight in self.masks.items():
                total_params += weight.numel()

                self.baseline_nonzero += weight.numel() * density

            # remove the first layer
            total_params = total_params - self.masks['fc.weight'].numel()
            density = (self.baseline_nonzero - self.masks['fc.weight'].numel() * 2 * self.density) / total_params

            self.masks.pop('fc.weight')


            is_epsilon_valid = False

            dense_layers = set()

            while not is_epsilon_valid:

                divisor = 0

                rhs = 0

                raw_probabilities = {}

                for name, mask in self.masks.items():

                    n_param = np.prod(mask.shape)

                    n_zeros = n_param * (1 - density)

                    n_ones = n_param * density

                    if name in dense_layers:

                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.

                        rhs -= n_zeros

                    else:
                        rhs += n_ones


                        if len(mask.shape) != 2:

                            raw_probabilities[name] = (

                                                              np.sum(mask.shape[:-1]) / np.prod(mask.shape[:-1])

                                                      ) ** erk_power_scale
                        else:

                            raw_probabilities[name] = (

                                                              np.sum(mask.shape) / np.prod(mask.shape)

                                                      ) ** erk_power_scale



                        divisor += raw_probabilities[name] * n_param

                epsilon = rhs / divisor

                max_prob = np.max(list(raw_probabilities.values()))

                max_prob_one = max_prob * epsilon

                if max_prob_one > 1:

                    is_epsilon_valid = False

                    for mask_name, mask_raw_prob in raw_probabilities.items():

                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")

                            dense_layers.add(mask_name)

                else:

                    is_epsilon_valid = True

            density_dict = {}

            total_nonzero = 0.0

            # With the valid epsilon, we can set sparsities of the remaning layers.

            for name, mask in self.masks.items():

                n_param = np.prod(mask.shape)

                if name in dense_layers:

                    density_dict[name] = 1.0

                else:

                    probability_one = epsilon * raw_probabilities[name]

                    density_dict[name] = probability_one

                print(

                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"

                )

                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()

            for name, weight in self.module.named_parameters():

                if name == 'fc.weight':
                    self.masks[name] = (torch.rand(weight.shape) < 2 * self.density).float().data.cuda()

                    total_nonzero += 2 * self.density * weight.numel()

            print(f"Overall sparsity {total_nonzero / total_params}")

            self.apply_mask()

        elif mode == 'ERK':
            print('initialize by ERK')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False

            dense_layers = set()
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - self.density)
                    n_ones = n_param * self.density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        self.apply_mask()

        total_size = 0
        sparse_size = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name in self.masks:
                    print(name, 'density:', (weight != 0).sum().item() / weight.numel())
                    total_size += weight.numel()
                    sparse_size += (weight != 0).sum().int().item()
        print('Total Model parameters:', total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        self.death_rate = self.death_rate_decay.get_dr()
        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.truncate_weights()
                _, _ = self.fired_masks_update()
                self.print_nonzero_counts()


    def add_module(self, module, density, sparse_init='ER'):
        self.module = module
        self.modules.append(module)
        self.dense_model = copy.deepcopy(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        self.init(mode=sparse_init, density=density)
        if self.reini == 'isometry':
            self.get_jacobian(module, self.train_loader, self.device, create_graph=False, strict=False)
            self.approximnate_isometry()
            self.get_jacobian(module, self.train_loader, self.device, create_graph=False, strict=False)
        elif self.reini == 'NTK':
            self.NTK_approx()


    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor][
                                                                              'momentum_buffer'] * self.masks[name]

    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_prune_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density, sparse_size / total_size, epoch))

    def truncate_weights(self):

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                # death
                if self.death_mode == 'magnitude':
                    new_mask = self.magnitude_death(mask, weight, name)
                elif self.death_mode == 'SET':
                    new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                elif self.death_mode == 'Taylor_FO':
                    new_mask = self.taylor_FO(mask, weight, name)
                elif self.death_mode == 'threshold':
                    new_mask = self.threshold_death(mask, weight, name)

                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.masks[name][:] = new_mask


        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth(name, new_mask, weight)

                if self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name, new_mask, weight)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask, weight)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth(name, new_mask, weight)

                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

        self.apply_mask()


    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name]==0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name]==0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def random_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask



    def momentum_neuron_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)


        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights

    #*********************************************Dynamic Isomytry****************************************************
    def get_jacobian(self, func, inputs, device, create_graph=False, strict=False):
        for batch_idx, (data, target) in enumerate(inputs):
            if batch_idx ==0:
                print('data len:', len(data))
                data, target = data.to(device), target.to(device)
                jacobian_4d = torch.autograd.functional.jacobian(func, data, create_graph=create_graph,
                                                                 strict=strict)  # 100, 10, 100, 784
                jacobian_4d = jacobian_4d.view(self.args.batch_size, 10, self.args.batch_size, -1)
                jacobian_3d = torch.zeros(jacobian_4d.size(0), jacobian_4d.size(1), jacobian_4d.size(3))  # 100, 10, 784
                for i in range(jacobian_3d.size(0)):
                    for j in range(jacobian_3d.size(1)):
                        jacobian_3d[i][j] = jacobian_4d[i][j][i]
                # u, s, vh = torch.svd(jacobian_3d.cpu().numpy()[1], full_matrices=True)
                svj = torch.svd(jacobian_3d)[1]
                print('SVJ is:', svj.mean(), 'SJV size is:', svj.size())
                return svj
            else: break

    def l1_authogonal(self, mu=0.01, density=0.5):
        """
        a python implementation solving the following problem:
        Given initial weight W, we minimize the optimization problem by optimizing M:
        min_{M}    || （M.*W)^T (M.*W) - Ir ||_1 + mu || M.*W ||_1
        where M is the target mask, which should be binary and M.*W is nearly orthogonal
        and M.*W is sparse
        """
        print('pruning at initializatin with l1norm authogonal')
        approxed_paras = copy.deepcopy(self.masks)
        for name in approxed_paras:
            approxed_paras[name].requires_grad = True
        optimizer = optim.SGD(approxed_paras.values(), momentum=0.9, lr=0.1)

        for e in range(0, 10000):
            if e == int(10000/3):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.01
            if e == int(10000*2/3):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001

            self.apply_mask()
            optimizer.zero_grad()
            layer_loss = {}
            for name, weight in self.dense_model.named_parameters():
                if name in self.masks:
                    para = approxed_paras[name] * weight
                    layer_loss[name] = torch.norm(torch.matmul(torch.transpose(para.view(-1, para.size()[1]),0,1), para.view(-1, para.size()[1])) - torch.eye(para.size()[1]).cuda()) \
                                       + mu*torch.norm(para, 1)

            loss = sum(layer_loss.values())
            # loss.to(self.device)
            loss.backward(retain_graph=True)
            optimizer.step()
            if e % 1000 == 0:
                print(e, loss.item())


        scores = []
        for name, weight in self.dense_model.named_parameters():
            if name in self.masks:
                scores.append(torch.abs(approxed_paras[name]* weight))

        all_scores = torch.cat([torch.flatten(x) for x in scores])

        num_params_to_keep = int(len(all_scores) * density)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        layer_wise_sparsities = []
        for g in scores:
            mask = (g >= acceptable_score).float()
            layer_wise_sparsities.append(mask)

        return layer_wise_sparsities

    def approximnate_isometry(self):
        # use dictionary to apply masks.
        print('approximate_isometry starts...')
        approxed_paras = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                approxed_paras[name] = weight
        optimizer = optim.SGD(approxed_paras.values(), momentum=0.9, lr=0.1)
        for e in range(1, 10001):
            if e == int(10000/3):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.01
            if e == int(10000*2/3):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
            self.apply_mask()
            optimizer.zero_grad()
            layer_loss = {}
            for name in approxed_paras:
                para = approxed_paras[name]
                layer_loss[name] = torch.norm(torch.matmul(torch.transpose(para.view(-1, para.size()[1]),0,1), para.view(-1, para.size()[1])) - torch.eye(para.size()[1]).cuda())
            loss = sum(layer_loss.values())
            # loss.to(self.device)
            loss.backward(retain_graph=True)
            optimizer.step()
            if e % 1000 == 0:
                print(e, loss.item())
        self.apply_mask()

    # ***********************************************NTK Approximation***************************************************
    def NTK_approx(self):
        print('approximate dense NTK...')
        # optimizer = optim.Adam(approxed_paras.values(), lr=self.args.lr, weight_decay=self.args.l2)
        # self.apply_mask()
        for epoch in range(1, self.args.epochs_NTK):
            self.ntk_app_train(self.train_loader, [self.dense_model, self.module], train_mode=False, num_batch=-1)


    def recal_bn(self, network, xloader, recalbn, device):
        for m in network.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.data.fill_(0)
                m.running_var.data.fill_(0)
                m.num_batches_tracked.data.zero_()
                m.momentum = None
        network.train()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(xloader):
                if i >= recalbn: break
                inputs = inputs.cuda(device=device, non_blocking=True)
                _, _ = network(inputs)
        return network

    def ntk_app_train(self, xloader, networks, recalbn=0, train_mode=False, num_batch=-1):
        device = torch.cuda.current_device()
        # if recalbn > 0:
        #     network = recal_bn(network, xloader, recalbn, device)
        #     if network_2 is not None:
        #         network_2 = recal_bn(network_2, xloader, recalbn, device)
        approxed_paras = {}
        for name, weight in networks[1].named_parameters():
            if name not in self.masks: continue
            approxed_paras[name] = weight

        networks[0].eval()
        networks[1].eval()

        # optimizer = optim.SGD(networks[1].parameters(), lr=0.001, weight_decay=5e-4)
        optimizer = optim.Adam(approxed_paras.values(), lr=self.args.LR_NTK, weight_decay=self.args.L2_NTK)
        ######
        for i, (inputs, targets) in enumerate(xloader):
            logits = []
            grads = [[] for _ in range(len(networks))]
            if num_batch > 0 and i >= num_batch: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            for net_idx, network in enumerate(networks):
                network.zero_grad()
                inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
                logit = network(inputs_)
                logits.append(logit)

                if isinstance(logit, tuple):
                    logit = logit[1]  # 201 networks: return features and logits
                for _idx in range(len(inputs_)):
                    logit[_idx:_idx + 1].backward(torch.ones_like(logit[_idx:_idx + 1]), retain_graph=True)
                    grad = []
                    for name, W in network.named_parameters():
                        if 'weight' in name and W.grad is not None:
                            grad.append(W.grad.view(-1).detach())
                    grads[net_idx].append(torch.cat(grad, -1))
                    network.zero_grad()
                    torch.cuda.empty_cache()
            ######
            # print(torch.equal(logits[0], logits[1]))
            grads = [torch.stack(_grads, 0) for _grads in grads]
            ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
            # conds = []
            # for ntk in ntks:
            #     eigenvalues, _ = torch.symeig(ntk)  # ascending
            #     conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))

            # calculating loss
            ker_dist, target_dist = self.kernel_dist_target_dist_l2_loss(ntks[1], logits[1],
                                                                         ntks[0], logits[0],)
            weighted_ker_dist = self.LAMBDA_KER_DIST * ker_dist

            transfer_loss = weighted_ker_dist + target_dist
            # transfer_loss = target_dist

            optimizer.zero_grad()
            transfer_loss.backward(retain_graph=True)
            optimizer.step()
            self.apply_mask()

            if i%10 == 0:
                print(f'iteration {i}:, loss is {transfer_loss.item()}')

    def kernel_dist_target_dist_l2_loss(self, student_ker_mat, student_pred, teacher_ker_mat, teacher_pred
                                        ):

        """ Compute kernel distance, target distance, and parameter l2 loss.
        Args:
            student_ker_mat: a student-network ntk matrix,
            student_pred: a student-network prediction matrix.
            teacher_ker_mat: a teacher-network ntk matrix.
            teacher_pred: a teacher-network prediction matrix.
        Returns:
            ker_dist: squared l2 difference between two kernel matrices, normalized by the size of the matrix.
            target_dist: squared l2 norm difference between two prediction matrices, normalized by the size of the matrix.
            params_norm_squared: squared l2 norm of parameters.
        """

        # the normalized squared difference between teacher and student NTK matrices
        ker_dist = torch.mean(torch.square(student_ker_mat - teacher_ker_mat))

        # the normalized squared difference between teacher and student network predictions
        target_dist = torch.mean(torch.square(student_pred - teacher_pred))

        return ker_dist, target_dist

