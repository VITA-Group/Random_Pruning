from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math

def add_sparse_args(parser):
    parser.add_argument('--ini', type=str, default='Grasp', help='Grasp, Snip')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--pruning-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--snip', action='store_true', help='Enable snip initialization. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--pop', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='ER', help='sparse initialization')
    parser.add_argument('--mix', type=float, default=0.0)
    # DST hyperparameters
    parser.add_argument('--method', type=str, default='DST', help='method name: DST, MPDS, GMP, NTK_path')
    parser.add_argument('--ini-density', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--final-density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--init_prune_epoch', type=int, default=8, help='The pruning rate / death rate.')
    parser.add_argument('--final_prune_epoch', type=int, default=125, help='The density of the overall sparse network.')
    parser.add_argument('--N', type=int, default=8, help='N of NM sparisty')
    parser.add_argument('--M', type=int, default=16, help='M of NM sparisty')
    # parser.add_argument('--sparse', action='store_false', default=False, help='Enable sparse mode. Default: True.')

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
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
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, args=None, snip_masks=None, train_loader=None, device=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.loader = train_loader
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.snip_masks = snip_masks # initial masks made by SNIP


        self.death_funcs = {}
        self.death_funcs['magnitude'] = self.magnitude_death
        self.death_funcs['SET'] = self.magnitude_and_negativity_death
        self.death_funcs['threshold'] = self.threshold_death

        self.growth_funcs = {}
        self.growth_funcs['random'] = self.random_growth
        self.growth_funcs['momentum'] = self.momentum_growth
        self.growth_funcs['momentum_neuron'] = self.momentum_neuron_growth

        self.masks = {}
        self.final_masks = {}
        self.grads = {}
        self.nonzero_masks = {}
        self.scores = {}
        self.pruning_rate = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.death_rate = death_rate
        self.name2death_rate = {}
        self.steps = 0

        # global growth/death state
        self.threshold = threshold
        self.growth_threshold = threshold
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        # if self.args.fix:
        #     self.prune_every_k_steps = None
        # else:
        #     self.prune_every_k_steps = self.args.update_frequency


    def init(self, mode='ER', density=0.05, erk_power_scale=1.0, grad_dict=None, customer_density=None):
        self.density = density
        if self.sparse_init == 'customer':
            # print('initialized by customer')
            self.baseline_nonzero = 0
            for index, name in enumerate(self.masks):
                self.masks[name][:] = (torch.rand(self.masks[name].shape) < (customer_density[index])).float()
                self.baseline_nonzero += self.masks[name].numel() * (customer_density[index])
            self.apply_mask()
        if self.sparse_init == 'prune':
            # used for pruning stabability test
            print('initialized by pruning')

            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = (weight!=0).cuda()
                    num_zeros = (weight==0).sum().item()
                    num_remove = (self.args.pruning_rate) * self.masks[name].sum().item()
                    k = math.ceil(num_zeros + num_remove)
                    if num_remove == 0.0: return weight.data != 0.0
                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    self.masks[name].data.view(-1)[idx[:k]] = 0.0
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            self.apply_mask()
        if self.sparse_init == 'prune_global':
            # used for pruning stabability test
            print('initialized by prune_global')
            self.baseline_nonzero = 0
            total_num_nonzoros = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = (weight!=0).cuda()
                    self.name2nonzeros[name] = (weight!=0).sum().item()
                    total_num_nonzoros += self.name2nonzeros[name]

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(total_num_nonzoros * (1 - self.args.pruning_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
            self.apply_mask()
        if self.sparse_init == 'prune_and_grow':
            # used for pruning stabability test
            print('initialized by pruning and growing')

            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    # prune
                    self.masks[name] = (weight!=0).cuda()
                    num_zeros = (weight==0).sum().item()
                    num_remove = (self.args.pruning_rate) * self.masks[name].sum().item()
                    k = math.ceil(num_zeros + num_remove)
                    if num_remove == 0.0: return weight.data != 0.0
                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    self.masks[name].data.view(-1)[idx[:k]] = 0.0
                    total_regrowth = (self.masks[name]==0).sum().item() - num_zeros

                    # set the pruned weights to zero
                    weight.data = weight.data * self.masks[name]
                    if 'momentum_buffer' in self.optimizer.state[weight]:
                        self.optimizer.state[weight]['momentum_buffer'] = self.optimizer.state[weight]['momentum_buffer'] * self.masks[name]

                    # grow
                    grad = grad_dict[name]
                    grad = grad * (self.masks[name] == 0).float()

                    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
                    self.masks[name].data.view(-1)[idx[:total_regrowth]] = 1.0
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            self.apply_mask()
        if self.sparse_init == 'prune_and_grow_global':
            # used for pruning stabability test
            print('initialized by prune_and_grow_global')
            self.baseline_nonzero = 0
            total_num_nonzoros = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = (weight!=0).cuda()
                    self.name2nonzeros[name] = (weight!=0).sum().item()
                    total_num_nonzoros += self.name2nonzeros[name]

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(total_num_nonzoros * (1 - self.args.pruning_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

                    # set the pruned weights to zero
                    weight.data = weight.data * self.masks[name]
                    if 'momentum_buffer' in self.optimizer.state[weight]:
                        self.optimizer.state[weight]['momentum_buffer'] = self.optimizer.state[weight]['momentum_buffer'] * self.masks[name]

            ### grow
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    total_regrowth = self.name2nonzeros[name] - (self.masks[name]!=0).sum().item()
                    grad = grad_dict[name]
                    grad = grad * (self.masks[name] == 0).float()

                    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
                    self.masks[name].data.view(-1)[idx[:total_regrowth]] = 1.0
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            self.apply_mask()

        if self.sparse_init == 'GMP':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
                    # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            self.apply_mask()
        elif self.sparse_init == 'resume':
            print('initialized with LTR OR LRR')
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print(name, (weight != 0.0).sum().item())
                    self.masks[name][:] = (weight != 0).float().data.cuda()
                    self.baseline_nonzero += weight.numel() * density
            self.apply_mask()
        elif self.sparse_init == 'uniform':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda() #lsw
                    # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                    self.baseline_nonzero += weight.numel()*density
            self.apply_mask()
        elif self.sparse_init == 'NM_sparsity':
            print('initialize by NM_sparsity')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    length = weight.numel()
                    group = int(length / self.args.M)

                    index = torch.argsort(torch.rand(*(weight.view(group, self.args.M).shape)), dim=1)[:, :int(self.args.M - self.args.N)].cuda()
                    self.masks[name] = self.masks[name].view(group, self.args.M).scatter_(dim=1, index=index,
                                                                                          value=0).reshape(weight.shape)
            self.apply_mask()
        elif self.sparse_init == 'fixed_ERK':
            print('initialize by fixed_ERK')
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
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
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
                # print(
                #     f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                # )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        # for name, mask in self.masks.copy().items():
        #     if (mask != 0).sum().int().item() == mask.numel():
        #         self.masks.pop(name)
        #         print(f"pop out {name}")

        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks) # used for over-paremeters
        # self.nonzero_masks = copy.deepcopy(self.masks)  # used for over-paremeters

        self.init_death_rate(self.death_rate)
        # self.print_nonzero_counts()

        # total_size = 0
        # for name, weight in self.masks.items():
        #     total_size  += weight.numel()
        # print('Total Model parameters:', total_size)
        #
        # sparse_size = 0
        # for name, weight in self.masks.items():
        #     sparse_size += (weight != 0).sum().int().item()
        #
        # print('Total initial parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))


    def init_death_rate(self, death_rate):
        for name in self.masks:
            self.name2death_rate[name] = death_rate

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        for name in self.masks:
            if self.args.decay_schedule == 'cosine':
                self.name2death_rate[name] = self.death_rate_decay.get_dr(self.name2death_rate[name])
            elif self.args.decay_schedule == 'constant':
                self.name2death_rate[name] = self.args.death_rate
            self.death_rate = self.name2death_rate[name]
        self.steps += 1

        if self.prune_every_k_steps is not None:
                if self.args.method == 'GDP':
                    if self.steps >= (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                        self.pruning(self.steps)
                        self.truncate_weights(self.steps)
                        self.print_nonzero_counts()
                    # _, _ = self.fired_masks_update()
                elif self.args.method == 'DST':
                    if self.steps % self.prune_every_k_steps == 0:
                        self.truncate_weights()
                        self.print_nonzero_counts()
                elif self.args.method == 'GMP':
                    if self.steps >= (self.args.init_prune_epoch * len(self.loader) * self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                        self.pruning(self.steps)
                elif self.args.method == 'NM_GDP':
                    pruning_times = self.args.M - self.args.N
                    final_iter = (self.args.final_prune_epoch * len(self.loader) * self.args.multiplier)
                    ini_iter = (self.args.init_prune_epoch * len(self.loader) * self.args.multiplier)
                    update_interval = (final_iter - ini_iter) / pruning_times

                    if self.steps % update_interval == 0 and self.steps >= ini_iter:
                        if self.steps <= final_iter:
                            self.NM_pruning(self.steps, update_interval)
                        self.truncate_weights_NM()
                        self.print_nonzero_counts()
                elif self.args.method == 'NM_DST':
                    if self.steps % self.prune_every_k_steps == 0:
                        self.truncate_weights_NM()
                        self.print_nonzero_counts()
                elif self.args.method == 'NM_GMP':
                    pruning_times = self.args.M - self.args.N
                    final_iter = (self.args.final_prune_epoch * len(self.loader) * self.args.multiplier)
                    ini_iter = (self.args.init_prune_epoch * len(self.loader) * self.args.multiplier)
                    update_interval = (final_iter - ini_iter) / pruning_times
                    if self.steps % update_interval == 0 and self.steps >= ini_iter:
                        if self.steps <= final_iter:
                            self.NM_pruning(self.steps, update_interval)

    def NM_pruning(self, step, update_interval):

            pruning_index = step / update_interval
            current_N = self.args.M - pruning_index

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    length = weight.numel()
                    group = int(length / self.args.M)
                    weight_temp = weight.abs().view(group, self.args.M)

                    index = torch.argsort(weight_temp, dim=1)[:, :int(self.args.M - current_N)]
                    self.masks[name] = self.masks[name].view(group, self.args.M).scatter_(dim=1, index=index, value=0).reshape(
                        weight.shape)

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Total parameters after pruning under sparsity level of {0}: {1}'.format(
                self.args.ini_density,
                sparse_size / total_size))


    def pruning(self, step):
        # prune_rate = 1 - self.args.final_density - self.args.ini_density
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = math.floor(self.args.final_prune_epoch * len(self.loader)*self.args.multiplier / self.prune_every_k_steps)
        print(f"final iter is {final_iter}")
        ini_iter = (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps

        total_prune_iter = final_iter - ini_iter
        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            prune_decay = (1 - ((
                                        curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.ini_density) + (self.args.ini_density - self.args.final_density) * (
                    1 - prune_decay)
            print('current pruning rate is:', curr_prune_rate)
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

            self.apply_mask()

            total_size = 0
            # for name, weight in self.masks.items():
            #     total_size += weight.numel()
            # print('Total Model parameters:', total_size)
            #
            # sparse_size = 0
            # for name, weight in self.masks.items():
            #     sparse_size += (weight != 0).sum().int().item()
            #
            # print('Total parameters after pruning under sparsity level of {0}: {1}'.format(
            #     self.args.ini_density,
            #     sparse_size / total_size))

    def add_module(self, module, density, sparse_init='ER', grad_dic=None, customer_density=None):
        self.sparse_init = sparse_init
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 4 or len(tensor.size()) == 2:
                self.names.append(name)
                self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False)

        self.init(mode=sparse_init, density=density, grad_dict=grad_dic, customer_density=customer_density)


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
                    tensor.data = tensor.data*self.masks[name]
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights_GMP(self, epoch=None):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.args.final_density
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

    def kernel_path_update(self):
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        # self.modules[0].eval()
        self.gather_statistics()
        name2regrowth = self.calc_growth_redistribution()

        # save current gradients
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.grads[name] = weight.grad.clone()

        # calculate scores
        signs = linearize(self.modules[0])
        (data, _) = next(iter(self.loader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).cuda()  # , dtype=torch.float64).to(device)
        output = self.modules[0](input)
        self.optimizer.zero_grad()
        torch.sum(output).backward()

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]

                # prune
                new_mask = self.kernel_pruning(mask, weight, name)
                self.pruning_rate[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                # grow
                new_mask = self.kernel_gradient_growth(name, new_mask, self.pruning_rate[name], weight)

            self.masks.pop(name)
            self.masks[name] = new_mask.float()

        nonlinearize(self.modules[0], signs)
        self.apply_mask()
        # for module in self.modules:
        #     for name, weight in module.named_parameters():
        #         if name not in self.masks: continue
        #         self.scores[name] = torch.clone(weight.grad * weight).detach().abs_()
        #
        # # Prune: Gather all scores in a single vector and normalise
        # all_scores = torch.cat([torch.flatten(x) for name, x in self.scores.items()])
        # num_params_to_keep = int(len(all_scores) * (1 - self.density - self.density * self.death_rate))
        # threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        # acceptable_score = threshold[-1]
        #
        # for module in self.modules:
        #     for name, weight in module.named_parameters():
        #         if name not in self.masks: continue
        #         self.masks[name] = (self.scores[name] >= acceptable_score).float()

        # nonlinearize(self.modules[0], signs)

    def truncate_weights_NM(self, step=None):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.name2nonzeros[name] = (self.masks[name]!=0).sum().item()
                mask = self.masks[name]
                # death
                new_mask, group_removal = self.magnitude_death_NM(mask, weight, self.death_rate)

                new_mask = self.gradient_growth_NM(name, new_mask, group_removal, weight)
                self.masks[name] = new_mask.float()
        self.apply_mask()

    def truncate_weights(self, step=None):

        self.gather_statistics()
        name2regrowth = self.calc_growth_redistribution()

        total_nonzero_new = 0
        total_removed = 0
        if self.death_mode == 'global_magnitude':
            total_removed = self.global_magnitude_death()
        else:
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # death
                    if self.death_mode == 'magnitude':
                        new_mask = self.magnitude_death(mask, weight, name)
                    # elif self.death_mode == 'mag_gra':
                    #     new_mask = self.mag_gra(mask, weight, name, epoch)
                    elif self.death_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                    elif self.death_mode == 'Taylor_FO':
                        new_mask = self.taylor_FO(mask, weight, name)
                    elif self.death_mode == 'threshold':
                        new_mask = self.threshold_death(mask, weight, name)
                    elif self.death_mode == 'magnitude_increase':
                        new_mask = self.magnitude_increase(weight, mask, name)
                    elif self.death_mode == 'new_pruning':
                        if self.snip_masks:
                            new_mask = self.CS_death(mask, self.snip_masks[index])
                            index += 1
                        else:
                            print('No snip masks are available.')

                    total_removed += self.name2nonzeros[name] - new_mask.sum().item()
                    self.pruning_rate[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                    self.masks[name][:] = new_mask
                    self.nonzero_masks[name] = new_mask.float()

        # self.apply_mask()
        if self.growth_mode == 'global_momentum':
            total_nonzero_new = self.global_momentum_growth(total_removed + self.adjusted_growth)
        else:
            if self.death_mode == 'threshold':
                expected_killed = sum(name2regrowth.values())
                #print(expected_killed, total_removed, self.threshold)
                if total_removed < (1.0-self.tolerance)*expected_killed:
                    self.threshold *= 2.0
                elif total_removed > (1.0+self.tolerance) * expected_killed:
                    self.threshold *= 0.5

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()

                    if self.death_mode == 'threshold':
                        total_regrowth = math.floor((total_removed/float(expected_killed))*name2regrowth[name]*self.growth_death_ratio)
                    elif self.redistribution_mode == 'none':
                        if name not in self.name2baseline_nonzero:
                            self.name2baseline_nonzero[name] = self.name2nonzeros[name]
                        old = self.name2baseline_nonzero[name]
                        new = new_mask.sum().item()
                        #print(old, new)
                        total_regrowth = int(old-new)
                    elif self.death_mode == 'global_magnitude':
                        expected_removed = self.baseline_nonzero*self.name2death_rate[name]
                        expected_vs_actual = total_removed/expected_removed
                        total_regrowth = math.floor(expected_vs_actual*name2regrowth[name]*self.growth_death_ratio)
                    else:
                        total_regrowth = math.floor(name2regrowth[name]*self.growth_death_ratio)

                    # growth
                    if self.growth_mode == 'random':
                        new_mask = self.random_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'momentum':
                        new_mask = self.momentum_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'gradient':
                        # implementation for Rigging Ticket
                        new_mask, grad = self.gradient_growth(name, new_mask, self.pruning_rate[name], weight)

                    elif self.growth_mode == 'momentum_neuron':
                        new_mask = self.momentum_neuron_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'mix_growth':
                        new_mask = self.mix_growth(name, new_mask, total_regrowth, weight)

                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()
                    total_nonzero_new += new_nonzero
        self.apply_mask()

        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (death-growth) residuals to adjust future growth
        # self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        # self.adjusted_growth = 0.25*self.adjusted_growth + (0.75*(self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)
        # print(self.total_nonzero, self.baseline_nonzero, self.adjusted_growth)

        if self.total_nonzero > 0:
            print('old, new nonzero count:', self.total_nonzero, total_nonzero_new, self.adjusted_growth)

    '''
                    REDISTRIBUTION
    '''

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                if self.redistribution_mode == 'momentum':
                    grad = self.get_momentum_for_weight(tensor)
                    self.name2variance[name] = torch.abs(grad[mask.byte()]).mean().item()#/(V1val*V2val)
                elif self.redistribution_mode == 'magnitude':
                    self.name2variance[name] = torch.abs(tensor)[mask.byte()].mean().item()
                elif self.redistribution_mode == 'nonzeros':
                    self.name2variance[name] = float((torch.abs(tensor) > self.threshold).sum().item())
                elif self.redistribution_mode == 'none':
                    self.name2variance[name] = 1.0
                elif self.redistribution_mode == 'magnitude_increase':
                    # only calculate the increased weights
                    mask_increased = torch.abs(tensor) > torch.abs(self.pre_tensor[name])
                    # weights_increased = (torch.abs(tensor) - torch.abs(self.pre_tensor[name])).mean().item()
                    # print(name, "Weight increased:", weights_increased)
                    # include all the non-zero weights
                    self.name2variance[name] = (torch.abs(tensor[mask_increased.byte()]) - torch.abs(self.pre_tensor[name][mask_increased.byte()])).mean().item()
                    # self.name2variance[name] = torch.abs(tensor[mask.byte()] - self.pre_tensor[name][mask.byte()]).mean().item()
                    # print("name", name, "abs_MI",self.name2variance[name])# mean of ABS of magnitude increased weights
                    # print("abs_M",torch.abs(tensor[mask.byte()] - self.pre_tensor[name][mask.byte()]).mean().item())  # mean() of absolute of all weights magnitude increased
                elif self.redistribution_mode == 'uniform_distribution':
                    self.name2variance[name] = 1
                else:
                    print('Unknown redistribution mode:{0}'.format(self.redistribution_mode))
                    raise Exception('Unknown redistribution mode!')

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                death_rate = self.name2death_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        death_rate = min(sparsity, death_rate)
                num_remove = math.ceil(death_rate*self.name2nonzeros[name])
                self.total_removed += num_remove
                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]

    def calc_growth_redistribution(self):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0
        for name in self.name2variance:
            self.name2variance[name] /= self.total_variance

        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        expected_var = 1.0/len(self.name2variance)
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                #death_rate = min(self.name2death_rate[name], max(0.05, (self.name2zeros[name]/float(self.masks[name].numel()))))
                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                death_rate = self.name2death_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        death_rate = min(sparsity, death_rate)
                num_remove = math.ceil(death_rate*self.name2nonzeros[name])
                #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
                num_nonzero = self.name2nonzeros[name]
                num_zero = self.name2zeros[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(self.name2variance[name]*(self.total_removed+self.adjusted_growth))
                regrowth += mean_residual

                #if regrowth > max_regrowth:
                #    name2regrowth[name] = max_regrowth
                if regrowth > 0.99*max_regrowth:
                    name2regrowth[name] = 0.99*max_regrowth
                    residual += regrowth - name2regrowth[name]
                else:
                    name2regrowth[name] = regrowth
            if len(name2regrowth) == 0: mean_residual = 0
            else:
                mean_residual = residual / len(name2regrowth)
            i += 1

        if i == 1000:
            print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))

        return name2regrowth


    '''
                    DEATH
    '''
    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.name2death_rate[name] * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask



    def kernel_pruning(self, mask, weight, name):

        score = torch.clone(weight.grad * weight).detach().abs_()

        num_remove = math.ceil(self.name2death_rate[name] * self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])

        num_zeros = self.name2zeros[name]
        x, idx = torch.sort(score.data.view(-1))
        k = math.ceil(num_zeros + num_remove)
        mask.data.view(-1)[idx[:k]] = 0.0
        return mask

    def magnitude_death_NM(self, mask, weight, pruning_rate):
        if (mask==0).sum().item() == 0: return mask, None
        length = weight.numel()
        group = int(length / self.args.M)
        weight_temp = weight.abs().view(group, self.args.M)

        num_zero_group = (weight_temp[0][:] == 0).sum().item()
        num_nonzero_group = weight_temp[0].numel() - (weight_temp[0][:] == 0).sum().item()

        num_remove = math.ceil(pruning_rate * num_nonzero_group)
        k = num_zero_group + num_remove
        idx = torch.argsort(weight_temp, dim=1)[:, :k]
        new_mask = mask.view(group, self.args.M).scatter_(dim=1, index=idx, value=0.0).reshape(weight.shape)

        return new_mask, num_remove

    def magnitude_death(self, mask, weight, name):
        sparsity = self.name2zeros[name]/float(self.masks[name].numel())
        death_rate = self.name2death_rate[name]
        if sparsity < 0.2:
            expected_variance = 1.0/len(list(self.name2variance.keys()))
            actual_variance = self.name2variance[name]
            expected_vs_actual = expected_variance/actual_variance
            if expected_vs_actual < 1.0:
                death_rate = min(sparsity, death_rate)
                print(name, expected_variance, actual_variance, expected_vs_actual, death_rate)
        num_remove = math.ceil(death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]
        num_nonzero = n-num_zeros

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)

    def global_magnitude_death(self):
        death_rate = 0.0
        for name in self.name2death_rate:
            if name in self.masks:
                death_rate = self.name2death_rate[name]
        tokill = math.ceil(death_rate*self.baseline_nonzero)
        total_removed = 0
        prev_removed = 0
        while total_removed < tokill*(1.0-self.tolerance) or (total_removed > tokill*(1.0+self.tolerance)):
            total_removed = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    remain = (torch.abs(weight.data) > self.threshold).sum().item()
                    total_removed += self.name2nonzeros[name] - remain

            if prev_removed == total_removed: break
            prev_removed = total_removed
            if total_removed > tokill*(1.0+self.tolerance):
                self.threshold *= 1.0-self.increment
                self.increment *= 0.99
            elif total_removed < tokill*(1.0-self.tolerance):
                self.threshold *= 1.0+self.increment
                self.increment *= 0.99

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name][:] = torch.abs(weight.data) > self.threshold

        return int(total_removed)


    def global_momentum_growth(self, total_regrowth):
        togrow = total_regrowth
        total_grown = 0
        last_grown = 0
        while total_grown < togrow*(1.0-self.tolerance) or (total_grown > togrow*(1.0+self.tolerance)):
            total_grown = 0
            total_possible = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    new_mask = self.masks[name]
                    grad = self.get_momentum_for_weight(weight)
                    grad = grad*(new_mask==0).float()
                    possible = (grad !=0.0).sum().item()
                    total_possible += possible
                    grown = (torch.abs(grad.data) > self.growth_threshold).sum().item()
                    total_grown += grown
            print(total_grown, self.growth_threshold, togrow, self.growth_increment, total_possible)
            if total_grown == last_grown: break
            last_grown = total_grown


            if total_grown > togrow*(1.0+self.tolerance):
                self.growth_threshold *= 1.02
                #self.growth_increment *= 0.95
            elif total_grown < togrow*(1.0-self.tolerance):
                self.growth_threshold *= 0.98
                #self.growth_increment *= 0.95

        total_new_nonzeros = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                new_mask = self.masks[name]
                grad = self.get_momentum_for_weight(weight)
                grad = grad*(new_mask==0).float()
                self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > self.growth_threshold)).float()
                total_new_nonzeros += new_mask.sum().item()
        return total_new_nonzeros


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
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

    def random_growth(self, name, new_mask, total_regrowth, weight):
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability #lsw
        # new_weights = torch.rand(new_mask.shape) < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def kernel_gradient_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.grads[name]
        grad = grad * (new_mask == 0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth_NM(self, name, new_mask, num_regrow, weight):
        if (new_mask == 0).sum().item() == 0: return new_mask
        grad = self.get_gradient_for_weights(weight)
        grad[new_mask != 0] = 0.0

        length = weight.numel()
        group = int(length / self.args.M)
        grad = grad.view(group, self.args.M)

        idx = torch.argsort(grad.abs(), dim=1, descending=True)[:, :num_regrow]

        new_mask = new_mask.view(group, self.args.M).scatter_(dim=1, index=idx, value=1).reshape(weight.shape)

        return new_mask


    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
        # if self.args.grow_switch == 'switch':
        #     max_value = float(torch.max(torch.abs(weight)))
        #     # num_positive_grad = (grad[new_mask - self.masks[name].data.byte()] > 0).sum().item()
        #     # num_negative_grad = (grad[new_mask - self.masks[name].data.byte()] < 0).sum().item()
        #     # print('total regrow:', num_positive_grad+num_negative_grad)
        #     # print('before:', (weight != 0).sum().item())
        #     with torch.no_grad():
        #         weight[(new_mask - self.masks[name].data.byte()) > 0] = torch.sign(grad[(new_mask - self.masks[name].data.byte()) > 0]) * torch.FloatTensor(int(total_regrowth)).uniform_(0, max_value).cuda()
        #         # weight[(grad > 0) & ((new_mask - self.masks[name].data.byte()) > 0)] = torch.FloatTensor(int(num_positive_grad)).uniform_(0, 0).cuda()
        #         # weight[(grad < 0) & ((new_mask - self.masks[name].data.byte()) > 0)] = torch.FloatTensor(int(num_negative_grad)).uniform_(0, 0).cuda()
        #     # print('after:', (weight != 0).sum().item())
        return new_mask, grad

    def mix_growth(self, name, new_mask, total_regrowth, weight):
        gradient_grow = int(total_regrowth * self.args.mix)
        random_grow = total_regrowth - gradient_grow
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (new_mask == 0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:gradient_grow]] = 1.0

        n = (new_mask == 0).sum().item()
        expeced_growth_probability = (random_grow / n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask = new_mask.bool() | new_weights

        return new_mask

    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
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
                print('Death rate: {0}\n'.format(self.name2death_rate[name]))
                break

    def reset_momentum(self):
        """
        Taken from: https://github.com/AlliedToasters/synapses/blob/master/synapses/SET_layer.py
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                weights = list(self.optimizer.state[tensor])
                for w in weights:
                    if w == 'momentum_buffer':
                        # momentum
                        if self.args.reset_mom_zero:
                            print('zero')
                            self.optimizer.state[tensor][w][mask == 0] = 0
                        else:
                            print('mean')
                            self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])
                        # self.optimizer.state[tensor][w][mask==0] = 0
                    elif w == 'square_avg' or \
                        w == 'exp_avg' or \
                        w == 'exp_avg_sq' or \
                        w == 'exp_inf':
                        # Adam
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])

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