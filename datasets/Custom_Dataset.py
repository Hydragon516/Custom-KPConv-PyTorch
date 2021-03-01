import time
import numpy as np
import pickle
import torch
import math

import os
from os import listdir
from os.path import exists, join
from torch.utils.data import Dataset

from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import PointCloudDataset
from datasets.common import grid_subsampling

from tqdm import tqdm
import config.config as config


class CustomDataset(PointCloudDataset):
    def __init__(self, train=True, subsampling=False, first_subsampling_dl=0.02):
        PointCloudDataset.__init__(self, 'Custom')

        self.label_to_names = {0: 'Cyclist',
                               1: 'Pedestrian',
                               2: 'Car'
                               }

        self.first_subsampling_dl = first_subsampling_dl
        self.subsampling = subsampling

        # Initialize a bunch of variables concerning class labels
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        self.path = config.DATA['root']
        self.train = train

        # Number of models and models used per epoch
        if self.train:
            self.num_models = 9843
            if config.SETTING['epoch_steps'] and config.SETTING['epoch_steps'] * config.SETTING['batch_num'] < self.num_models:
                self.epoch_n = config.SETTING['epoch_steps'] * config.SETTING['batch_num']
            else:
                self.epoch_n = self.num_models
        else:
            self.num_models = 2468
            self.epoch_n = min(self.num_models, config.SETTING['validation_size'] * config.SETTING['batch_num'])

        self.input_points, self.input_normals, self.input_labels = self.load_subsampled_clouds(orient_correction=True)

        return

    def __len__(self):
        return self.num_models

    def __getitem__(self, idx_list):
        tp_list = []
        tn_list = []
        tl_list = []
        ti_list = []
        s_list = []
        R_list = []

        for p_i in idx_list:

            # Get points and labels
            points = self.input_points[p_i].astype(np.float32)
            normals = self.input_normals[p_i].astype(np.float32)
            label = self.label_to_idx[self.input_labels[p_i]]

            # Data augmentation
            points, normals, scale, R = self.augmentation_transform(points, normals)

            # Stack batch
            tp_list += [points]
            tn_list += [normals]
            tl_list += [label]
            ti_list += [p_i]
            s_list += [scale]
            R_list += [R]

        stacked_points = np.concatenate(tp_list, axis=0)
        stacked_normals = np.concatenate(tn_list, axis=0)
        labels = np.array(tl_list, dtype=np.int64)
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if config.SETTING['in_features_dim'] == 1:
            pass
        elif config.SETTING['in_features_dim'] == 4:
            stacked_features = np.hstack((stacked_features, stacked_normals))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        # Get the whole input list
        input_list = self.classification_inputs(stacked_points,
                                                stacked_features,
                                                labels,
                                                stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, model_inds]

        return input_list

    def load_subsampled_clouds(self, orient_correction):
        train_list_path = []
        if self.train is True:
            train_list_file = open(os.path.join(self.path, "train.txt"), 'r')
        else:
            train_list_file = open(os.path.join(self.path, "test.txt"), 'r')

        while True:
            line = train_list_file.readline()
            if not line: 
                break
            train_list_path.append(line)
        
        train_list_file.close()

        # Initialize containers
        input_points = []
        input_normals = []
        label_names = []

        # Collect point clouds
        for i, txt_file_path in enumerate(tqdm(train_list_path)):
            # Read points
            txt_file_path = txt_file_path.replace("\n", "")
            
            class_name = txt_file_path.split("/")[-2:-1]
            label_names.append(class_name[0])
            
            data = np.loadtxt(txt_file_path, delimiter=',', dtype=np.float32)
            
            # Subsample them
            if self.subsampling is True:
                points, normals = grid_subsampling(data[:, :3], features=data[:, 3:], sampleDl=self.first_subsampling_dl)
            else:
                points = data[:, :3]
                normals = data[:, 3:]

            # Add to list
            input_points += [points]
            input_normals += [normals]

        # Get labels
        input_labels = np.array([self.name_to_label[name] for name in label_names])

        if orient_correction:
            input_points = [pp[:, [0, 2, 1]] for pp in input_points]
            input_normals = [nn[:, [0, 2, 1]] for nn in input_normals]

        return input_points, input_normals, input_labels


class CustomDataSampler(Sampler):
    def __init__(self, dataset: CustomDataset, use_potential=True, balance_labels=False):
        Sampler.__init__(self, dataset)

        # Does the sampler use potential for regular sampling
        self.use_potential = use_potential

        # Should be balance the classes when sampling
        self.balance_labels = balance_labels

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Create potentials
        if self.use_potential:
            self.potentials = np.random.rand(len(dataset.input_labels)) * 0.1 + 0.1
        else:
            self.potentials = None

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = 10000

        return

    def __iter__(self):
        """
        Yield next batch indices here
        """

        ##########################################
        # Initialize the list of generated indices
        ##########################################

        if self.use_potential:
            if self.balance_labels:

                gen_indices = []
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                for i, l in enumerate(self.dataset.label_values):

                    # Get the potentials of the objects of this class
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]
                    class_potentials = self.potentials[label_inds]

                    # Get the indices to generate thanks to potentials
                    if pick_n < class_potentials.shape[0]:
                        pick_indices = np.argpartition(class_potentials, pick_n)[:pick_n]
                    else:
                        pick_indices = np.random.permutation(class_potentials.shape[0])
                    class_indices = label_inds[pick_indices]
                    gen_indices.append(class_indices)

                # Stack the chosen indices of all classes
                gen_indices = np.random.permutation(np.hstack(gen_indices))

            else:

                # Get indices with the minimum potential
                if self.dataset.epoch_n < self.potentials.shape[0]:
                    gen_indices = np.argpartition(self.potentials, self.dataset.epoch_n)[:self.dataset.epoch_n]
                else:
                    gen_indices = np.random.permutation(self.potentials.shape[0])
                gen_indices = np.random.permutation(gen_indices)

            # Update potentials (Change the order for the next epoch)
            self.potentials[gen_indices] = np.ceil(self.potentials[gen_indices])
            self.potentials[gen_indices] += np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1

        else:
            if self.balance_labels:
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                gen_indices = []
                for l in self.dataset.label_values:
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]
                    rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                    gen_indices += [rand_inds]
                gen_indices = np.random.permutation(np.hstack(gen_indices))
            else:
                gen_indices = np.random.permutation(self.dataset.num_models)[:self.dataset.epoch_n]

        ################
        # Generator loop
        ################

        # Initialize concatenation lists
        ti_list = []
        batch_n = 0

        # Generator loop
        for p_i in gen_indices:

            # Size of picked cloud
            n = self.dataset.input_points[p_i].shape[0]

            # In case batch is full, yield it and reset it
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            # Add data to current batch
            ti_list += [p_i]

            # Update batch size
            batch_n += n

        yield np.array(ti_list, dtype=np.int32)

        return 0

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return None

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = False

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        key = '{:.3f}_{:d}'.format(config.SETTING['first_subsampling_dl'],
                                   config.SETTING['batch_num'])
        if key in batch_lim_dict:
            self.batch_limit = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                v = str(int(batch_lim_dict[key]))
            else:
                v = '?'

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(config.SETTING['num_layers']):
            dl = config.SETTING['first_subsampling_dl'] * (2**layer_ind)
            if (config.SETTING['deform_layers'])[layer_ind]:
                r = dl * config.SETTING['deform_radius']
            else:
                r = dl * config.SETTING['conv_radius']

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == config.SETTING['num_layers']:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(config.SETTING['num_layers']):
                dl = config.SETTING['first_subsampling_dl'] * (2**layer_ind)
                if (config.SETTING['deform_layers'])[layer_ind]:
                    r = dl * config.SETTING['deform_radius']
                else:
                    r = dl * config.SETTING['conv_radius']
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    v = str(neighb_lim_dict[key])
                else:
                    v = '?'

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (config.SETTING['conv_radius'] + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((config.SETTING['num_layers'], hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = config.SETTING['batch_num']

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.labels)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            key = '{:.3f}_{:d}'.format(config.SETTING['first_subsampling_dl'],
                                       config.SETTING['batch_num'])
            batch_lim_dict[key] = self.batch_limit
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(config.SETTING['num_layers']):
                dl = config.SETTING['first_subsampling_dl'] * (2 ** layer_ind)
                if (config.SETTING['deform_layers'])[layer_ind]:
                    r = dl * config.SETTING['deform_radius']
                else:
                    r = dl * config.SETTING['conv_radius']
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class ModelNet40CustomBatch:
    """Custom batch definition with memory pinning for ModelNet40"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 5) // 4

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.model_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.model_inds = self.model_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.model_inds = self.model_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def ModelNet40Collate(batch_data):
    return ModelNet40CustomBatch(batch_data)