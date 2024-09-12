"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import torch
import numpy as np
from eran import ERAN
from read_net_file import *
from read_zonotope_file import read_zonotope
# import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ai_milp import *
import argparse
from config import config
from constraint_utils import *
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
import spatial
from copy import deepcopy
# from tensorflow_translator import *
from onnx_translator import *
from optimizer import *
from analyzer import *
from pprint import pprint
# if config.domain=='gpupoly' or config.domain=='refinegpupoly':
from refine_gpupoly import *
from utils import parse_vnn_lib_prop, translate_output_constraints, translate_input_to_box, negate_cstr_or_list_old
import signal

#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname



def parse_input_box(text):
    intervals_list = []
    for line in text.split('\n'):
        if line!="":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb,ub] = interval.split(",")
                intervals.append((np.double(lb), np.double(ub)))
            intervals_list.append(intervals)

    # return every combination
    boxes = itertools.product(*intervals_list)
    return list(boxes)


def show_ascii_spec(lb, ub, n_rows, n_cols, n_channels):
    print('==================================================================')
    for i in range(n_rows):
        print('  ', end='')
        for j in range(n_cols):
            print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ', end='')
        for j in range(n_cols):
            print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ')
    print('==================================================================')


def normalize(image, means, stds, dataset):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds!=None:
                image[i] /= stds[i]
    elif dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        
        is_gpupoly = (domain=='gpupoly' or domain=='refinegpupoly')
        if is_conv and not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]
            #for i in range(1024):
            #    image[i*3] = tmp[i]
            #    image[i*3+1] = tmp[i+1024]
            #    image[i*3+2] = tmp[i+2048]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1


def normalize_plane(plane, mean, std, channel, is_constant):
    plane_ = plane.clone()

    if is_constant:
        plane_ -= mean[channel]

    plane_ /= std[channel]

    return plane_


def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    # normalization taken out of the network
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]


def denormalize(image, means, stds, dataset):
    if dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(3072):
            image[i] = tmp[i]


def model_predict(base, input):
    if is_onnx:
        pred = base.run(input)
    else:
        pred = base.run(base.graph.get_operation_by_name(model.op.name).outputs[0], {base.graph.get_operations()[0].name + ':0': input})
    return pred


def estimate_grads(specLB, specUB, dim_samples=3, input_shape=[1]):
    # Estimate gradients using central difference quotient and average over dim_samples+1 in the range of the input bounds
    # Very computationally costly
    specLB = np.array(specLB, dtype=np.float32)
    specUB = np.array(specUB, dtype=np.float32)
    inputs = [(((dim_samples - i) * specLB + i * specUB) / dim_samples).reshape(*input_shape) for i in range(dim_samples + 1)]
    diffs = np.zeros(len(specLB))

    # refactor this out of this method
    if is_onnx:
        runnable = rt.prepare(model, 'CPU')
    elif sess is None:
        config = tf.ConfigProto(device_count={'GPU': 0})
        runnable = tf.Session(config=config)
    else:
        runnable = sess

    for sample in range(dim_samples + 1):
        pred = model_predict(runnable, inputs[sample])

        for index in range(len(specLB)):
            if sample < dim_samples:
                l_input = [m if i != index else u for i, m, u in zip(range(len(specLB)), inputs[sample], inputs[sample+1])]
                l_input = np.array(l_input, dtype=np.float32)
                l_i_pred = model_predict(runnable, l_input)
            else:
                l_i_pred = pred
            if sample > 0:
                u_input = [m if i != index else l for i, m, l in zip(range(len(specLB)), inputs[sample], inputs[sample-1])]
                u_input = np.array(u_input, dtype=np.float32)
                u_i_pred = model_predict(runnable, u_input)
            else:
                u_i_pred = pred
            diff = np.sum([abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)])
            diffs[index] += diff
    return diffs / dim_samples



progress = 0.0
def print_progress(depth):
    if config.debug:
        global progress, rec_start
        progress += np.power(2.,-depth)
        sys.stdout.write('\r%.10f percent, %.02f s\n' % (100 * progress, time.time()-rec_start))

def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        if config.subset == None:
            try:
                csvfile = open('../data/{}_test_full.csv'.format(dataset), 'r')
            except:
                csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
                print("Only the first 100 samples are available.")
        else:
            filename = '../data/'+ dataset+ '_test_' + config.subset + '.csv'
            csvfile = open(filename, 'r')
    tests = csv.reader(csvfile, delimiter=',')

    return tests


def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out!")

signal.signal(signal.SIGALRM, timeout_handler)

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
parser.add_argument('--imgid', type=int, default=None, help='the single image id for execution')
parser.add_argument('--GRENA', type=str2bool, default=False, help='enable GRENA refinement process')
parser.add_argument('--timeout_AR', type=float, default=300, help='timeout (in seconds) for the abstract refinement process')
parser.add_argument('--multi_prune', type=int, default=1, help='enable GRENA refinement process')
parser.add_argument('--zonotope', type=str, default=config.zonotope, help='file to specify the zonotope matrix')
parser.add_argument('--subset', type=str, default=config.subset, help='suffix of the file to specify the subset of the test dataset to use')
parser.add_argument('--target', type=str, default=config.target, help='file specify the targets for the attack')
parser.add_argument('--epsfile', type=str, default=config.epsfile, help='file specify the epsilons for the L_oo attack')
parser.add_argument('--vnn_lib_spec', type=str, default=config.vnn_lib_spec, help='VNN_LIB spec file, defining input and output constraints')
parser.add_argument('--specnumber', type=int, default=config.specnumber, help='the property number for the acasxu networks')
parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly')
parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, acasxu, or fashion')
parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
parser.add_argument('--timeout_final_lp', type=float, default=config.timeout_final_lp,  help='timeout for the final LP solver')
parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
parser.add_argument('--timeout_final_milp', type=float, default=config.timeout_final_lp,  help='timeout for the final MILP solver')
parser.add_argument('--timeout_complete', type=float, default=None,  help='Cumulative timeout for the complete verifier, superseeds timeout_final_milp if set')
parser.add_argument('--max_milp_neurons', type=int, default=config.max_milp_neurons,  help='number of layers to encode using MILP.')
parser.add_argument('--partial_milp', type=int, default=config.partial_milp,  help='Maximum number of neurons to use for partial MILP encoding')

parser.add_argument('--numproc', type=int, default=config.numproc,  help='number of processes for MILP / LP / k-ReLU')
parser.add_argument('--sparse_n', type=int, default=config.sparse_n,  help='Number of variables to group by k-ReLU')
parser.add_argument('--use_default_heuristic', type=str2bool, default=config.use_default_heuristic,  help='whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation')
parser.add_argument('--use_milp', type=str2bool, default=config.use_milp,  help='whether to use milp or not')
parser.add_argument('--refine_neurons', action='store_true', default=config.refine_neurons, help='whether to refine intermediate neurons')
parser.add_argument('--n_milp_refine', type=int, default=config.n_milp_refine, help='Number of milp refined layers')
parser.add_argument('--mean', nargs='+', type=float, default=config.mean, help='the mean used to normalize the data with')
parser.add_argument('--std', nargs='+', type=float, default=config.std, help='the standard deviation used to normalize the data with')
parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data location')
parser.add_argument('--geometric_config', type=str, default=config.geometric_config, help='config location')
parser.add_argument('--num_params', type=int, default=config.num_params, help='Number of transformation parameters')
parser.add_argument('--num_tests', type=int, default=config.num_tests, help='Number of images to test')
parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
parser.add_argument('--debug', type=str2bool, default=config.debug, help='Whether to display debug info')
parser.add_argument('--attack', action='store_true', default=config.attack, help='Whether to attack')
parser.add_argument('--geometric', '-g', dest='geometric', default=config.geometric, action='store_true', help='Whether to do geometric analysis')
parser.add_argument('--input_box', default=config.input_box,  help='input box to use')
parser.add_argument('--output_constraints', default=config.output_constraints, help='custom output constraints to check')
parser.add_argument('--normalized_region', type=str2bool, default=config.normalized_region, help='Whether to normalize the adversarial region')
parser.add_argument('--spatial', action='store_true', default=config.spatial, help='whether to do vector field analysis')
parser.add_argument('--t-norm', type=str, default=config.t_norm, help='vector field norm (1, 2, or inf)')
parser.add_argument('--delta', type=float, default=config.delta, help='vector field displacement magnitude')
parser.add_argument('--gamma', type=float, default=config.gamma, help='vector field smoothness constraint')
parser.add_argument('--k', type=int, default=config.k, help='refine group size')
parser.add_argument('--s', type=int, default=config.s, help='refine group sparsity parameter')
parser.add_argument('--quant_step', type=float, default=config.quant_step, help='Quantization step for quantized networks')
parser.add_argument("--approx_k", type=str2bool, default=config.approx_k, help="Use approximate fast k neuron constraints")


# Logging options
parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')
parser.add_argument('--output_dir', type=str, default=config.output_dir, help='Directory of all output files')
parser.add_argument('--bounds_save_filename', type=str, default=config.bounds_save_filename, help='Save file path for Gurobi-solved bounds')
parser.add_argument('--use_wralu', type=str, default=config.use_wralu, help='Type of WraLU solver to use: "sci", "sciplus" or "sciall". If not specified, default to using original `fkrelu` solver (ie. don\'t use WraLU).')


args = parser.parse_args()
for k, v in vars(args).items():
    setattr(config, k, v)
# if args.timeout_complete is not None:
#     raise DeprecationWarning("'--timeout_complete' is depreciated. Use '--timeout_final_milp' instead")
config.json = vars(args)
pprint(config.json)

if config.specnumber and not config.input_box and not config.output_constraints:
    config.input_box = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_input_prenormalized.txt'
    config.output_constraints = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_constraints.txt'

assert config.netname, 'a network has to be provided for analysis.'

bounds_save_path = os.path.join(config.output_dir, config.bounds_save_filename)
bounds_save_path = os.path.abspath(bounds_save_path)
os.makedirs(config.output_dir, exist_ok=True)

import logging
logging.basicConfig(filename=os.path.join(os.path.dirname(bounds_save_path), "log"), level=logging.CRITICAL, format=f'use_wralu={config.use_wralu}, eps={config.epsilon}, sparse_n={config.sparse_n}, k={config.k}, s={config.s} - %(message)s')

netname = config.netname
assert os.path.isfile(netname), f"Model file not found. Please check \"{netname}\" is correct."
filename, file_extension = os.path.splitext(netname)

is_trained_with_pytorch = file_extension==".pyt"
is_saved_tf_model = file_extension==".meta"
is_pb_file = file_extension==".pb"
is_tensorflow = file_extension== ".tf"
is_onnx = file_extension == ".onnx"
assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

epsilon = config.epsilon
#assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

zonotope_file = config.zonotope
zonotope = None
zonotope_bool = (zonotope_file!=None)
if zonotope_bool:
    zonotope = read_zonotope(zonotope_file)

domain = config.domain

if zonotope_bool:
    assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
elif not config.geometric:
    assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly', 'gpupoly', 'refinegpupoly'], "domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly"

dataset = config.dataset

# if zonotope_bool==False:
#    assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"

mean = 0
std = 0

complete = (config.complete==True)


print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)

non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

sess = None
if is_saved_tf_model or is_pb_file:
    netfolder = os.path.dirname(netname)

    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.Session()
    if is_saved_tf_model:
        saver = tf.train.import_meta_graph(netname)
        saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
    else:
        with tf.gfile.GFile(netname, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.graph_util.import_graph_def(graph_def, name='')
    ops = sess.graph.get_operations()
    last_layer_index = -1
    while ops[last_layer_index].type in non_layer_operation_types:
        last_layer_index -= 1
    model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')

    eran = ERAN(model, sess)

else:
    if(zonotope_bool==True):
        num_pixels = len(zonotope)
    elif(dataset=='mnist'):
        num_pixels = 784
    elif (dataset=='cifar10'):
        num_pixels = 3072
    elif(dataset=='acasxu'):
        num_pixels = 5
    if is_onnx:
        model, is_conv = read_onnx_net(netname)
    else:
        model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch, (domain == 'gpupoly' or domain == 'refinegpupoly'))
    if domain == 'gpupoly' or domain == 'refinegpupoly':
        if is_onnx:
            translator = ONNXTranslator(model, True)
        else:
            translator = TFTranslator(model)
        operations, resources = translator.translate()
        optimizer = Optimizer(operations, resources)
        nn = layers()
        network, relu_layers, num_gpu_layers = optimizer.get_gpupoly(nn) 
    else:    
        eran = ERAN(model, is_onnx=is_onnx)

if not is_trained_with_pytorch:
    if dataset == 'mnist' and not config.geometric:
        means = [0]
        stds = [1]
    elif dataset == 'acasxu':
        means = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0]
        stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
    elif dataset == "cifar10":
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2023, 0.1994, 0.2010]
    else:
        means = [0.5, 0.5, 0.5]
        stds = [1, 1, 1]

is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

if config.mean is not None:
    means = config.mean
    stds = config.std

os.sched_setaffinity(0,cpu_affinity)

correctly_classified_images = 0
verified_images = 0
unsafe_images = 0
falsified_images = 0
cum_time = 0

if config.vnn_lib_spec is not None:
    # input and output constraints in homogenized representation x >= C_lb * [x_0, eps, 1]; C_out [y, 1] >= 0
    C_lb, C_ub, C_out = parse_vnn_lib_prop(config.vnn_lib_spec)
    constraints = translate_output_constraints(C_out)
    boxes = translate_input_to_box(C_lb, C_ub, x_0=None, eps=None, domain_bounds=None)
else:
    if config.output_constraints:
        constraints = get_constraints_from_file(config.output_constraints)
    else:
        constraints = None

    if dataset and config.input_box is None:
        tests = get_tests(dataset, config.geometric)
    else:
        tests = open(config.input_box, 'r').read()
        boxes = parse_input_box(tests)

def init(args):
    global failed_already
    failed_already = args

correct_list = []     
model_name = os.path.splitext(os.path.basename(config.netname))[0]
fullpath = f"GRENA_result_model={model_name}_eps={config.epsilon}.csv"
net_name_list = netname.split("/")
net_file = net_name_list[-1]

IOIL_lbs, IOIL_ubs = [], []

for i, test in enumerate(tests):
    if config.from_test and i < config.from_test:
        continue

    if config.num_tests is not None and i >= config.from_test + config.num_tests:
        break

    if args.imgid is not None and i is not args.imgid:
        continue
    elif args.imgid is not None and i == args.imgid:
        print("verify for single image id!!!!!!!!!!!!!!!!!", args.imgid)
        
    image= np.float64(test[1:len(test)])/np.float64(255)
    specLB = np.copy(image)
    specUB = np.copy(image)
    print("The normalization for this network is", means, stds)
    normalize(specLB, means, stds, dataset)
    normalize(specUB, means, stds, dataset)

    # check if the clean image can be classified correctly
    is_correctly_classified = False
    start = time.time()

    try:
        signal.alarm(int(args.timeout_AR))  # Start timeout timer

        if domain == 'gpupoly' or domain == 'refinegpupoly':
            is_correctly_classified = network.test(specLB, specUB, int(test[0]), True)
        else:
            label,nn,nlb,nub,_,_ = eran.analyze_box(
                specLB,
                specUB,
                init_domain(domain),
                config.timeout_lp,
                config.timeout_milp,
                config.use_default_heuristic,
                K=config.k,
                s=config.s,
            )
            print("concrete ", nlb[-1])
            if label == int(test[0]):
                is_correctly_classified = True
        # only conduct robustness verification of the correctly classified clean image
        if is_correctly_classified == True:
            status = "null"
            label = int(test[0])
            perturbed_label = None
            correctly_classified_images +=1
            correct_list.append(i)
            if config.normalized_region==True:
                specLB = np.clip(image - epsilon,0,1)
                specUB = np.clip(image + epsilon,0,1)
                normalize(specLB, means, stds, dataset)
                normalize(specUB, means, stds, dataset)
            else:
                specLB = specLB - epsilon
                specUB = specUB + epsilon

            if config.target == None:
                prop = -1

            # add input intervals
            IOIL_lbs.append(specLB)
            IOIL_ubs.append(specUB)


            print("label is", label)
            if domain == 'gpupoly' or domain =='refinegpupoly':
                assert False, "We disable this gpu branch!!!"
            else:
                if domain.endswith("poly"):
                    # refinepoly enters this place, to conduct first abstract interpretation
                    perturbed_label, nn, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                        config.timeout_lp,
                                                                                        config.timeout_milp,
                                                                                        config.use_default_heuristic,
                                                                                        label=label, prop=prop, K=0, s=0,
                                                                                        timeout_final_lp=config.timeout_final_lp,
                                                                                        timeout_final_milp=config.timeout_final_milp,
                                                                                        use_milp=False,
                                                                                        complete=False,
                                                                                        terminate_on_failure=not config.complete,
                                                                                        partial_milp=0,
                                                                                        max_milp_neurons=0,
                                                                                        approx_k=0)
                    print("perturbed_label is", perturbed_label)
                    # the nlb nub info already contains all the lower bounds and upper bounds
                    # OVERWRITE design IOIL_lbs to include only input and ReLU input bounds
                    for idx, layertype in enumerate(nn.layertypes):
                        if layertype == 'ReLU': 
                            IOIL_lbs.append(np.array(nlb[idx-1]))
                            IOIL_ubs.append(np.array(nub[idx-1]))

                    
                    print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)
                if not (perturbed_label==label):
                    # if initial deeppoly fails, also enter this domain to re-run with prima constraints
                    perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_with_gt(specLB, specUB, domain,
                                                                                        config.timeout_lp,
                                                                                        config.timeout_milp,
                                                                                        config.use_default_heuristic,
                                                                                        label=label, prop=prop,
                                                                                        K=config.k, s=config.s,
                                                                                        timeout_final_lp=config.timeout_final_lp,
                                                                                        timeout_final_milp=config.timeout_final_milp,
                                                                                        use_milp=config.use_milp,
                                                                                        complete=False,
                                                                                        terminate_on_failure=not config.complete,
                                                                                        partial_milp=config.partial_milp,
                                                                                        max_milp_neurons=config.max_milp_neurons,
                                                                                        approx_k=config.approx_k,
                                                                                        IOIL_lbs=IOIL_lbs,
                                                                                        IOIL_ubs=IOIL_ubs,
                                                                                        GRENA=config.GRENA,
                                                                                        multi_prune=config.multi_prune,
                                                                                        onnx_path=config.netname,
                                                                                        bounds_save_path=bounds_save_path,
                                                                                        use_wralu=config.use_wralu)
                    print("nlb ", nlb[-1], " nub ", nub[-1], "adv labels ", failed_labels)
                if (perturbed_label==label):
                    # verification succeeds
                    print("img", i, "Verified", label)
                    status = "Verified"
                    verified_images += 1
                else:
                    if (len(failed_labels)) > 0:
                        print("img", i, "Falsified")
                        status = "Falsified"
                        falsified_images += 1
                    else:
                        print("img", i, "Unknown")
                        status = "Unknown"
                        unsafe_images += 1
            end = time.time()
            cum_time += end - start # only count samples where we did try to certify
            with open(fullpath, 'a+', newline='') as write_obj:
                csv_writer = csv.writer(write_obj)
                csv_writer.writerow([net_file, str(dataset), "img "+str(i)+" with label "+str(int(test[0])), "eps="+str(epsilon), "GRENA", str(end - start)+" secs", status])
        else:
            print("img",i,"not considered, incorrectly classified")
            end = time.time()

    except TimeoutError:
        end = time.time()
        status = "Unknown"
        unsafe_images += 1
        cum_time += end - start
        with open(fullpath, 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow([net_file, str(dataset), "img "+str(i)+" with label "+str(int(test[0])), "eps="+str(epsilon), "GRENA", str(end - start)+" secs", status])
        try:
            print("img", i, "Time out with unknown result. label =", label)
        except NameError:
            print("img", i, 'Time out with unknown result. "label" variable has not been initialised yet.')
    finally:
        signal.alarm(0)  # Clear timeout timer


    print(f"progress: {1 + i - config.from_test}/{config.num_tests}, "
            f"correct:  {correctly_classified_images}/{1 + i - config.from_test}, "
            f"verified: {verified_images}/{correctly_classified_images}, "
            f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
            f"falsified: {falsified_images}/{correctly_classified_images}, ",
            f"time: {end - start:.3f}; {0 if cum_time==0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")
if (config.GRENA):
    with open(fullpath, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(["verified", str(verified_images)+'/'+str(correctly_classified_images)])
        csv_writer.writerow(["unsafe", str(unsafe_images)+'/'+str(correctly_classified_images)])
        csv_writer.writerow(["falsified", str(falsified_images)+'/'+str(correctly_classified_images)])
        csv_writer.writerow(["average time", str(cum_time / correctly_classified_images)])
print('analysis precision ',verified_images,'/ ', correctly_classified_images)
print('correct image list', correct_list)
