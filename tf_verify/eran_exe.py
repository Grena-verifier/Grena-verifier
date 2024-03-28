import sys
import os
import csv
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import torch
import numpy as np
from eran import ERAN
from read_net_file import *
from read_zonotope_file import read_zonotope
import tensorflow as tf
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
from tensorflow_translator import *
from onnx_translator import *
from optimizer import *
from analyzer import *
from pprint import pprint
# if config.domain=='gpupoly' or config.domain=='refinegpupoly':
from refine_gpupoly import *
from utils import parse_vnn_lib_prop, translate_output_constraints, translate_input_to_box, negate_cstr_or_list_old

#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

is_tf_version_2=tf.__version__[0]=='2'

if is_tf_version_2:
    tf= tf.compat.v1


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


def acasxu_recursive(specLB, specUB, max_depth=10, depth=0):
    hold,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
    global failed_already
    if hold:
        print_progress(depth)
        return hold, None
    elif depth >= max_depth:
        if failed_already.value and config.complete:
            try:
                verified_flag, adv_examples, _ = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
            except Exception as ex:
                print(f"{ex}Exception occured for the following inputs:")
                print(specLB, specUB, max_depth, depth)
                #verified_flag, adv_examples, _ = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                raise ex
            print_progress(depth)
            found_adex = False
            if verified_flag == False:
                if adv_examples!=None:
                    #print("adv image ", adv_image)
                    for adv_image in adv_examples:
                        for or_list in constraints:
                            if found_adex: break
                            negated_cstr = negate_cstr_or_list_old(or_list)
                            hold_adex,_,nlb,nub,_,_ = eran.analyze_box(adv_image, adv_image, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, negated_cstr)
                            found_adex = hold_adex or found_adex
                        #print("hold ", hold, "domain", domain)
                        if found_adex:
                            print("property violated at ", adv_image, "output_score", nlb[-1])
                            failed_already.value = 0
                            break
            return verified_flag, None if not found_adex else adv_image
        else:
            return False, None
    else:
        # grads = estimate_grads(specLB, specUB, input_shape=eran.input_shape)
        # # grads + small epsilon so if gradient estimation becomes 0 it will divide the biggest interval.
        # smears = np.multiply(grads + 0.00001, [u-l for u, l in zip(specUB, specLB)])

        #start = time.time()
        nn.set_last_weights(constraints)
        grads_lower, grads_upper = nn.back_propagate_gradient(nlb, nub)
        smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]

        index = np.argmax(smears)
        m = (specLB[index]+specUB[index])/2

        result_a, adex_a = acasxu_recursive(specLB, [ub if i != index else m for i, ub in enumerate(specUB)], max_depth, depth + 1)
        if adex_a is None:
            result_b, adex_b = acasxu_recursive([lb if i != index else m for i, lb in enumerate(specLB)], specUB, max_depth, depth + 1)
        else:
            adex_b = None
            result_b = False
        adex = adex_a if adex_a is not None else (adex_b if adex_b is not None else None)
        return failed_already.value and result_a and result_b, adex



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

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
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

#if len(sys.argv) < 4 or len(sys.argv) > 5:
#    print('usage: python3.6 netname epsilon domain dataset')
#    exit(1)

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
domain = config.domain
dataset = config.dataset
mean = 0
std = 0
complete = (config.complete==True)
print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
model, is_conv = read_onnx_net(netname)
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
os.sched_setaffinity(0,cpu_affinity)
correctly_classified_images = 0
verified_images = 0
unsafe_images = 0
cum_time = 0
constraints = None
if dataset and config.input_box is None:
    tests = get_tests(dataset, config.geometric)
    
def init(args):
    global failed_already
    failed_already = args
    
net_name_list = netname.split("/")
net_file = net_name_list[-1]
correct_list = []        
count = 0
for i, test in enumerate(tests):
    image= np.float64(test[1:len(test)])/np.float64(255)
    specLB = np.copy(image)
    specUB = np.copy(image)
    normalize(specLB, means, stds, dataset)
    normalize(specUB, means, stds, dataset)
    is_correctly_classified = False
    label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
    print("concrete ", nlb[-1])
    if label == int(test[0]):
        is_correctly_classified = True
    if is_correctly_classified == True:
        status = "Failed"
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
        prop = -1
        start = time.time()
        perturbed_label, _, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, "deeppoly",
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
        end = time.time()
        layer_list = eran.operations
        if(perturbed_label==label):
            print("Verified for image", i)
            status = "Verified"
            count = count + 1
        else:
            print("Failed for image", i)
        print(end-start)
        assert len(layer_list) == len(nlb), "The layer list doesn't match with the computed concrete bounds"
        # for j, type in enumerate(layer_list):
        #     if type == 'Relu':
        #         # compute the number of stable (activated, inactivated) and unstable
        #         act, deact, unstable = 0, 0, 0
        #         lbs, ubs = nlb[j], nub[j]
        #         for k in range(len(lbs)):
        #             if lbs[k] > 0.0:
        #                 act = act + 1
        #             elif ubs[k] == 0.0:
        #                 deact = deact + 1
        #             else:
        #                 unstable = unstable + 1
        #         assert deact + act + unstable == len(lbs), "The total amount doesn't match!"
        #         lb_fullpath = "ERAN_bench.csv"
        #         with open(lb_fullpath, 'a+', newline='') as write_obj:
        #             csv_writer = csv.writer(write_obj)
        #             csv_writer.writerow([net_file, "img "+str(i)+" with eps="+str(epsilon), "Layer "+str(j), len(lbs), act, deact, unstable, status])
                
print(count, "/", len(correct_list))