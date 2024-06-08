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


import pickle
from elina_abstract0 import *
from elina_manager import *
from fconv import generate_sparse_cover
from deeppoly_nodes import *
from deepzono_nodes import *
from krelu import *
from functools import reduce
from ai_milp import milp_callback
from typing import List, Literal, Union
from torch import Tensor, nn
# from ml-part-time.src.compare_against_gurobi import compare_against_gurobi
from ml_bound_solver.src.preprocessing.solver_inputs import SolverInputs
from ml_bound_solver.src.preprocessing.preprocessing_utils import remove_first_n_modules
from ml_bound_solver.src.solve import solve
from ml_bound_solver.src.utils import load_onnx_model
# from ml-part-time.src.utils import seed_everything

import gc
import torch
import cdd
import numpy as np
import os.path
import logging


def dump_solver_inputs(
    lbounds: List[np.ndarray],
    ubounds: List[np.ndarray],

    Pall: List[np.ndarray],
    Phatall: List[np.ndarray],
    smallpall: List[np.ndarray],

    Hmatrix: np.ndarray,
    dvector: np.ndarray,

    save_filename: str = 'dump.pth'
) -> None:
    """
    Saves the provided solver input parameters to a Pytorch `.pth` file
    specified by the `save_filename` parameter.
    """
    # Check that the variable types are correct.
    list_error_msg = "`{var}` is not type `list[np.ndarray]`."
    array_error_msg =  "`{var}` is not type `np.ndarray`."
    assert isinstance(lbounds, List) and all(isinstance(item, np.ndarray) for item in lbounds), list_error_msg.format(var='lbounds')
    assert isinstance(ubounds, List) and all(isinstance(item, np.ndarray) for item in ubounds), list_error_msg.format(var='ubounds')
    assert isinstance(Pall, List) and all(isinstance(item, np.ndarray) for item in Pall), list_error_msg.format(var='Pall')
    assert isinstance(Phatall, List) and all(isinstance(item, np.ndarray) for item in Phatall), list_error_msg.format(var='Phatall')
    assert isinstance(smallpall, List), array_error_msg.format(var='smallpall')
    assert isinstance(Hmatrix, np.ndarray), array_error_msg.format(var='Hmatrix')
    assert isinstance(dvector, np.ndarray), array_error_msg.format(var='dvector')    

    # Convert numpy arrays to PyTorch tensors and then into the formats used in the solver.
    L: List[Tensor] = [torch.tensor(x).float().squeeze() for x in lbounds]
    U: List[Tensor] = [torch.tensor(x).float().squeeze() for x in ubounds]
    H: Tensor = torch.tensor(Hmatrix).float().squeeze()
    d: Tensor = torch.tensor(dvector).float().squeeze()
    P: List[Tensor] = [torch.tensor(x).float().squeeze() for x in Pall]
    P_hat: List[Tensor] = [torch.tensor(x).float().squeeze() for x in Phatall]
    p: List[Tensor] = [torch.tensor(x).float().squeeze() for x in smallpall]

    # Save to file.
    torch.save({
        'L': L,
        'U': U,
        'H': H,
        'd': d,
        'P': P,
        'P_hat': P_hat,
        'p': p,
    }, save_filename)

# def wrap_SolverInputs(
#     lbounds: List[np.ndarray],
#     ubounds: List[np.ndarray],

#     Pall: List[np.ndarray],
#     Phatall: List[np.ndarray],
#     smallpall: List[np.ndarray],

#     Hmatrix: np.ndarray,
#     dvector: np.ndarray,

#     ground_truth_neuron_index: int,
#     model: nn.Module
# ) -> SolverInputs:
#     """
#     wrap up the set of inputs as class SolverInputs defined
#     """
#     # Convert numpy arrays to PyTorch tensors and then into the formats used in the solver.
#     L: List[Tensor] = [torch.tensor(x).float().squeeze() for x in lbounds]
#     U: List[Tensor] = [torch.tensor(x).float().squeeze() for x in ubounds]
#     H: Tensor = torch.tensor(Hmatrix).float().squeeze()
#     d: Tensor = torch.tensor(dvector).float().squeeze()
#     P: List[Tensor] = [torch.tensor(x).float().squeeze() for x in Pall]
#     P_hat: List[Tensor] = [torch.tensor(x).float().squeeze() for x in Phatall]
#     p: List[Tensor] = [torch.tensor(x).float().squeeze() for x in smallpall]

#     # Save to class SolverInputs.
#     solver_inputs = SolverInputs(
#         model=model,
#         ground_truth_neuron_index=ground_truth_neuron_index,
#         L_list = L,
#         U_list = U,
#         H = H,
#         d = d,
#         P_list = P,
#         P_hat_list = P_hat,
#         p_list = p,
#     )
#     return solver_inputs

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.filters = []
        self.numfilters = []
        self.filter_size = [] 
        self.input_shape = []
        self.strides = []
        self.padding = []
        self.out_shapes = []
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.pad_counter = 0
        self.pool_counter = 0
        self.concat_counter = 0
        self.tile_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.lastlayer = None
        self.last_weights = None
        self.label = -1
        self.prop = -1

    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.pool_counter + self.activation_counter + self.concat_counter + self.tile_counter +self.pad_counter

    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    def set_last_weights(self, constraints):
        length = 0.0       
        last_weights = [0 for weights in self.weights[-1][0]]
        for or_list in constraints:
            for (i, j, cons) in or_list:
                if j == -1:
                    last_weights = [l + w_i + float(cons) for l,w_i in zip(last_weights, self.weights[-1][i])]
                else:
                    last_weights = [l + w_i + w_j + float(cons) for l,w_i, w_j in zip(last_weights, self.weights[-1][i], self.weights[-1][j])]
                length += 1
        self.last_weights = [w/length for w in last_weights]


    def back_propagate_gradient(self, nlb, nub):
        #assert self.is_ffn(), 'only supported for FFN'

        grad_lower = self.last_weights.copy()
        grad_upper = self.last_weights.copy()
        last_layer_size = len(grad_lower)
        for layer in range(len(self.weights)-2, -1, -1):
            weights = self.weights[layer]
            lb = nlb[layer]
            ub = nub[layer]
            layer_size = len(weights[0])
            grad_l = [0] * layer_size
            grad_u = [0] * layer_size

            for j in range(last_layer_size):

                if ub[j] <= 0:
                    grad_lower[j], grad_upper[j] = 0, 0

                elif lb[j] <= 0:
                    grad_upper[j] = grad_upper[j] if grad_upper[j] > 0 else 0
                    grad_lower[j] = grad_lower[j] if grad_lower[j] < 0 else 0

                for i in range(layer_size):
                    if weights[j][i] >= 0:
                        grad_l[i] += weights[j][i] * grad_lower[j]
                        grad_u[i] += weights[j][i] * grad_upper[j]
                    else:
                        grad_l[i] += weights[j][i] * grad_upper[j]
                        grad_u[i] += weights[j][i] * grad_lower[j]
            last_layer_size = layer_size
            grad_lower = grad_l
            grad_upper = grad_u
        return grad_lower, grad_upper


class Analyzer:
    def __init__(self, ir_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label,
                 prop, testing = False, K=3, s=-2, timeout_final_lp=100, timeout_final_milp=100, use_milp=False,
                 complete=False, partial_milp=False, max_milp_neurons=30, approx_k=True, GRENA=False,
                 use_wralu: Union[None, Literal["sci", "sciplus", "sciall"]] = None):
        """
        Arguments
        ---------
        ir_list: list
            list of Node-Objects (e.g. from DeepzonoNodes), first one must create an abstract element
        domain: str
            either 'deepzono', 'refinezono' or 'deeppoly'
        """
        self.ir_list = ir_list
        self.is_greater = is_greater_zono
        self.refine = False
        if domain == 'deeppoly' or domain == 'refinepoly':
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
        elif domain == 'deepzono' or domain == 'refinezono':
            self.man = zonoml_manager_alloc()
            self.is_greater = is_greater_zono
        if domain == 'refinezono' or domain == 'refinepoly':
            self.refine = True
        self.domain = domain
        self.nn = nn
        self.timeout_lp = timeout_lp
        self.timeout_milp = timeout_milp
        self.timeout_final_lp = timeout_final_lp
        self.timeout_final_milp = timeout_final_milp
        self.use_milp = use_milp
        self.output_constraints = output_constraints
        self.use_default_heuristic = use_default_heuristic
        self.testing = testing
        self.relu_groups = []
        self.label = label
        self.prop = prop
        self.complete = complete
        self.GRENA = GRENA
        self.K=K
        self.s=s
        self.partial_milp=partial_milp
        self.max_milp_neurons=max_milp_neurons
        self.approx_k = approx_k
        self.use_wralu = use_wralu

    def __del__(self):
        elina_manager_free(self.man)
        
    def get_abstract0(self):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        element = self.ir_list[0].transformer(self.man)
        nlb = []
        nub = []
        testing_nlb = []
        testing_nub = []
        for i in range(1, len(self.ir_list)):
            if type(self.ir_list[i]) in [DeeppolyReluNode,DeeppolySigmoidNode,DeeppolyTanhNode,DeepzonoRelu,DeepzonoSigmoid,DeepzonoTanh]:
                element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub,
                                                                  self.relu_groups, 'refine' in self.domain,
                                                                  self.timeout_lp, self.timeout_milp,
                                                                  self.use_default_heuristic, self.testing,
                                                                  K=self.K, s=self.s, use_milp=self.use_milp,
                                                                  approx=self.approx_k, use_wralu=self.use_wralu)
            else:
                element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub,
                                                                  self.relu_groups, 'refine' in self.domain,
                                                                  self.timeout_lp, self.timeout_milp,
                                                                  self.use_default_heuristic, self.testing)

            if self.testing and isinstance(element_test_bounds, tuple):
                element, test_lb, test_ub = element_test_bounds
                testing_nlb.append(test_lb)
                testing_nub.append(test_ub)
            else:
                element = element_test_bounds
        if self.domain in ["refinezono", "refinepoly"]:
            gc.collect()
        if self.testing:
            return element, testing_nlb, testing_nub
        return element, nlb, nub
    
    def analyze(self,terminate_on_failure=True):
        """
        analyses the network with the given input
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        element, nlb, nub = self.get_abstract0()
        
        # if self.domain == "deeppoly" or self.domain == "refinepoly":
        #     linexprarray = backsubstituted_expr_for_layer(self.man, element, 1, True)
        #     for neuron in range(1):
        #         print("******EXPR*****")
        #         elina_linexpr0_print(linexprarray[neuron],None)
        #         print()
        # output_size = 0
        if self.domain == 'deepzono' or self.domain == 'refinezono':
            output_size = self.ir_list[-1].output_length
        else:
            output_size = self.ir_list[-1].output_length#reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)
        
        dominant_class = -1
        if(self.domain=='refinepoly'):

            #relu_needed = [1] * self.nn.numlayer
            self.nn.ffn_counter = 0
            self.nn.conv_counter = 0
            self.nn.pool_counter = 0
            self.nn.pad_counter = 0
            self.nn.concat_counter = 0
            self.nn.tile_counter = 0
            self.nn.residual_counter = 0
            self.nn.activation_counter = 0
            counter, var_list, model = create_model(self.nn, self.nn.specLB, self.nn.specUB, nlb, nub, self.relu_groups, self.nn.numlayer, self.complete)
            if self.partial_milp != 0:
                self.nn.ffn_counter = 0
                self.nn.conv_counter = 0
                self.nn.pool_counter = 0
                self.nn.pad_counter = 0
                self.nn.concat_counter = 0
                self.nn.tile_counter = 0
                self.nn.residual_counter = 0
                self.nn.activation_counter = 0
                counter_partial_milp, var_list_partial_milp, model_partial_milp = create_model(self.nn, self.nn.specLB,
                                                                                               self.nn.specUB, nlb, nub,
                                                                                               self.relu_groups,
                                                                                               self.nn.numlayer,
                                                                                               self.complete,
                                                                                               partial_milp=self.partial_milp,
                                                                                               max_milp_neurons=self.max_milp_neurons)
                model_partial_milp.setParam(GRB.Param.TimeLimit, self.timeout_final_milp)

            if self.complete:
                model.setParam(GRB.Param.TimeLimit, self.timeout_final_milp)
            else:
                model.setParam(GRB.Param.TimeLimit, self.timeout_final_lp)

            model.setParam(GRB.Param.Cutoff, 0.01)  # Indicates that you aren't interested in solutions whose objective values are worse than the specified value. If the objective value for the optimal solution is equal to or better than the specified cutoff, the solver will return the optimal solution. Otherwise, it will terminate with a CUTOFF status.


            num_var = len(var_list)
            output_size = num_var - counter

        label_failed = []
        x = None
        if self.output_constraints is None:
            
            candidate_labels = []
            if self.label == -1:
                for i in range(output_size):
                    candidate_labels.append(i)
            else:
                candidate_labels.append(self.label)

            adv_labels = []
            if self.prop == -1:
                for i in range(output_size):
                    adv_labels.append(i)
            else:
                adv_labels.append(self.prop)   

            for label in candidate_labels:
                flag = True
                for adv_label in adv_labels:
                    if self.domain == 'deepzono' or self.domain == 'refinezono':
                        if label == adv_label:
                            continue
                        elif self.is_greater(self.man, element, label, adv_label):
                            continue
                        else:
                            flag = False
                            label_failed.append(adv_label)
                            if terminate_on_failure:
                                break
                    else:
                        if label == adv_label:
                            continue
                        elif self.is_greater(self.man, element, label, adv_label, self.use_default_heuristic):
                            continue
                        else:
                            if(self.domain=='refinepoly'):
                                obj = LinExpr()
                                obj += 1 * var_list[counter + label]
                                obj += -1 * var_list[counter + adv_label]
                                model.setObjective(obj, GRB.MINIMIZE)
                                if self.complete:
                                    model.optimize(milp_callback)
                                    if not hasattr(model,"objbound") or model.objbound <= 0:
                                        flag = False
                                        if self.label != -1:
                                            label_failed.append(adv_label)
                                        if model.solcount > 0:
                                            x = model.x[0:len(self.nn.specLB)]
                                        if terminate_on_failure:
                                            break
                                else:
                                    model.optimize(lp_callback)
                                    # model.optimize()
                                    # if model.Status == 11:
                                    #     model.optimize() #very rarely lp_callback seems to leave model in interrupted state

                                    try:
                                        print(
                                            f"Model status: {model.Status}, Objval against label {adv_label}: {model.objval:.4f}, Final solve time: {model.Runtime:.3f}")
                                    except:
                                        print(
                                            f"Model status: {model.Status}, Objval retrival failed, Final solve time: {model.Runtime:.3f}")

                                    if model.Status == 6 or (model.Status == 2 and model.objval > 0):
                                        # Cutoff active, or optimal with positive objective => sound against adv_label
                                        pass
                                    elif self.partial_milp != 0:
                                        obj = LinExpr()
                                        obj += 1 * var_list_partial_milp[counter_partial_milp + label]
                                        obj += -1 * var_list_partial_milp[counter_partial_milp + adv_label]
                                        model_partial_milp.setObjective(obj, GRB.MINIMIZE)
                                        model_partial_milp.optimize(milp_callback)
                                        try:
                                            print(
                                                f"Partial MILP model status: {model_partial_milp.Status}, Objbound against label {adv_label}: {model_partial_milp.ObjBound:.4f}, Final solve time: {model_partial_milp.Runtime:.3f}")
                                        except:
                                            print(
                                                f"Partial MILP model status: {model_partial_milp.Status}, Objbound retrival failed, Final solve time: {model_partial_milp.Runtime:.3f}")

                                        if model_partial_milp.Status in [2,9,11] and model_partial_milp.ObjBound > 0:
                                            pass
                                        elif model_partial_milp.Status not in [2,9,11]:
                                            print("Partial milp model was not successful status is", model_partial_milp.Status)
                                            model_partial_milp.write("final.mps")
                                            flag = False
                                        else:
                                            flag = False
                                    elif model.Status != 2:
                                        print("Model was not successful status is",
                                              model.Status)
                                        model.write("final.mps")
                                        flag = False
                                    else:
                                        flag = False
                                    if flag and model.Status==2 and model.objval < 0:
                                        if model.objval != math.inf:
                                            x = model.x[0:len(self.nn.specLB)]

                            else:
                                flag = False
                    if not flag:
                        if terminate_on_failure:
                            break
                        elif self.label != -1:
                            label_failed.append(adv_label)
                if flag:
                    dominant_class = label
                    break
        else:
            # AND
            dominant_class = True
            for or_list in self.output_constraints:
                # OR
                or_result = False
                
                for is_greater_tuple in or_list:
                    if is_greater_tuple[1] == -1:
                        if nub[-1][is_greater_tuple[0]] < float(is_greater_tuple[2]):
                            or_result = True
                            break
                    else: 
                        if self.domain == 'deepzono' or self.domain == 'refinezono':
                            if self.is_greater(self.man, element, is_greater_tuple[0], is_greater_tuple[1]):
                                or_result = True
                                break
                        else:
                            if self.is_greater(self.man, element, is_greater_tuple[0], is_greater_tuple[1], self.use_default_heuristic):
                                or_result = True
                                break

                if not or_result:
                    dominant_class = False
                    break
        elina_abstract0_free(self.man, element)
        return dominant_class, nlb, nub, label_failed, x

    def obtain_output_bound(self, element):
        layerno = len(self.nn.layertypes) -1
        length = get_num_neurons_in_layer(self.man, element, layerno)
        bounds = box_for_layer(self.man, element, layerno)
        itv = [bounds[i] for i in range(length)]
        nlb = [x.contents.inf.contents.val.dbl for x in itv]
        nub = [x.contents.sup.contents.val.dbl for x in itv]
        elina_interval_array_free(bounds,length)
        lbi = np.asarray(nlb, dtype=np.double)
        ubi = np.asarray(nub, dtype=np.double)      
        print(lbi)
        print(ubi)
        return lbi, ubi

    def analyze_poly(
        self,
        terminate_on_failure=True,
        ground_truth_label=-1,
        IOIL_lbs=None,
        IOIL_ubs=None,
        multi_prune=3,
        onnx_path = None,
        bounds_save_path: str = "dump.pkl",
    ):
        """
        analyses the network with the given input
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        assert ground_truth_label!=-1, "The ground truth label cannot be -1!!!!!!!!!!!!!Please pass the correct parameter"
        # self.K=0
        # self.s=0 # remove the prima execution for this execution 
        print("ONNX network path is", onnx_path)
        element, nlb, nub = self.get_abstract0()
        # print("end krelu??")
        output_size = self.ir_list[-1].output_length
        dominant_class = -1
        label_failed = []
        x = None
        candidate_labels = []
        adv_labels = []
        for i in range(output_size):
            adv_labels.append(i)

        label = ground_truth_label # just one gt label in our scenerio

        final_adv_labels = {}
        for adv_label in adv_labels:
            if label == adv_label:
                continue
            else:
                lb = label_deviation_lb(self.man, element, label, adv_label) # if the gt label already dominant
                if lb < 0:
                    final_adv_labels[adv_label] = lb
                else:
                    continue
        # print(final_adv_labels)  
        sorted_adv = sorted(final_adv_labels.items(), key = lambda x:x[1], reverse=False)
        # print(sorted_adv)
        final_adv_labels = [adv[0] for adv in sorted_adv]
        if self.GRENA == True:
            # run the abstract refinement process
            last_iter = 5
            pruned_labels = []
            if(len(final_adv_labels) != 0):
                # generate krelu constraints
                P_allayer_ori, Phat_allayer_ori, smallp_allayer_ori, relu_groups_ori = self.generate_krelu_cons(element, full_vars=False)
                while(len(final_adv_labels)>0):
                    if(last_iter > 2):
                        pending_num = 1
                    else:
                        pending_num = min(multi_prune, len(final_adv_labels))
                    print("Handle adv labels:", final_adv_labels[:pending_num])
                    solStatus, last_iter = self.eliminate_adv_labels(final_adv_labels[:pending_num], pending_num, ground_truth_label, pruned_labels, element, nlb, nub, IOIL_lbs, IOIL_ubs, P_allayer_ori, Phat_allayer_ori, smallp_allayer_ori, relu_groups_ori, output_size, onnx_path)
                    print("solStatus, last_iter are:", solStatus, last_iter)
                    if(solStatus == -1):
                        print("Falsified!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        break
                    elif(solStatus == 0):
                        print("Unknown!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        break
                    elif(solStatus == 1):
                        print("Adv labels", final_adv_labels[:pending_num], "pruned")
                        for item in final_adv_labels[:pending_num]:
                            pruned_labels.append(item)
                            final_adv_labels.remove(item)
            else:
                print("Verified!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if(len(final_adv_labels) == 0):
                print("Verified with pruning!!!!!!!!!!!!!!!!!!!!!!!!")
                dominant_class = ground_truth_label
        else:
            # only conduct solving comparison
            print("final_adv_labels", final_adv_labels)
            start_time = time.time()
            # P_allayer, Phat_allayer, smallp_allayer are constraint representation sent to tailored solver
            # relu_groups is the constraint respresentation sent to gurobi solver
            P_allayer, Phat_allayer, smallp_allayer, relu_groups = self.generate_krelu_cons(element, full_vars=False)
            # assert len(relu_groups) > 0 and all(x is not None for x in relu_groups)
            # logging.critical(f"num of krelu constraints: {sum(y.cons.shape[0] for x in relu_groups for y in x)}")
            execution_time = time.time() - start_time
            logging.critical(f"Generate constraints: {execution_time:.5f}s")
            Hmatrix, dvector = self.obtain_output_cons_cddlib(final_adv_labels[:1], 1, ground_truth_label, [], element, len(self.nn.layertypes)-1)
            # solve gurobi bounds
            start_list, var_list, model = build_gurobi_model(self.nn, self.nn.specLB, self.nn.specUB, nlb, nub, relu_groups, self.nn.numlayer, Hmatrix, dvector, output_size)
            model.optimize()
            iter = 0
            if(model.Status != 3):
                start_time = time.time()
                gurobi_lbs, gurobi_ubs = self.solve_neuron_bounds_gurobi(model, var_list, start_list, element, True, bounds_save_path)
                execution_time = time.time() - start_time
                logging.critical(f"Gurobi: {execution_time:.5f} seconds")
                torch_model, input_shape = load_onnx_model(onnx_path, return_input_shape=True)
                # print(torch_model)
                if('conv' in onnx_path):
                    remove_first_n_modules(torch_model, 4)  # Add this line to remove norm layers.
                solver_inputs = SolverInputs(
                    model=torch_model,
                    input_shape=input_shape,
                    ground_truth_neuron_index=ground_truth_label,
                    L_list=IOIL_lbs,
                    U_list=IOIL_ubs,
                    H=Hmatrix,
                    d=dvector,
                    P_list=P_allayer,
                    P_hat_list=Phat_allayer,
                    p_list=smallp_allayer,
                )
                # for index in range(len(IOIL_lbs)):
                #     print(f'lenght of IOIL_lbs and IOIL_ubs are {IOIL_lbs[index].shape} {IOIL_ubs[index].shape}')
                # for index in range(len(P_allayer)):
                #     print(f'shape of P_allayer, Phat_allayer, smallp_allayer are {P_allayer[index].shape}, {Phat_allayer[index].shape}, {smallp_allayer[index].shape}')
                start_time = time.time()
                is_falsified, tailored_lbs, tailored_ubs, _ = solve(
                                                            solver_inputs,
                                                            device=torch.device("cuda"),
                                                            return_solver=True,
                                                        )
                execution_time = time.time() - start_time
                write_self_lbs = [tailored_lb.reshape(-1) for tailored_lb in tailored_lbs]
                write_self_ubs = [tailored_ub.reshape(-1) for tailored_ub in tailored_ubs]
                # bounds_save_path
                with open(bounds_save_path, 'wb') as file:
                    pickle.dump({
                        'tailored_lbs': tailored_lbs,
                        'tailored_ubs': tailored_ubs,
                        'gurobi_lbs': gurobi_lbs,
                        'gurobi_ubs': gurobi_ubs,
                        'IOIL_lbs': IOIL_lbs,
                        'IOIL_ubs': IOIL_ubs,
                    }, file)
                logging.critical(f"Tailored solver: {execution_time:.5f} seconds")
            else:
                print("Infeasible!!!!!!")
        elina_abstract0_free(self.man, element)
        return dominant_class, nlb, nub, label_failed, x
    
    def solve_neuron_bounds_gurobi(self, model, var_list, start_list, element, full_vars = False, bounds_save_path: str = "dump.pkl"):
        ### resolve input bounds
        length = start_list[1]
        print("input dimension is", length)
        gurobi_lbs, gurobi_ubs = [], []
        lbs, ubs = np.zeros(length, dtype = np.float64), np.zeros(length, dtype = np.float64)
        for i in range(length):
            obj = LinExpr()
            obj += 1 * var_list[i]
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            if(model.Status == 3):
                return None, None # means verified
            lbs[i] = model.objbound

            model.setObjective(obj, GRB.MAXIMIZE)
            model.optimize()
            if(model.Status == 3):
                return None, None # means verified
            ubs[i] = model.objbound
        gurobi_lbs.append(lbs)
        gurobi_ubs.append(ubs)

        ### resolve intermediate layers that are RELU INPUTS
        for i in range(1, len(start_list)):
            layerno = i - 1
            if self.nn.layertypes[layerno] in ['SkipCat']:
                continue # do nothing
            elif(self.nn.layertypes[layerno]=='ReLU'):
                # handle the input of ReLU layer
                input_layerno = layerno - 1
                length = get_num_neurons_in_layer(self.man, element, input_layerno)
                bounds = box_for_layer(self.man, element, input_layerno)
                itv = [bounds[j] for j in range(length)]
                lb = [x.contents.inf.contents.val.dbl for x in itv]
                ub = [x.contents.sup.contents.val.dbl for x in itv]
                unstable_index = [j for j in range(length) if lb[j] < 0 and ub[j] > 0]
                elina_interval_array_free(bounds,length)
                solve_neuron_count = length if full_vars else len(unstable_index)
                lbs, ubs = np.zeros(solve_neuron_count, dtype = np.float64), np.zeros(solve_neuron_count, dtype = np.float64)
                neuron_index = range(length) if full_vars else unstable_index
                for index, j in enumerate(neuron_index):
                    obj = LinExpr()
                    obj += 1 * var_list[j+start_list[i-1]] # i-1 is the input layer index
                    model.setObjective(obj, GRB.MINIMIZE)
                    model.optimize()
                    lbs[index] = model.objbound

                    model.setObjective(obj, GRB.MAXIMIZE)
                    model.optimize()
                    ubs[index] = model.objbound
                gurobi_lbs.append(lbs)
                gurobi_ubs.append(ubs)
            else:
                continue

        return gurobi_lbs, gurobi_ubs

    def obtain_curDP_bounds_and_update_IOIL(self, element, input_lbs, input_ubs):
        nlb, nub = [], []
        new_lbs, new_ubs = [input_lbs], [input_ubs]
        for layerno, layertype in enumerate(self.nn.layertypes):
            length = get_num_neurons_in_layer(self.man, element, layerno)
            bounds = box_for_layer(self.man, element, layerno)
            itv = [bounds[i] for i in range(length)]
            lb = [x.contents.inf.contents.val.dbl for x in itv]
            ub = [x.contents.sup.contents.val.dbl for x in itv]
            if layertype == 'ReLU':
                ### add up the last layer's bounds
                new_lbs.append(np.array(nlb[-1]))
                new_ubs.append(np.array(nub[-1]))
            nlb.append(lb)
            nub.append(ub)
            elina_interval_array_free(bounds,length)
        return nlb, nub, new_lbs, new_ubs

    def eliminate_adv_labels(self, multi_label_list, num_multi, gt_label, pruned_labels, element, nlb_ori, nub_ori, IOIL_lbs, IOIL_ubs, P_allayer_ori, Phat_allayer_ori, smallp_allayer_ori, relu_groups_ori, output_num = 10, onnx_path = None):
        # revert back to oringal deeppoly abstract domain
        start = time.time()
        clear_neurons_status(self.man, element)
        run_deeppoly(self.man, element)
        P_allayer, Phat_allayer, smallp_allayer, relu_groups = P_allayer_ori, Phat_allayer_ori, smallp_allayer_ori, relu_groups_ori
        Hmatrix, dvector = self.obtain_output_cons_cddlib(multi_label_list, num_multi, gt_label, pruned_labels, element, len(self.nn.layertypes)-1, output_num)
        # uses GUROBI to check SAT
        _, _, model = build_gurobi_model(self.nn, self.nn.specLB, self.nn.specUB, nlb_ori, nub_ori, relu_groups, self.nn.numlayer, Hmatrix, dvector, output_num)
        # self.solve_neuron_bounds_gurobi(model, var_list, start_list, element)
        # check SAT
        model.optimize()
        iter = 0
        if(model.Status == 3):
            return 1, iter # means infeasible, thus verified
        else:
            while(model.Status != 3 and iter <= 5):
                falsified = False
                print("Start iter", iter)
                torch_model = load_onnx_model(onnx_path)
                len_unstable_vars = []
                for i in range(1, len(IOIL_lbs)):
                    len_unstable_vars.append(len([j for j in range(len(IOIL_lbs[i])) if IOIL_lbs[i][j] < 0 and IOIL_ubs[i][j] > 0]))
                print("len_unstable_vars for all intermediate layers before solving", len_unstable_vars)
                
                torch_model, input_shape = load_onnx_model(onnx_path, return_input_shape=True)
                # print(torch_model)
                remove_first_n_modules(torch_model, 4)  # Add this line to remove norm layers.
                solver_inputs = SolverInputs(
                    model=torch_model,
                    input_shape=input_shape,
                    ground_truth_neuron_index=gt_label,
                    L_list=IOIL_lbs,
                    U_list=IOIL_ubs,
                    H=Hmatrix,
                    d=dvector,
                    P_list=P_allayer,
                    P_hat_list=Phat_allayer,
                    p_list=smallp_allayer,
                )
                is_falsified, tailored_lbs, tailored_ubs, _ = solve(
                                                            solver_inputs,
                                                            device=torch.device("cuda"),
                                                            return_solver=True,
                                                        )
                if(is_falsified):
                    return -1, iter
                IOIL_lbs = [tailored_lb.reshape(-1) for tailored_lb in tailored_lbs]
                IOIL_ubs = [tailored_ub.reshape(-1) for tailored_ub in tailored_ubs]

                len_unstable_vars = []
                for i in range(1, len(IOIL_lbs)):
                    len_unstable_vars.append(len([j for j in range(len(IOIL_lbs[i])) if IOIL_lbs[i][j] < 0 and IOIL_ubs[i][j] > 0]))
                print("len_unstable_vars for all intermediate layers after solving", len_unstable_vars)
                flat_lbs = np.asarray([item for sublist in IOIL_lbs for item in sublist], dtype=np.float64)
                flat_ubs = np.asarray([item for sublist in IOIL_ubs for item in sublist], dtype=np.float64)
                num_each_layer = [len(item) for item in IOIL_lbs]
                print(f"one iteration time is:{time.time()-start}")
                update_bounds_from_LPsolve(self.man, element, len(IOIL_lbs), num_each_layer, flat_lbs, flat_ubs)
                assert self.domain == "refinepoly"
                P_allayer, Phat_allayer, smallp_allayer, relu_groups = self.generate_krelu_cons(element)
                # rerun deeppoly with new bounds will reduce other neurons as well, need to reformulate IOIL_lbs/ubs with current results
                nlb, nub, IOIL_lbs, IOIL_ubs = self.obtain_curDP_bounds_and_update_IOIL(element, IOIL_lbs[0], IOIL_ubs[0])
                _, _, model = build_gurobi_model(self.nn, IOIL_lbs[0], IOIL_ubs[0], nlb, nub, relu_groups, self.nn.numlayer, Hmatrix, dvector, output_num)
                model.optimize()
                print("End iter", iter)
                iter += 1
            if(model.Status == 3):
                return 1, iter
            else:
                return 0, iter

    def obtain_output_cons_cddlib(self, multi_label_list, num_multi, gt_label, pruned_labels, element, output_index, output_num = 10):
        # obtain input Octagon for this case
        pruned_num = len(pruned_labels)
        fullvar = [gt_label]
        if(num_multi == 1):
            Hmatrix, dvector = np.zeros((1+pruned_num, output_num), dtype = np.float64), np.zeros(1+pruned_num, dtype = np.float64)
            Hmatrix[0][gt_label], Hmatrix[0][multi_label_list] = 1, -1
            length = 1
        else:
            fullvar.extend(multi_label_list)
            coeffs_list = []
            print(fullvar)
            size = 3**(num_multi+1) - 1
            linexpr0 = elina_linexpr0_array_alloc(size)
            for coeffs in itertools.product([-1, 0, 1], repeat=num_multi+1):
                if all(c == 0 for c in coeffs):
                    continue
                else:
                    coeffs_list.append(coeffs)
            for i, coeffs in enumerate(coeffs_list):
                linexpr0[i] = generate_linexpr0(0, fullvar, coeffs)
            upper_bound = get_upper_bound_for_linexpr0(self.man, element, linexpr0, size, output_index)
            '''For index i, we have constraint linexpr0[i] (which is fullvar*coeffs) <= upper_bound[i]'''
            # for i, coeffs in enumerate(coeffs_list):
            #     print("coeff is", coeffs, ";", 'ubound is', upper_bound[i])

            inputM = []
            for i, coeffs in enumerate(coeffs_list):
                inputM.append([upper_bound[i]]+[-c for c in coeffs])
            # InitM = cdd.Matrix(inputM, number_type = 'float')
            InitM = cdd.Matrix(inputM, number_type = 'fraction')
            InitM.rep_type = cdd.RepType.INEQUALITY
            merge_Vlist = []
            for i in range(num_multi):
                M = InitM.copy()
                new = [0, -1] + [1 if x==i else 0 for x in range(num_multi)]
                M.extend([new])
                poly = cdd.Polyhedron(M)
                # convert to V representation
                ext = poly.get_generators()
                for j in range(ext.row_size):
                    row = ext.__getitem__(j)
                    if(row[0] == 0):
                        break
                    else:
                        merge_Vlist.append(row)
            # collect all vertex and reduce redundance
            # VMatrix = cdd.Matrix(merge_Vlist, number_type = 'float')
            VMatrix = cdd.Matrix(merge_Vlist, number_type = 'fraction')
            VMatrix.rep_type = cdd.RepType.GENERATOR
            res = VMatrix.canonicalize() # this will directly eliminate those duplicate vertex
            finalPoly = cdd.Polyhedron(VMatrix)
            inequs = finalPoly.get_inequalities()
            # print(inequs)
            # print(inequs.NumberType)
            # the total number of output constraints
            length = inequs.row_size 
            total_len = length + len(pruned_labels)
            Hmatrix, dvector = np.zeros((total_len, output_num), dtype = np.float64), np.zeros(total_len, dtype = np.float64)
            for j in range(length):
                row = inequs.__getitem__(j)
                dvector[j] = -row[0]
                # print("row is",row)
                Hmatrix[j][fullvar] = [-c for c in row[1:]]
        for j in range(pruned_num):
            dvector[j+length] = 0
            Hmatrix[j+length][gt_label] = -1
            Hmatrix[j+length][pruned_labels[j]] = 1
        # dump into file
        # dump_tensors_to_file([Hmatrix], 'Hmatrix')
        # dump_tensors_to_file([dvector], 'dvector')
        # print(Hmatrix, dvector)
        return Hmatrix, dvector

    @staticmethod
    def index_grouping(grouplen: int, K: int, step: int = 2) -> List[List[int]]:
        return [
            list(range(i, i + K))
                for i in range(0, grouplen - K + 1, step)
        ]

    def relu_grouping(self, length, lb, ub):
        assert length == len(lb) == len(ub)

        all_vars = [i for i in range(length) if lb[i] < 0 < ub[i]]
        areas = {var: -lb[var] * ub[var] for var in all_vars}
        unstable_vars = all_vars
        assert len(all_vars) == len(areas)
        sparse_n = config.sparse_n
        cutoff = 0.05
        # Sort vars by descending area
        all_vars = sorted(all_vars, key=lambda var: -areas[var])

        vars_above_cutoff = [i for i in all_vars if areas[i] >= cutoff]
        n_vars_above_cutoff = len(vars_above_cutoff)

        kact_args = []
        print("len(vars_above_cutoff)", len(vars_above_cutoff))
        if len(vars_above_cutoff) >= self.K and config.sparse_n >= self.K:
            grouplen = min(sparse_n, len(vars_above_cutoff))
            # print(grouplen)
            group = vars_above_cutoff[:grouplen]
            vars_above_cutoff = vars_above_cutoff[grouplen:]
            if grouplen <= self.K:
                kact_args.append(group)
            elif self.K>2:
                sparsed_combs = self.index_grouping(grouplen, self.K)
                for comb in sparsed_combs:
                    kact_args.append(tuple([group[i] for i in comb]))
            elif self.K==2:
                raise RuntimeError("K=2 is not supported")

        # Also just apply 1-relu for every var.
        # for var in all_vars:
        #     kact_args.append([var])

        # print("krelu: n", config.sparse_n,
        #     "split_zero", len(all_vars),
        #     "after cutoff", n_vars_above_cutoff,
        #     "number of args", len(kact_args))
        # print("number of args", len(kact_args))
        return kact_args, unstable_vars

    def generate_krelu_cons(self, element, full_vars = False):
        # obtain krelu constraints from multi-relu feature
        P_allayer_list = []
        Phat_allayer_list = []
        smallp_allayer_list = []
        groupNum_each_layer = []
        relu_groups = []
        act_layer_indexes = [i for i, x in enumerate(self.nn.layertypes) if x == "ReLU"]
        for _, act_layer in enumerate(act_layer_indexes): # handle each activation layer
            layerno = act_layer - 1  # get the pre-activation
            length = get_num_neurons_in_layer(self.man, element, layerno)
            bounds = box_for_layer(self.man, element, layerno)
            itv = [bounds[i] for i in range(length)]
            nlb = [x.contents.inf.contents.val.dbl for x in itv]
            nub = [x.contents.sup.contents.val.dbl for x in itv]
            elina_interval_array_free(bounds,length)
            lbi = np.asarray(nlb, dtype=np.double)
            ubi = np.asarray(nub, dtype=np.double)
            kact_args, unstable_vars = self.relu_grouping(length, lbi, ubi)
            # print("unstable_vars from prima", len(unstable_vars))
            if(len(kact_args) >= 1):
                # generate krelu constraints if we have relu group
                tdim = ElinaDim(length)
                KAct.man = self.man
                KAct.element = element
                KAct.tdim = tdim
                KAct.length = length
                KAct.layerno = layerno
                KAct.offset = 0
                KAct.domain = self.domain
                KAct.type = "ReLU"
                total_size = 0    
                for varsid in kact_args:
                    size = 3**len(varsid) - 1
                    total_size = total_size + size
                linexpr0 = elina_linexpr0_array_alloc(total_size)
                i = 0
                # generate input Octagon
                for varsid in kact_args:
                    for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                        if all(c == 0 for c in coeffs):
                            continue
                        linexpr0[i] = generate_linexpr0(0, varsid, coeffs)
                        i = i + 1
                upper_bound = get_upper_bound_for_linexpr0(self.man,element,linexpr0, total_size, layerno)
                i=0
                input_hrep_array, lb_array, ub_array = [], [], []
                for varsid in kact_args:
                    input_hrep = []
                    for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                        if all(c == 0 for c in coeffs):
                            continue
                        input_hrep.append([upper_bound[i]] + [-c for c in coeffs])
                        i = i + 1
                    input_hrep_array.append(input_hrep)
                    lb_array.append([lbi[varid] for varid in varsid])
                    ub_array.append([ubi[varid] for varid in varsid])
                # call function and obtain krelu constraints
                with multiprocessing.Pool(config.numproc) as pool:
                    num_inputs = len(input_hrep_array)
                    kact_results = pool.starmap(make_kactivation_obj, zip(input_hrep_array, lb_array, ub_array, [self.approx_k] * num_inputs, [self.use_wralu] * num_inputs))
                relu_groups.append(kact_results)
                gid = 0
                cons_count = 0
                for inst in kact_results:
                    varsid = kact_args[gid]
                    inst.varsid = varsid
                    gid += 1
                    cons_count += len(inst.cons)
                # declare P, Phat and small p
                if(full_vars):
                    Pmatrix, Phat, smallp = np.zeros((cons_count, length), dtype = np.float64), np.zeros((cons_count, length), dtype = np.float64), np.zeros(cons_count, dtype = np.float64)
                else:
                    unstable_count = len(unstable_vars)
                    Pmatrix, Phat, smallp = np.zeros((cons_count, unstable_count), dtype = np.float64), np.zeros((cons_count, unstable_count), dtype = np.float64), np.zeros(cons_count, dtype = np.float64)
                    dictionary = dict(zip(unstable_vars, range(unstable_count)))
                    # print(dictionary)
                # assign elements in the matrix
                row_ptr = 0
                for inst in kact_results:
                    index_list = list(inst.varsid)
                    k = len(index_list)
                    if(full_vars == False):
                        index_list = [dictionary[varid] for varid in index_list]
                    for row in inst.cons:
                        smallp[row_ptr] = row[0]
                        Pmatrix[row_ptr][index_list] = -row[1: 1+k]
                        Phat[row_ptr][index_list] = -row[k+1: 1+2*k]
                        row_ptr+=1    
                P_allayer_list.append(Pmatrix)
                Phat_allayer_list.append(Phat)
                smallp_allayer_list.append(smallp)
            else:
                num_unstable = len(unstable_vars)
                P_allayer_list.append(np.zeros((1, num_unstable)))
                Phat_allayer_list.append(np.zeros((1, num_unstable)))
                smallp_allayer_list.append(np.zeros((1,)))
                relu_groups.append(None)
            # record the number of 3-relu group per layer
            groupNum_each_layer.append(len(kact_args))
            # print("constriant number is", row_ptr)

        # check the data type
        # for i in range(len(P_allayer_list)):
        #     if(P_allayer_list[i] is not None):
        #         print("shape checking", P_allayer_list[i].shape, Phat_allayer_list[i].shape, smallp_allayer_list[i].shape)
        #     else:
        #         print("No KReLU for ", i, "th ReLU layer")
        # export the matrix in dump.py
        # dump_tensors_to_file(P_allayer_list, 'Pall')
        # dump_tensors_to_file(Phat_allayer_list, 'Phatall')
        # dump_tensors_to_file(smallp_allayer_list, 'smallpall')
        # print(groupNum_each_layer)
        return P_allayer_list, Phat_allayer_list, smallp_allayer_list, relu_groups
          