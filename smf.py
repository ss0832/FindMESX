import os
import sys
import glob
import copy
import time
import datetime
import psi4
import random
import math
import argparse
import itertools

from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import numpy as np
#This program is a prototype.
#It can be used to find minima of intersecting regions (intersecting seams). 

#SMF: J. Am. Chem. Soc. 2015, 137, 3433
"""
    FindMESX
    Copyright (C) 2023 ss0832

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
#please input psi4 inputfile.
(electronic charges) (spin multiply)
(element1) x y z
(element2) x y z
(element3) x y z
....
"""

"""
Psi4
 D. G. A. Smith, L. A. Burns, A. C. Simmonett, R. M. Parrish, M. C. Schieber, R. Galvelis, P. Kraus, H. Kruse, R. Di Remigio, A. Alenaizan, A. M. James, S. Lehtola, J. P. Misiewicz, M. Scheurer, R. A. Shaw, J. B. Schriber, Y. Xie, Z. L. Glick, D. A. Sirianni, J. S. O'Brien, J. M. Waldrop, A. Kumar, E. G. Hohenstein, B. P. Pritchard, B. R. Brooks, H. F. Schaefer III, A. Yu. Sokolov, K. Patkowski, A. E. DePrince III, U. Bozkaya, R. A. King, F. A. Evangelista, J. M. Turney, T. D. Crawford, C. D. Sherrill, "Psi4 1.4: Open-Source Software for High-Throughput Quantum Chemistry", J. Chem. Phys. 152(18) 184108 (2020).
"""
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input psi4 files')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-ns", "--NSTEP",  type=int, default='300', help='iter. number')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument("-d", "--DELTA",  type=str, default='x', help='move step')
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["AdaBelief"], help='optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE (quasi-Newton method) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    args = parser.parse_args()
    return args
    
class UnitValueLib: 
    def __init__(self):
        self.hartree2kcalmol = 627.509 #
        self.bohr2angstroms = 0.52917721067 #
        self.hartree2kjmol = 2625.500 #
        return


class CalculateMoveVector:
    def __init__(self, DELTA, Opt_params, Model_hess, FC_COUNT=-1, temperature=0.0):
        self.Opt_params = Opt_params 
        self.DELTA = DELTA
        self.Model_hess = Model_hess
        self.temperature = temperature
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.FC_COUNT = FC_COUNT
        self.MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.RMS_FORCE_SWITCHING_THRESHOLD = 0.0008
        
    def calc_move_vector(self, iter, geom_num_list, new_g, opt_method_list, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector, e, pre_e, initial_geom_num_list):
        def update_trust_radii(trust_radii, dE, dE_predicted, displacement):
            if dE != 0:
                r =  dE_predicted / dE
            else:
                r = 1.0
            print("dE_predicted/dE : ",r)
            print("disp. - trust_radii :",abs(np.linalg.norm(displacement, ord=2) - trust_radii))
            if r < 0.25:
                return min(np.linalg.norm(displacement, ord=2) / 4, trust_radii / 4)
            elif r > 0.75 and abs(np.linalg.norm(displacement, ord=2) - trust_radii) < 1e-2:
                trust_radii *= 2.0
            else:
                pass
                    
            return np.clip(trust_radii, 0.001, 1.0)
            
            
        def TRM_FSB_dogleg_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):#this function doesnt work well.

            #-----------------------
            if iter == 1:
                self.Model_hess = Model_hess_tmp(self.Model_hess.model_hess, momentum_disp=float(self.DELTA))#momentum_disp is trust_radii.
            
            trust_radii = self.Model_hess.momentum_disp
            
            aprrox_bias_e_shift = abs(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), pre_g.reshape(len(geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), self.Model_hess.model_hess),(geom_num_list - pre_geom).reshape( len(geom_num_list)*3,1)))
            
            
            
            #----------------------
            
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
    
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))
            Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            new_hess = self.Model_hess.model_hess + delta_hess
            
            #---------------------- dogleg
            new_g_reshape = new_g.reshape(len(geom_num_list)*3, 1)
            ipsilon = 1e-8
            p_u = ((np.dot(new_g_reshape.T, new_g_reshape))/(np.dot(np.dot(new_g_reshape.T, new_hess), new_g_reshape)) + ipsilon)*new_g_reshape
            p_b = np.dot(np.linalg.inv(new_hess), new_g_reshape)
            if np.linalg.norm(p_u) >= trust_radii:
                move_vector = (trust_radii * (new_g_reshape/(np.linalg.norm(new_g_reshape) + ipsilon))).reshape(len(geom_num_list), 3)
            elif np.linalg.norm(p_b) <= trust_radii:
                move_vector = p_b.reshape(len(geom_num_list), 3)
            else:
                
                tau = np.sqrt((trust_radii ** 2 - np.linalg.norm(p_u) ** 2)/(np.linalg.norm(p_b - p_u) + ipsilon) ** 2)
                
                print(tau)
                
                if tau <= 1.0:
                    move_vector = (tau * p_u).reshape(len(geom_num_list), 3)
                else:
                    move_vector = ((2.0 - tau) * p_u + (tau - 1.0) * p_b).reshape(len(geom_num_list), 3)
                
            #---------------------
            
            move_vector = trust_radii * (move_vector/(np.linalg.norm(move_vector) + ipsilon))
            
            trust_radii = update_trust_radii(trust_radii, abs(bias_e - pre_bias_e), aprrox_bias_e_shift, geom_num_list - pre_geom)
            print("trust_radii: ",trust_radii)
            self.Model_hess = Model_hess_tmp(new_hess, momentum_disp=trust_radii)#valuable named 'momentum_disp' is trust_radii.
            return move_vector
    
        def TRM_BFGS_dogleg_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):#this function doesnt work well.
 
            if iter == 1:
                self.Model_hess = Model_hess_tmp(self.Model_hess.model_hess, momentum_disp=float(self.DELTA))#momentum_disp is trust_radii.
            
            trust_radii = self.Model_hess.momentum_disp
           
           
            aprrox_bias_e_shift = abs(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), pre_g.reshape(len(geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), self.Model_hess.model_hess),(geom_num_list - pre_geom).reshape( len(geom_num_list)*3,1)))
            
           
            
            #----------------------
            
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            

            #print(Model_hess.model_hess)
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
            #print(A)

            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))
     
            delta_hess = delta_hess_BFGS
            
                
            move_vector = (np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            aprrox_bias_e_shift = abs(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), pre_g.reshape(len(geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), self.Model_hess.model_hess),(geom_num_list - pre_geom).reshape(len(geom_num_list)*3,1)))
            
            trust_radii = update_trust_radii(trust_radii, abs(bias_e - pre_bias_e), aprrox_bias_e_shift, geom_num_list - pre_geom)
            new_hess = self.Model_hess.model_hess + delta_hess
            
            #---------------------- dogleg
            new_g_reshape = new_g.reshape(len(geom_num_list)*3, 1)
            ipsilon = 1e-8
            p_u = ((np.dot(new_g_reshape.T, new_g_reshape))/(np.dot(np.dot(new_g_reshape.T, new_hess), new_g_reshape)) + ipsilon)*new_g_reshape
            p_b = np.dot(np.linalg.inv(new_hess), new_g_reshape)
            if np.linalg.norm(p_u) >= trust_radii:
                move_vector = (trust_radii*(new_g_reshape/(np.linalg.norm(new_g_reshape) + ipsilon))).reshape(len(geom_num_list), 3)
            elif np.linalg.norm(p_b) <= trust_radii:
                move_vector = p_b.reshape(len(geom_num_list), 3)
            else:
                tau = np.sqrt((trust_radii ** 2 - np.linalg.norm(p_u) ** 2)/(np.linalg.norm(p_b - p_u) + ipsilon) ** 2)
                if tau <= 1.0:
                    move_vector = (tau * p_u).reshape(len(geom_num_list), 3)
                else:
                    move_vector = ((2.0 - tau) * p_u + (tau - 1.0) * p_b).reshape(len(geom_num_list), 3)
                
            #---------------------
            trust_radii = update_trust_radii(trust_radii, abs(bias_e - pre_bias_e), aprrox_bias_e_shift, geom_num_list - pre_geom)
            print("trust_radii: ",trust_radii)
            self.Model_hess = Model_hess_tmp(new_hess, momentum_disp=trust_radii)#valuable named 'momentum_disp' is trust_radii.
            return move_vector
            
        def RFO_BFGS_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):
            print("RFO_BFGS_quasi_newton_method")
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            DELTA_for_QNM = self.DELTA
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector
                
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector

            
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
            

            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))

            delta_hess = delta_hess_BFGS
            
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess
            
            matrix_for_RFO = np.append(new_hess, new_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(new_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            lambda_for_calc = min(0.0, float(eigenvalue[np.argmin(eigenvalue)]))
            

                
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            DELTA_for_QNM = self.DELTA
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector
                
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM)
            self.Model_hess = Model_hess_tmp(new_hess)
            
            return move_vector
            
        def BFGS_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):
            print("BFGS_quasi_newton_method")
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            
          
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
           
            
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))

            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS
            else:
                new_hess = self.Model_hess.model_hess
                
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
              
                return move_vector 
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
        

        def RFO_FSB_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):
            print("RFO_FSB_quasi_newton_method")
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            DELTA_for_QNM = self.DELTA
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector
                
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector


            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))
            Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess
            
            matrix_for_RFO = np.append(new_hess, new_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(new_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            lambda_for_calc = min(0.0, float(eigenvalue[np.argmin(eigenvalue)]))
            

                
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            DELTA_for_QNM = self.DELTA
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector
                
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM)
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
            
        def FSB_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):
            print("FSB_quasi_newton_method")
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            
   
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
    
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))
            Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector 
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
        
        # arXiv:2307.13744v1
        def momentum_based_BFGS(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):
            print("momentum_based_BFGS")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:

                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
                
            beta = 0.50
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * new_g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
            
            A = delta_momentum_grad - np.dot(self.Model_hess.model_hess, delta_momentum_disp)
            
            
            delta_hess_BFGS = (np.dot(delta_momentum_grad, delta_momentum_grad.T) / np.dot(delta_momentum_disp.T, delta_momentum_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, delta_momentum_disp) , delta_momentum_disp.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(delta_momentum_disp.T, self.Model_hess.model_hess), delta_momentum_disp))

            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS
            else:
                new_hess = self.Model_hess.model_hess
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector 
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            
            return move_vector 
        
        def momentum_based_FSB(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):
            print("momentum_based_FSB")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.50
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
              
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * new_g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
           
            A = delta_momentum_grad - np.dot(self.Model_hess.model_hess, delta_momentum_disp)
          
            
            
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, delta_momentum_disp) 
            delta_hess_BFGS = (np.dot(delta_momentum_grad, delta_momentum_grad.T) / np.dot(delta_momentum_disp.T, delta_momentum_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, delta_momentum_disp) , delta_momentum_disp.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(delta_momentum_disp.T, self.Model_hess.model_hess), delta_momentum_disp))
            Bofill_const = np.dot(np.dot(np.dot(A.T, delta_momentum_disp), A.T), delta_momentum_disp) / np.dot(np.dot(np.dot(A.T, A), delta_momentum_disp.T), delta_momentum_disp)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector 
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
                
          
        def RFO_momentum_based_BFGS(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):
            print("RFO_momentum_based_BFGS")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.5
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ", np.nanmean(displacement))
                return move_vector
            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * new_g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
          
            A = delta_momentum_grad - np.dot(self.Model_hess.model_hess, delta_momentum_disp)
           
            
            delta_hess_BFGS = (np.dot(delta_momentum_grad, delta_momentum_grad.T) / np.dot(delta_momentum_disp.T, delta_momentum_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, delta_momentum_disp) , delta_momentum_disp.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(delta_momentum_disp.T, self.Model_hess.model_hess), delta_momentum_disp))

            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS
            else:
                new_hess = self.Model_hess.model_hess
             
            DELTA_for_QNM = self.DELTA

            matrix_for_RFO = np.append(new_hess, new_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(new_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            lambda_for_calc = min(0.0, float(eigenvalue[np.argmin(eigenvalue)]))
            

            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                
                return move_vector 
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
        
        def RFO_momentum_based_FSB(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector):
            print("RFO_momentum_based_FSB")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.5
            print("beta :", beta)
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
              
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * new_g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
            
            A = delta_momentum_grad - np.dot(self.Model_hess.model_hess, delta_momentum_disp)
            
            
            
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, delta_momentum_disp) 
            delta_hess_BFGS = (np.dot(delta_momentum_grad, delta_momentum_grad.T) / np.dot(delta_momentum_disp.T, delta_momentum_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, delta_momentum_disp) , delta_momentum_disp.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(delta_momentum_disp.T, self.Model_hess.model_hess), delta_momentum_disp))
            Bofill_const = np.dot(np.dot(np.dot(A.T, delta_momentum_disp), A.T), delta_momentum_disp) / np.dot(np.dot(np.dot(A.T, A), delta_momentum_disp.T), delta_momentum_disp)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess

            DELTA_for_QNM = self.DELTA
            
            matrix_for_RFO = np.append(new_hess, new_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(new_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            lambda_for_calc = min(0.0, float(eigenvalue[np.argmin(eigenvalue)]))
            

            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector 
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
                   
        #arXiv:1412.6980v9
        def AdaMax(geom_num_list, new_g):#not worked well
            print("AdaMax")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            if adam_count == 1:
                adamax_u = 1e-8
                self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, self.Opt_params.adam_v, 0, adamax_u)
                
            else:
                adamax_u = self.Opt_params.eve_d_tilde#eve_d_tilde = adamax_u 
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
            new_adamax_u = max(beta_v*adamax_u, np.linalg.norm(new_g))
               
            move_vector = []

            for i in range(len(geom_num_list)):
                move_vector.append((self.DELTA / (beta_m ** adam_count)) * (adam_m[i] / new_adamax_u))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, new_adamax_u)
            return move_vector
            
        #https://cs229.stanford.edu/proj2015/054_report.pdf
        def NAdam(geom_num_list, new_g):
            print("NAdam")
            mu = 0.975
            nu = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = []
            new_adam_v_hat = []
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(mu*adam_m[i] + (1.0 - mu)*(new_g[i]))
                new_adam_v[i] = copy.copy((nu*adam_v[i]) + (1.0 - nu)*(new_g[i]) ** 2)
                new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64") * ( mu / (1.0 - mu ** adam_count)) + np.array(new_g[i], dtype="float64") * ((1.0 - mu)/(1.0 - mu ** adam_count)))        
                new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64") * (nu / (1.0 - nu ** adam_count)))
            
            move_vector = []
            for i in range(len(geom_num_list)):
                move_vector.append( (self.DELTA*new_adam_m_hat[i]) / (np.sqrt(new_adam_v_hat[i] + Epsilon)))
                
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        
        
        #FIRE
        #Physical Review Letters, Vol. 97, 170201 (2006)
        def FIRE(geom_num_list, new_g):#MD-like optimization method. This method tends to converge local minima.
            print("FIRE")
            adam_count = self.Opt_params.adam_count
            N_acc = 5
            f_inc = 1.10
            f_acc = 0.99
            f_dec = 0.50
            dt_max = 0.8
            alpha_start = 0.1
            if adam_count == 1:
                self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, self.Opt_params.adam_v, 0, [0.1, alpha_start, 0])
                #valuable named 'eve_d_tilde' is parameters for FIRE.
                # [0]:dt [1]:alpha [2]:n_reset
            dt = self.Opt_params.eve_d_tilde[0]
            alpha = self.Opt_params.eve_d_tilde[1]
            n_reset = self.Opt_params.eve_d_tilde[2]
            
            pre_velocity = self.Opt_params.adam_v
            
            velocity = (1.0 - alpha) * pre_velocity + alpha * (np.linalg.norm(pre_velocity, ord=2)/np.linalg.norm(new_g, ord=2)) * new_g
            
            if adam_count > 1 and np.dot(pre_velocity.reshape(1, len(geom_num_list)*3), new_g.reshape(len(geom_num_list)*3, 1)) > 0:
                if n_reset > N_acc:
                    dt = min(dt * f_inc, dt_max)
                    alpha = alpha * f_acc
                n_reset += 1
            else:
                velocity *= 0.0
                alpha = alpha_start
                dt *= f_dec
                n_reset = 0
            
            velocity += dt*new_g
            
            move_vector = velocity * 0.0
            move_vector = copy.copy(dt * velocity)
            
            print("dt, alpha, n_reset :", dt, alpha, n_reset)
            self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, velocity, adam_count, [dt, alpha, n_reset])
            return move_vector
        #RAdam
        #arXiv:1908.03265v4
        def RADAM(geom_num_list, new_g):
            print("RADAM")
            beta_m = 0.9
            beta_v = 0.99
            rho_inf = 2.0 / (1.0- beta_v) - 1.0 
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = []
            new_adam_v_hat = []
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy((beta_v*adam_v[i]) + (1.0-beta_v)*(new_g[i] - new_adam_m[i])**2) + Epsilon
                new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64")/(1.0-beta_m**adam_count))        
                new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64")/(1.0-beta_v**adam_count))
            rho = rho_inf - (2.0*adam_count*beta_v**adam_count)/(1.0 -beta_v**adam_count)
                        
            move_vector = []
            if rho > 4.0:
                l_alpha = []
                for j in range(len(new_adam_v)):
                    l_alpha.append(np.sqrt((abs(1.0 - beta_v**adam_count))/new_adam_v[j]))
                l_alpha = np.array(l_alpha, dtype="float64")
                r = np.sqrt(((rho-4.0)*(rho-2.0)*rho_inf)/((rho_inf-4.0)*(rho_inf-2.0)*rho))
                for i in range(len(geom_num_list)):
                    move_vector.append(self.DELTA*r*new_adam_m_hat[i]*l_alpha[i])
            else:
                for i in range(len(geom_num_list)):
                    move_vector.append(self.DELTA*new_adam_m_hat[i])
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #AdaBelief
        #ref. arXiv:2010.07468v5
        def AdaBelief(geom_num_list, new_g):
            print("AdaBelief")
            beta_m = 0.9
            beta_v = 0.99
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i]-new_adam_m[i])**2)
               
            move_vector = []

            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m[i]/np.sqrt(new_adam_v[i]+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #AdaDiff
        #ref. https://iopscience.iop.org/article/10.1088/1742-6596/2010/1/012027/pdf  Dian Huang et al 2021 J. Phys.: Conf. Ser. 2010 012027
        def AdaDiff(geom_num_list, new_g, pre_g):
            print("AdaDiff")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i])**2 + (1.0-beta_v) * (new_g[i] - pre_g[i]) ** 2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        
        #EVE
        #ref.arXiv:1611.01505v3
        def EVE(geom_num_list, new_g, bias_e, pre_bias_e, pre_g):
            print("EVE")
            beta_m = 0.9
            beta_v = 0.999
            beta_d = 0.999
            c = 10
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            if adam_count > 1:
                eve_d_tilde = self.Opt_params.eve_d_tilde
            else:
                eve_d_tilde = 1.0
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i])/(1 - beta_v**adam_count))
                
            if adam_count > 1:
                eve_d = abs(bias_e - pre_bias_e)/ min(bias_e, pre_bias_e)
                eve_d_hat = np.clip(eve_d, 1/c , c)
                eve_d_tilde = beta_d*eve_d_tilde + (1.0 - beta_d)*eve_d_hat
                
            else:
                pass
            
            for i in range(len(geom_num_list)):
                 move_vector.append((self.DELTA/eve_d_tilde)*new_adam_m_hat[i]/(np.sqrt(new_adam_v_hat[i])+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, eve_d_tilde)
            return move_vector
        #AdamW
        #arXiv:2302.06675v4
        def AdamW(geom_num_list, new_g):
            print("AdamW")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            AdamW_lambda = 0.001
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon) + AdamW_lambda * geom_num_list[i])
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #Adam
        #arXiv:1412.6980
        def Adam(geom_num_list, new_g):
            print("Adam")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon))
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #Adafactor
        #arXiv:1804.04235v1
        def Adafactor(geom_num_list, new_g):
            print("Adafactor")
            Epsilon_1 = 1e-08
            Epsilon_2 = self.DELTA

            adam_count = self.Opt_params.adam_count
            beta = 1 - adam_count ** (-0.8)
            rho = min(0.01, 1/np.sqrt(adam_count))
            alpha = max(np.sqrt(np.square(geom_num_list).mean()),  Epsilon_2) * rho
            adam_v = self.Opt_params.adam_m
            adam_u = self.Opt_params.adam_v
            new_adam_v = adam_v*0.0
            new_adam_u = adam_u*0.0
            new_adam_v = adam_v*0.0
            new_adam_u = adam_u*0.0
            new_adam_u_hat = adam_u*0.0
            for i in range(len(geom_num_list)):
                new_adam_v[i] = copy.copy(beta*adam_v[i] + (1.0-beta)*((new_g[i])**2 + np.array([1,1,1]) * Epsilon_1))
                new_adam_u[i] = copy.copy(new_g[i]/np.sqrt(new_adam_v[i]))
                
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_u_hat[i] = copy.copy(new_adam_u[i] / max(1, np.sqrt(np.square(new_adam_u).mean())))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(alpha*new_adam_u_hat[i])
                 
            self.Opt_params = Opt_calc_tmps(new_adam_v, new_adam_u, adam_count)
        
            return move_vector
        #Prodigy
        #arXiv:2306.06101v1
        def Prodigy(geom_num_list, new_g, initial_geom_num_list):
            print("Prodigy")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            adam_count = self.Opt_params.adam_count

            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            if adam_count == 1:
                adam_r = 0.0
                adam_s = adam_m*0.0
                d = 1e-1
                new_d = d
                self.Opt_params = Opt_calc_tmps(adam_m, adam_v, adam_count - 1, [d, adam_r, adam_s])
            else:
                d = self.Opt_params.eve_d_tilde[0]
                adam_r = self.Opt_params.eve_d_tilde[1]
                adam_s = self.Opt_params.eve_d_tilde[2]
                
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            new_adam_s = adam_s*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]*d))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i]*d)**2)
                
                new_adam_s[i] = np.sqrt(beta_v)*adam_s[i] + (1.0 - np.sqrt(beta_v))*self.DELTA*new_g[i]*d**2  
            new_adam_r = np.sqrt(beta_v)*adam_r + (1.0 - np.sqrt(beta_v))*(np.dot(new_g.reshape(1,len(new_g)*3), (initial_geom_num_list - geom_num_list).reshape(len(new_g)*3,1)))*self.DELTA*d**2
            
            new_d = float(max((new_adam_r / np.linalg.norm(new_adam_s ,ord=1)), d))
            move_vector = []

            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_d*new_adam_m[i]/(np.sqrt(new_adam_v[i])+Epsilon*d))
            
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, [new_d, new_adam_r, new_adam_s])
            return move_vector
        
        #AdaBound
        #arXiv:1902.09843v1
        def Adabound(geom_num_list, new_g):
            print("AdaBound")
            adam_count = self.Opt_params.adam_count
            move_vector = []
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            if adam_count == 1:
                adam_m = self.Opt_params.adam_m
                adam_v = np.zeros((len(geom_num_list),3,3))
            else:
                adam_m = self.Opt_params.adam_m
                adam_v = self.Opt_params.adam_v
                
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            V = adam_m*0.0
            Eta = adam_m*0.0
            Eta_hat = adam_m*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(np.dot(np.array([new_g[i]]).T, np.array([new_g[i]]))))
                V[i] = copy.copy(np.diag(new_adam_v[i]))
                
                Eta_hat[i] = copy.copy(np.clip(self.DELTA/np.sqrt(V[i]), 0.1 - (0.1/(1.0 - beta_v) ** (adam_count + 1)) ,0.1 + (0.1/(1.0 - beta_v) ** adam_count) ))
                Eta[i] = copy.copy(Eta_hat[i]/np.sqrt(adam_count))
                    
            for i in range(len(geom_num_list)):
                move_vector.append(Eta[i] * new_adam_m[i])
            
            return move_vector    
        
        #Adadelta
        #arXiv:1212.5701v1
        def Adadelta(geom_num_list, new_g):#delta is not required. This method tends to converge local minima.
            print("Adadelta")
            rho = 0.9
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            Epsilon = 1e-06
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(rho * adam_m[i] + (1.0 - rho)*(new_g[i]) ** 2)
            move_vector = []
            
            for i in range(len(geom_num_list)):
                if adam_count > 1:
                    move_vector.append(new_g[i] * (np.sqrt(np.square(adam_v).mean()) + Epsilon)/(np.sqrt(np.square(new_adam_m).mean()) + Epsilon))
                else:
                    move_vector.append(new_g[i])
            if abs(np.sqrt(np.square(move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(new_g).mean())) > self.RMS_FORCE_THRESHOLD:
                move_vector = new_g

            for i in range(len(geom_num_list)):
                new_adam_v[i] = copy.copy(rho * adam_v[i] + (1.0 - rho) * (move_vector[i]) ** 2)
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        

        def MD_like_Perturbation(move_vector):#This function is just for fun. Thus, it is no scientific basis.
            """Langevin equation"""
            Boltzmann_constant = 3.16681*10**(-6) # hartree/K
            damping_coefficient = 10.0
	
            temperature = self.temperature
            perturbation = self.DELTA * np.sqrt(2.0 * damping_coefficient * Boltzmann_constant * temperature) * np.random.normal(loc=0.0, scale=1.0, size=3*len(move_vector)).reshape(len(move_vector), 3)

            return perturbation

        move_vector_list = []
     
        #---------------------------------
        
        for opt_method in opt_method_list:
            # group of steepest descent
            if opt_method == "RADAM":
                tmp_move_vector = RADAM(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adam":
                tmp_move_vector = Adam(geom_num_list, new_g) 
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adadelta":
                tmp_move_vector = Adadelta(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdamW":
                tmp_move_vector = AdamW(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaDiff":
                tmp_move_vector = AdaDiff(geom_num_list, new_g, pre_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adafactor":
                tmp_move_vector = Adafactor(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaBelief":
                tmp_move_vector = AdaBelief(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adabound":
                tmp_move_vector = Adabound(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "EVE":
                tmp_move_vector = EVE(geom_num_list, new_g, bias_e, pre_bias_e, pre_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Prodigy":
                tmp_move_vector = Prodigy(geom_num_list, new_g, pre_geom)#initial_geom_num_list is not assigned.
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaMax":
                tmp_move_vector = AdaMax(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "NAdam":    
                tmp_move_vector = NAdam(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "FIRE":
                tmp_move_vector = FIRE(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            # group of quasi-Newton method
            
            elif opt_method == "TRM_FSB":
                if iter != 0:
                    tmp_move_vector = TRM_FSB_dogleg_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "TRM_BFGS":
                if iter != 0:
                    tmp_move_vector = TRM_BFGS_dogleg_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
            
            elif opt_method == "BFGS":
                if iter != 0:
                    tmp_move_vector = BFGS_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_BFGS":
                if iter != 0:
                    tmp_move_vector = RFO_BFGS_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "FSB":
                if iter != 0:
                    tmp_move_vector = FSB_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "RFO_FSB":
                if iter != 0:
                    tmp_move_vector = RFO_FSB_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "mBFGS":
                if iter != 0:
                    tmp_move_vector = momentum_based_BFGS(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "mFSB":
                if iter != 0:
                    tmp_move_vector = momentum_based_FSB(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_mBFGS":
                if iter != 0:
                    tmp_move_vector = RFO_momentum_based_BFGS(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_mFSB":
                if iter != 0:
                    tmp_move_vector = RFO_momentum_based_FSB(geom_num_list, new_g, pre_g, pre_geom, bias_e, pre_bias_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g 
                    move_vector_list.append(tmp_move_vector)
            else:
                print("optimization method that this program is not sppourted is selected... thus, default method is selected.")
                tmp_move_vector = AdaBelief(geom_num_list, new_g)
        #---------------------------------
        
        if len(move_vector_list) > 1:
            if abs(new_g.max()) < self.MAX_FORCE_SWITCHING_THRESHOLD and abs(np.sqrt(np.square(new_g).mean())) < self.RMS_FORCE_SWITCHING_THRESHOLD:
                move_vector = copy.copy(move_vector_list[1])
                print("Chosen method: ", opt_method_list[1])
            else:
                move_vector = copy.copy(move_vector_list[0])
                print("Chosen method: ", opt_method_list[0])
        else:
            move_vector = copy.copy(move_vector_list[0])
        
        perturbation = MD_like_Perturbation(move_vector)
        
        #if np.linalg.norm(move_vector) > 0.5:
        #    move_vector = 0.5 * move_vector / np.linalg.norm(move_vector)
        
        
        move_vector += perturbation
        print("perturbation: ", np.linalg.norm(perturbation))
        
        print("step radii: ", np.linalg.norm(move_vector))
        
        hess_eigenvalue, _ = np.linalg.eig(self.Model_hess.model_hess)
        
        print("NORMAL MODE EIGENVALUE:\n",np.sort(hess_eigenvalue),"\n")
        
        #---------------------------------
        new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms
        
        return new_geometry, np.array(move_vector, dtype="float64"), iter, self.Opt_params, self.Model_hess
     
class Opt_calc_tmps:
    def __init__(self, adam_m, adam_v, adam_count, eve_d_tilde=0.0):
        self.adam_m = adam_m
        self.adam_v = adam_v
        self.adam_count = 1 + adam_count
        self.eve_d_tilde = eve_d_tilde
            
class Model_hess_tmp:
    def __init__(self, model_hess, momentum_disp=0, momentum_grad=0):
        self.model_hess = model_hess
        self.momentum_disp = momentum_disp
        self.momentum_grad = momentum_grad



class FindMESX:
    def __init__(self, args):

        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
 
        self.ENERGY_LIST_FOR_PLOTTING = [] #
        self.SMF_ENERGY_LIST_FOR_PLOTTING = [] #
        self.NUM_LIST = [] #

        self.MAX_FORCE_THRESHOLD = 0.0003 #
        self.RMS_FORCE_THRESHOLD = 0.0002 #
        self.MAX_DISPLACEMENT_THRESHOLD = 0.0015 # 
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0010 #
        
        self.args = args #

        #---------------------------

        
        if args.DELTA == "x":
            if args.opt_method[0] == "FSB":
                args.DELTA = 0.15
            elif args.opt_method[0] == "RFO_FSB":
                args.DELTA = 0.15
            elif args.opt_method[0] == "BFGS":
                args.DELTA = 0.15
            elif args.opt_method[0] == "RFO_BFGS":
                args.DELTA = 0.15
                
            elif args.opt_method[0] == "mBFGS":
                args.DELTA = 0.50
            elif args.opt_method[0] == "mFSB":
                args.DELTA = 0.50
            elif args.opt_method[0] == "RFO_mBFGS":
                args.DELTA = 0.30
            elif args.opt_method[0] == "RFO_mFSB":
                args.DELTA = 0.30

            elif args.opt_method[0] == "Adabound":
                args.DELTA = 0.01
            elif args.opt_method[0] == "AdaMax":
                args.DELTA = 0.01
                
            elif args.opt_method[0] == "TRM_FSB":
                args.DELTA = 0.60
            elif args.opt_method[0] == "TRM_BFGS":
                args.DELTA = 0.60
            else:
                args.DELTA = 0.06
        else:
            pass 
            
        self.DELTA = float(args.DELTA) # 

        self.N_THREAD = args.N_THREAD #
        self.SET_MEMORY = args.SET_MEMORY #
        self.START_FILE = args.INPUT #
        self.NSTEP = args.NSTEP #
        #-----------------------------
        self.BASIS_SET = args.basisset # 
        self.FUNCTIONAL = args.functional # 
        
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input")
            sys.exit(0)
        
        self.SUB_BASIS_SET = "" # 
        if len(args.sub_basisset) > 0:
            self.SUB_BASIS_SET +="\nassign "+str(self.BASIS_SET)+"\n" # 
            for j in range(int(len(args.sub_basisset)/2)):
                self.SUB_BASIS_SET += "assign "+args.sub_basisset[2*j]+" "+args.sub_basisset[2*j+1]+"\n"
            print("Basis Sets defined by User are detected.")
            print(self.SUB_BASIS_SET) #
        #-----------------------------
        self.optmethod = args.opt_method
        self.SMF_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_SMF_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time())+"/"
        os.makedirs(self.SMF_FOLDER_DIRECTORY, exist_ok=True) #
        
        self.reactant_Model_hess = None #
        self.product_Model_hess = None #
        
        self.reactant_Opt_params = None #
        self.product_Opt_params = None #
        self.FC_COUNT = -1
        
        return
        
        
    def make_geometry_list(self):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        rea_geometry_list = []
        pro_geometry_list = []
        with open(self.START_FILE[:-4]+"_reactant.xyz","r") as f:
            rea_words = f.readlines()
        with open(self.START_FILE[:-4]+"_product.xyz","r") as f:
            pro_words = f.readlines()
        
        rea_start_data = []
        pro_start_data = []
        for word in rea_words:
            rea_start_data.append(word.split())
        for word in pro_words:
            pro_start_data.append(word.split())
            
        rea_electric_charge_and_multiplicity = rea_start_data[0]
        pro_electric_charge_and_multiplicity = pro_start_data[0]

        element_list = []
            


        for i in range(1, len(rea_start_data)):
            element_list.append(rea_start_data[i][0])

        rea_geometry_list.append(rea_start_data)
        pro_geometry_list.append(pro_start_data)

        return rea_geometry_list, pro_geometry_list, element_list, rea_electric_charge_and_multiplicity, pro_electric_charge_and_multiplicity

    def make_geometry_list_2(self, new_geometry, element_list, electric_charge_and_multiplicity):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """load structure updated geometry for next QM calculation"""
        new_geometry = new_geometry.tolist()
        
        geometry_list = []
        print("geometry:")
        new_data = [electric_charge_and_multiplicity]
        for num, geometry in enumerate(new_geometry):
           
            geometry = list(map(str, geometry))
            geometry = [element_list[num]] + geometry
            new_data.append(geometry)
            print(" ".join(geometry))
            
        geometry_list.append(new_data)
        print()
        return geometry_list

    def make_psi4_input_file(self, rea_geometry_list, pro_geometry_list, iter):
        """structure updated geometry is saved."""
        rea_file_directory = self.SMF_FOLDER_DIRECTORY+"samples_"+str(self.START_FILE[:-4])+"_"+str(iter)+"_reactant"
        pro_file_directory = self.SMF_FOLDER_DIRECTORY+"samples_"+str(self.START_FILE[:-4])+"_"+str(iter)+"_product"
        try:
            os.mkdir(rea_file_directory)
            os.mkdir(pro_file_directory)
        except:
            pass
        for y, geometry in enumerate(rea_geometry_list):
            with open(rea_file_directory+"/"+self.START_FILE[:-4]+"_"+str(y)+".xyz","w") as w:
                for rows in geometry:
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        for y, geometry in enumerate(pro_geometry_list):
            with open(pro_file_directory+"/"+self.START_FILE[:-4]+"_"+str(y)+".xyz","w") as w:
                for rows in geometry:
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")

        return rea_file_directory, pro_file_directory

    def sinple_plot(self, num_list, energy_list):
        
        fig = plt.figure()

        ax1 = fig.add_subplot(1, 1, 1)
        

        ax1.plot(num_list, energy_list, "g--.")
        

        ax1.set_xlabel('ITR.')
        

        ax1.set_ylabel('electronic Energy [kcal/mol]')
        
        
        plt.tight_layout()
        plt.savefig(self.SMF_FOLDER_DIRECTORY+"Energy_plot_sinple_"+str(time.time())+".png", format="png", dpi=300)
        plt.close()
        return

    def xyz_file_make(self):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9]_reactant/*.xyz") + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]_reactant/*.xyz") + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]_reactant/*.xyz") + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]_reactant/*.xyz") + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]_reactant/*.xyz") + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]_reactant/*.xyz") + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]_product/*.xyz")[::-1] + glob.glob(self.SMF_FOLDER_DIRECTORY+"samples_*_[0-9]_product/*.xyz")[::-1]   
        #print(file_list,"\n")
        
        
        for m, file in enumerate(file_list):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
                with open(self.SMF_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w:
                    atom_num = len(sample)-1
                    w.write(str(atom_num)+"\n")
                    w.write("Frame "+str(m)+"\n")
                del sample[0]
                for i in sample:
                    with open(self.SMF_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w2:
                        w2.write(i)
        print("\ngeometry collection is completed...\n")
        return
    def psi4_calculation(self, file_directory, element_list, electric_charge_and_multiplicity, iter):
        """execute QM calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                print("\n",input_file,"\n")
                
                psi4.set_options({'reference': 'uks'})
                logfile = file_directory+"/"+self.START_FILE[:-4]+'_'+str(num)+'.log'
                psi4.set_options({"MAXITER": 700})
                if len(self.SUB_BASIS_SET) > 0:
                    psi4.basis_helper(self.SUB_BASIS_SET, name='User_Basis_Set', set_option=False)
                    psi4.set_options({"basis":'User_Basis_Set'})
                else:
                    psi4.set_options({"basis":self.BASIS_SET})
                
                psi4.set_output_file(logfile)
                psi4.set_num_threads(nthread=self.N_THREAD)
                psi4.set_memory(self.SET_MEMORY)
                
               
                with open(input_file,"r") as f:
                    input_data = f.read()
                    input_data = psi4.geometry(input_data)
                    input_data_for_display = np.array(input_data.geometry(), dtype = "float64")
                            
                g, wfn = psi4.gradient(self.FUNCTIONAL, molecule=input_data, return_wfn=True)

                g = np.array(g, dtype = "float64")

                with open(input_file[:-4]+".log","r") as f:
                    word_list = f.readlines()
                    for word in word_list:
                        if "    Total Energy =             " in word:
                            word = word.replace("    Total Energy =             ","")
                            e = (float(word))
                print("\n")

                
                if self.FC_COUNT == -1:
                    pass
                
                elif iter % self.FC_COUNT == 0:
                    
                    """exact hessian"""
                    _, wfn = psi4.frequencies(self.FUNCTIONAL, return_wfn=True, ref_gradient=wfn.gradient())
                    exact_hess = np.array(wfn.hessian())
                    
                    freqs = np.array(wfn.frequencies())
                    
                    print("frequencies: \n",freqs)
                    self.Model_hess = Model_hess_tmp(exact_hess, momentum_disp=self.Model_hess.momentum_disp, momentum_grad=self.Model_hess.momentum_grad)
                
                


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return 0, 0, 0, finish_frag 
                
            psi4.core.clean() 
        return e, g, input_data_for_display, finish_frag

    def calc_biaspot(self, rea_e, pro_e, rea_g, pro_g, rea_geom_num_list, pro_geom_num_list, element_list,  pre_rea_g, pre_pro_g, pre_rea_geom, pre_pro_geom, iter, initial_rea_geom_num_list, initial_pro_geom_num_list):#calclate bais optential for SMF
        alpha = 0.0001
        smf_e = 0.5 * (pro_e + rea_e) + (pro_e - rea_e) ** 2 / alpha 

        new_pro_g = 0.5 * pro_g + 2.0 * (pro_e - rea_e) * pro_g / alpha
        new_rea_g = 0.5 * rea_g - 2.0 * (pro_e - rea_e) * rea_g / alpha


        print("new_rea_g:\n", new_rea_g)
        print("new_pro_g:\n", new_pro_g)
        
        return smf_e, new_rea_g, new_pro_g



    def main(self):
        
        finish_frag = False
        rea_geometry_list, pro_geometry_list, element_list, rea_electric_charge_and_multiplicity, pro_electric_charge_and_multiplicity = self.make_geometry_list()
        reactant_file_directory, product_file_directory = self.make_psi4_input_file(rea_geometry_list, pro_geometry_list, 0)

        reactant_energy_list = []
        product_energy_list = []

        #------------------------------------
        
        adam_m = []
        adam_v = []    
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))        
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        self.reactant_Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.product_Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.reactant_Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
        self.product_Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
         

        #-----------------------------------
        with open(self.SMF_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(args))
        pre_bias_e = 0.0
        pre_rea_e = 0.0
        pre_pro_e = 0.0
        pre_rea_g = []
        pre_pro_g = []
        for i in range(len(element_list)):
            pre_rea_g.append(np.array([0,0,0], dtype="float64"))
            pre_pro_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_rea_move_vector = pre_rea_g
        pre_pro_move_vector = pre_pro_g

      
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = glob.glob(self.SMF_FOLDER_DIRECTORY+"*.txt")
            for file in exit_file_detect:
                if "end.txt" in file:
                    exit_flag = True
                    break
            if exit_flag:
                psi4.core.clean()
                break
            print("\n# ITR. "+str(iter)+"\n")




            rea_e, rea_g, rea_geom_num_list, finish_frag = self.psi4_calculation(reactant_file_directory, element_list, 
            rea_electric_charge_and_multiplicity, iter)
            pro_e, pro_g, pro_geom_num_list, finish_frag = self.psi4_calculation(product_file_directory, element_list, 
            rea_electric_charge_and_multiplicity, iter)

            if iter < 2:
                initial_rea_geom_num_list = rea_geom_num_list
                initial_pro_geom_num_list = pro_geom_num_list
                pre_rea_geom = initial_rea_geom_num_list
                pre_pro_geom = initial_pro_geom_num_list
            elif iter > 2:
                pass
                #if pre_pro_e > pro_e:
                #    initial_pro_geom_num_list = pre_pre_pro_geom
                
                #if pre_rea_e > rea_e:
                #    initial_rea_geom_num_list = pre_pre_rea_geom

            #-------------------energy profile 
            if iter == 0:
                with open(self.SMF_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("reactant energy [hartree], product energy [hartree] \n")
            with open(self.SMF_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(rea_e)+","+str(pro_e)+"\n")


            #-------------------
            if finish_frag:#If QM calculation doesn't end, the process of this program is terminated. 
                break   
            
            bias_e, new_rea_g, new_pro_g = self.calc_biaspot(rea_e, pro_e, rea_g, pro_g, rea_geom_num_list, pro_geom_num_list, element_list,  pre_rea_g, pre_pro_g, pre_rea_geom, pre_pro_geom, iter, initial_rea_geom_num_list, initial_pro_geom_num_list)#new_geometry:ang.
            
            rea_CMV = CalculateMoveVector(self.DELTA, self.reactant_Opt_params, self.reactant_Model_hess)
            pro_CMV = CalculateMoveVector(self.DELTA, self.product_Opt_params, self.product_Model_hess)
            
            rea_new_geometry, rea_move_vector, iter, self.reactant_Opt_params, self.reactant_Model_hess = rea_CMV.calc_move_vector(iter, rea_geom_num_list, new_rea_g, self.optmethod, pre_rea_g, pre_rea_geom, bias_e, pre_bias_e, pre_rea_move_vector, rea_e, pre_rea_e, initial_rea_geom_num_list)
            pro_new_geometry, pro_move_vector, iter, self.product_Opt_params, self.product_Model_hess = pro_CMV.calc_move_vector(iter, pro_geom_num_list, new_pro_g, self.optmethod, pre_pro_g, pre_pro_geom, bias_e, pre_bias_e, pre_pro_move_vector, pro_e, pre_pro_e, initial_pro_geom_num_list)
            
            print("caluculation results (unit a.u.):")
            print("OPT method            : {} ".format(self.optmethod))
            print("                         Value                        ")
            
            print("ENERGY                (reactant) : {:>15.12f} ".format(rea_e))
            print("Maxinum  Force        (reactant) : {:>15.12f}              ".format(abs(new_rea_g.max())))
            print("RMS      Force        (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(new_rea_g).mean()))))
            print("Maxinum  Displacement (reactant) : {:>15.12f}              ".format(abs(rea_move_vector.max())))
            print("RMS      Displacement (reactant) : {:>15.12f}              ".format(abs(np.sqrt(np.square(rea_move_vector).mean()))))
            print("ENERGY SHIFT          (reactant) : {:>15.12f} ".format(rea_e - pre_rea_e))
            print("ENERGY                (product ) : {:>15.12f} ".format(pro_e))
            print("Maxinum  Force        (product ) : {:>15.12f}              ".format(abs(new_pro_g.max())))
            print("RMS      Force        (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(new_pro_g).mean()))))
            print("Maxinum  Displacement (product ) : {:>15.12f}              ".format(abs(pro_move_vector.max())))
            print("RMS      Displacement (product ) : {:>15.12f}              ".format(abs(np.sqrt(np.square(pro_move_vector).mean()))))
            print("ENERGY SHIFT          (product ) : {:>15.12f} ".format(pro_e - pre_pro_e))
            print("BIAS  ENERGY          : {:>15.12f} ".format(bias_e))
            print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(bias_e - pre_bias_e))
            
            delta_geom = rea_new_geometry - pro_new_geometry
            if abs(new_rea_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt(np.square(new_rea_g).mean())) < self.RMS_FORCE_THRESHOLD and  abs(rea_move_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(rea_move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD and abs(new_pro_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt(np.square(new_pro_g).mean())) < self.RMS_FORCE_THRESHOLD and  abs(pro_move_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(pro_move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
                break

            if iter > 1:
                pre_pre_rea_geom = pre_rea_geom
                pre_pre_pro_geom = pre_pro_geom
            pre_bias_e = bias_e#Hartree
            pre_rea_e = rea_e
            pre_pro_e = pro_e
            pre_rea_g = new_rea_g#Hartree/Bohr
            pre_pro_g = new_pro_g#Hartree/Bohr
            pre_rea_geom = rea_geom_num_list#Bohr
            pre_pro_geom = pro_geom_num_list#Bohr
            pre_rea_move_vector = rea_move_vector
            pre_pro_move_vector = pro_move_vector
            rea_geometry_list = self.make_geometry_list_2(rea_new_geometry, element_list, rea_electric_charge_and_multiplicity)
            pro_geometry_list = self.make_geometry_list_2(pro_new_geometry, element_list, pro_electric_charge_and_multiplicity)
            reactant_file_directory, product_file_directory = self.make_psi4_input_file(rea_geometry_list, pro_geometry_list, iter+1)
            self.SMF_ENERGY_LIST_FOR_PLOTTING.append(bias_e)
            reactant_energy_list.append(rea_e)
            product_energy_list.append(pro_e)

        self.ENERGY_LIST_FOR_PLOTTING = np.array(reactant_energy_list + product_energy_list[::-1], dtype="float64")
        self.SMF_ENERGY_LIST_FOR_PLOTTING = np.array(self.SMF_ENERGY_LIST_FOR_PLOTTING, dtype="float64")

        self.sinple_plot(np.arange(iter*2), self.ENERGY_LIST_FOR_PLOTTING)#
        self.sinple_plot(np.arange(iter), self.SMF_ENERGY_LIST_FOR_PLOTTING)

        self.xyz_file_make()
        #-----------------------
        
        local_max_energy_list_index = argrelextrema(np.array(self.ENERGY_LIST_FOR_PLOTTING), np.greater)
        with open(self.SMF_FOLDER_DIRECTORY+"approx_TS.txt","w") as f:
            for j in local_max_energy_list_index[0].tolist():
                f.write(str(np.arange(iter*2)[j])+"\n")
        
        inverse_energy_list = (-1)*np.array(self.ENERGY_LIST_FOR_PLOTTING, dtype="float64")
        local_min_energy_list_index = argrelextrema(np.array(inverse_energy_list), np.greater)
        with open(self.SMF_FOLDER_DIRECTORY+"approx_EQ.txt","w") as f:
            for j in local_min_energy_list_index[0].tolist():
                f.write(str(np.arange(iter*2)[j])+"\n")
        
        with open(self.SMF_FOLDER_DIRECTORY+"energy_profile.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
    
        return



if __name__ == "__main__":
    args = parser()
    mesx = FindMESX(args)
    mesx.main()
