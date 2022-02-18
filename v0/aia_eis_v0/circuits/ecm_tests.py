import math
import numpy as np
import pandas as pd
import sys
sys.path.append('../')

from circuits.circuit_pack import *
from circuits.ecm import ECM
from circuits.ecm_simulator import ecm_simulator_1
from IS.IS import IS_0
from utils.frequency_generator import fre_generator
from utils.visualize_utils.impedance_plots import nyquist_multiPlots_1

# -------------------- Test ECM on R(RC) -------------------
R0, R1, C0 = 50, 200, 1e-3
limit_list = [[1, 1000], [1, 1000], [0, 1]]
fre_list, w_list = fre_generator(f_start=4, f_end=-1, pts_decade=5)
z_arr = np.array([RaRCb(w, R0, R1, C0) for w in w_list])

RaRCb_IS = IS_0()
RaRCb_IS.raw_z_arr = z_arr
RaRCb_IS.exp_area = 1.0
RaRCb_IS.z_arr = z_arr
RaRCb_IS.fre_arr = np.array(fre_list)
RaRCb_IS.w_arr = np.array(w_list)

RaRCb_ecm = ECM(ecm_serial='3_014', proba=1.0, fre=None, limit=limit_list, para=None, z_sim=None)
RaRCb_ecm.w_arr = np.array(w_list)
RaRCb_ecm.identify_para(IS=RaRCb_IS)
print(RaRCb_ecm.para_arr)
z_sim_arr = RaRCb_ecm.simulate_Z()

z_pack_list = [z_arr.tolist(), z_sim_arr.tolist()]
nyquist_multiPlots_1(z_pack_list=z_pack_list, x_lim=[30, 250], y_lim=[0, 120], plot_label_list=['Ideal IS', 'Mine-Fit'])
# -------------------- Test ECM on R(RC) -------------------