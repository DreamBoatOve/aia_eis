from IS import IS_0
from circuits.vogit_0 import Vogit

RaRCb_IS = IS_0()
RaRCb_IS.read_from_EcmCls(fp='../plugins_test/jupyter_code/rbp_files/0/R(RC)_ecm_pkl/', fn='2021_08_04_ecm.pkl')

RaRCb_vogit = Vogit(impSpe=RaRCb_IS)

OA_obj_fun_mode = 'imag'
print(OA_obj_fun_mode)
RaRCb_vogit.lin_KK(OA_obj_fun_mode=OA_obj_fun_mode, save_iter=True)
print('M=', RaRCb_vogit.M, 'u=',RaRCb_vogit.u)