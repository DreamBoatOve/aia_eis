import math

from circuits.circuit_pack import *
"""
Module Function
    已知ECM和拟合的参数，把参数准备好，送个ECM对应的函数
"""

def ecm_simulator_1(ecm_serial, para_list, fre=None , w =None):
    # ecm_serial = '2_030'
    element_num_str, serial_str = ecm_serial.split('_')
    # L, C, W, T, R这都算是一个element
    element_num, serial = int(element_num_str), int(serial_str)

    if element_num == 2:
        if serial == 1:
            # ecm_2_001  R0R1
            # def RR(w, R0, R1):
            R0, R1 = para_list
            z_list = [RR(i, R0, R1) for i in w]
        elif serial == 2:
            # ecm_2_002  (R0R1)
            # def aRRb(w, R0, R1):
            R0, R1 = para_list
            z_list = [aRRb(i, R0, R1) for i in w]
        elif serial == 3:
            # ecm_2_003  R0L0
            # RL(w, R0, L0):
            R0, L0 = para_list
            z_list = [RL(i, R0, L(i, L0)) for i in w]
        elif serial == 4:
            # ecm_2_004  (R0L0)
            # def aRLb(w, R0, L0):
            R0, L0 = para_list
            z_list = [aRLb(i, R0, L0) for i in w]
        elif serial == 5:
            # ecm_2_005  R0C0
            # def RC(w, R0, C0):
            R0, C0 = para_list
            z_list = [aRCb(i, R0, C0) for i in w]
        elif serial == 6:
            # ecm_2_006  (R0C0)
            # def aRCb(w, R0, C0):
            R0, C0 = para_list
            z_list = [aRCb(w, R0, C0) for i in w]
        elif serial == 7:
            # ecm_2_007  R0Q0
            # def RQ(w, R0, Q0_pair):
            R0, Q0_pair = para_list
            z_list = [RQ(i, R0, Q0_pair) for i in w]
    elif element_num == 3:
        if serial == 1:
            # ecm_3_001  R0R1R2
            # def RRR(w, R0, R1, R2):
            R0, R1, R2 = para_list
            z_list = [RRR(i, R0, R1, R2) for i in w]
        elif serial == 2:
            # ecm_3_002	R0(R1R2)
            # def RaRRb(w, R0, R1, R2):
            R0, R1, R2 = para_list
            z_list = [RaRRb(i, R0, R1, R2) for i in w]
        elif serial == 3:
            # ecm_3_003	(R0R1R2)
            # def aRRRb(w, R0, R1, R2):
            R0, R1, R2 = para_list
            z_list = [aRRRb(i, R0, R1, R2) for i in w]
        elif serial == 4:
            # ecm_3_004	R0R1L0
            # def RRL(w, R0, R1, L0):
            R0, R1, L0 = para_list
            z_list = [RRL(i, R0, R1, L0) for i in w]
        elif serial == 5:
            # ecm_3_005	R0(R1L0)
            # def RaRLb(w, R0, R1, L0):
            R0, R1, L0 = para_list
            z_list = [RaRLb(i, R0, R1, L0) for i in w]
        elif serial == 6:
            # ecm_3_006	(R0R1)L0
            # def aRRbL(w, R0, R1, L0):
            R0, R1, L0 = para_list
            z_list = [aRRbL(i, R0, R1, L0) for i in w]
        elif serial == 7:
            # ecm_3_007	(R0R1L0)
            # def aRRLb(w, R0, R1, L0):
            R0, R1, L0 = para_list
            z_list = [aRRLb(i, R0, R1, L0) for i in w]
        elif serial == 8:
            # ecm_3_008	R0L0L1
            # def RLL(w, R0, L0, L1):
            R0, L0, L1 = para_list
            z_list = [RLL(i, R0, L0, L1) for i in w]
        elif serial == 9:
            # ecm_3_009	R0(L0L1)
            # def RaLLb(w, R0, L0, L1):
            R0, L0, L1 = para_list
            z_list = [RaLLb(w, R0, L0, L1) for i in w]
        elif serial == 10:
            # ecm_3_010	(R0L0L1)
            # def aRLLb(w, R0, L0, L1):
            R0, L0, L1 = para_list
            z_list = [aRLLb(w, R0, L0, L1) for i in w]
        elif serial == 11:
            # ecm_3_011	(R0L0)L1
            # def aRLbL(w, R0, L0, L1):
            R0, L0, L1 = para_list
            z_list = [aRLbL(w, R0, L0, L1) for i in w]
        elif serial == 12:
            # ecm_3_012	R0R1C0
            # def RRC(w, R0, R1, C0):
            R0, R1, C0 = para_list
            z_list = [RRC(w, R0, R1, C0) for i in w]
        elif serial == 13:
            # ecm_3_012	R0R1C0
            # def RRC(w, R0, R1, C0):
            R0, R1, C0 = para_list
            z_list = [RRC(w, R0, R1, C0) for i in w]
        elif serial == 14:
            # ecm_3_014	R0(R1C0)
            # DPFC: ECM-0 R(CR)
            # DPFC: ECM-0 R0(C0R1)
            # RaCRb == R0aC0R1b, Simplified Randles Cell
            # def RaRCb(w, R0, R1, C0):
            R0, R1, C0 = para_list
            z_list = [RaRCb(i, R0, R1, C0) for i in w]
        elif serial == 15:
            # ecm_3_015	(R0R1C0)
            # def aRRCb(w, R0, R1, C0):
            R0, R1, C0 = para_list
            z_list = [aRRCb(w, R0, R1, C0) for i in w]
        elif serial == 16:
            # ecm_3_016	R0C0C1
            # def RCC(w, R0, C0, C1):
            R0, C0, C1 = para_list
            z_list = [RCC(w, R0, C0, C1) for i in w]
        elif serial == 17:
            # ecm_3_017	(R0C0)C1
            # def aRCbC(w, R0, C0, C1):
            R0, C0, C1 = para_list
            z_list = [aRCbC(i, R0, C0, C1) for i in w]
        elif serial == 18:
            #ecm_3_018	R0(C0C1)
            # def RaCCb(w, R0, C0, C1):
            R0, C0, C1 = para_list
            z_list = [RaCCb(w, R0, C0, C1) for i in w]
        elif serial == 19:
            # ecm_3_019	(R0C0C1)
            # def aRCCb(w, R0, C0, C1):
            R0, C0, C1 = para_list
            z_list = [aRCCb(w, R0, C0, C1) for i in w]
        elif serial == 20:
            # ecm_3_020	R0R1Q0
            # def RRQ(w, R0, R1, Q0_pair):
            R0, R1 = para_list[0:2]
            Q0_pair = para_list[2:]
            z_list = [RRQ(w, R0, R1, Q0_pair) for i in w]
        elif serial == 21:
            # ecm_3_021	(R0R1)Q0
            # def aRRbQ(w, R0, R1, Q0_pair):
            R0, R1 = para_list[0:2]
            Q0_pair = para_list[2:]
            z_list = [aRRbQ(w, R0, R1, Q0_pair) for i in w]
        elif serial == 22:
            # ecm_3_022	R0(R1Q0)
            # DPFC: ECM-1 R(QR)
            # def RaRQb(w, R0, R1, Q0_pair):
            R0, R1 = para_list[0:2]
            Q0_pair = para_list[2:]
            z_list = [RaRQb(w, R0, R1, Q0_pair) for i in w]
        elif serial == 23:
            # ecm_3_023	(R0R1Q0)
            # def aRRQb(w, R0, R1, Q0_pair):
            R0, R1 = para_list[0:2]
            Q0_pair = para_list[2:]
            z_list = [aRRQb(w, R0, R1, Q0_pair) for i in w]
        elif serial == 24:
            # ecm_3_024	RQ0Q1
            # def RQQ(w, R0, Q0_pair, Q1_pair):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            Q1_pair = para_list[3:]
            z_list = [RQQ(w, R0, Q0_pair, Q1_pair) for i in w]
        elif serial == 25:
            # ecm_3_025	(R0Q0)Q1
            # def aRQbQ(w, R0, Q0_pair, Q1_pair):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            Q1_pair = para_list[3:]
            z_list = [aRQbQ(w, R0, Q0_pair, Q1_pair) for i in w]
        elif serial == 26:
            # ecm_3_026	R(Q0Q1)
            # def RaQQb(w, R0, Q0_pair, Q1_pair):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            Q1_pair = para_list[3:]
            z_list = [RaQQb(w, R0, Q0_pair, Q1_pair) for i in w]
        elif serial == 27:
            # 3_027 C0C1Q0
            # def CCQ(w, C0, C1, Q0_pair):
            C0, C1 = para_list[:2]
            Q0_pair = para_list[2:]
            z_list = [CCQ(w, C0, C1, Q0_pair) for i in w]
        elif serial == 28:
            # 3_028 C0(C1Q0)
            # def CaCQb(w, C0, C1, Q0_pair):
            C0, C1 = para_list[:2]
            Q0_pair = para_list[2:]
            z_list = [CaCQb(w, C0, C1, Q0_pair) for i in w]
        elif serial == 29:
            # 3_029 (C0C1Q0)
            # def aCCQb(w, C0, C1, Q0_pair):
            C0, C1 = para_list[:2]
            Q0_pair = para_list[2:]
            z_list = [aCCQb(w, C0, C1, Q0_pair) for i in w]
        elif serial == 30:
            # 3_030 (C0C1)Q0
            # def aCCbQ(w, C0, C1, Q0_pair):
            C0, C1 = para_list[:2]
            Q0_pair = para_list[2:]
            z_list = [aCCbQ(w, C0, C1, Q0_pair) for i in w]
        elif serial == 31:
            # 3_031 C0Q0Q1
            # def CQQ(w, C0, Q0_pair, Q1_pair):
            C0 = para_list[0]
            Q0_pair = para_list[1:3]
            Q1_pair = para_list[3:5]
            z_list = [CQQ(w, C0, Q0_pair, Q1_pair) for i in w]
        elif serial == 32:
            # 3_032 C0(Q0Q1)
            # def CaQQb(w, C0, Q0_pair, Q1_pair):
            C0 = para_list[0]
            Q0_pair = para_list[1:3]
            Q1_pair = para_list[3:5]
            z_list = [CaQQb(w, C0, Q0_pair, Q1_pair) for i in w]
        elif serial == 33:
            # 3_033 Q0(C0Q1)
            # def QaCQb(w, C0, Q0_pair, Q1_pair):
            C0 = para_list[0]
            Q0_pair = para_list[1:3]
            Q1_pair = para_list[3:5]
            z_list = [QaCQb(w, C0, Q0_pair, Q1_pair) for i in w]
        elif serial == 34:
            # 3_034 (C0Q0Q1)
            # def aCQQb(w, C0, Q0_pair, Q1_pair):
            C0 = para_list[0]
            Q0_pair = para_list[1:3]
            Q1_pair = para_list[3:5]
            z_list = [aCQQb(w, C0, Q0_pair, Q1_pair) for i in w]
        elif serial == 35:
            # 3_035 (Q0(R0Q1))
            # def aQaRQbb(w, Q0_pair, R0, Q1_pair):
            Q0_pair = para_list[:2]
            R0 = para_list[2]
            Q1_pair = para_list[3:]
            z_list = [aQaRQbb(w, Q0_pair, R0, Q1_pair) for i in w]
        # 3_036 (R(RO))
        # 3_037
        # 3_038
        # ...
        elif serial == 51:
            # 3_051 (C0(C1Q0))
            # def aC0aC1Q0bb(w, C0, C1, Q0_pair):
            C0, C1 = para_list[:2]
            Q0_pair = para_list[2:]
            z_list = [aC0aC1Q0bb(w, C0, C1, Q0_pair) for i in w]
    elif element_num == 4:
        if serial == 1:
            # ecm_4_001	(C0(R0(C1R1)))
            # def aCaRaCRbbb(w, C0, R0, C1, R1):
            C0, R0, C1, R1 = para_list
            z_list = [aCaRaCRbbb(i, C0, R0, C1, R1) for i in w]
        elif serial == 2:
            # ecm_4_002	(C0(R0(L0R1)))
            # def aCaRaLRbbb(w, C0, R0, L0, R1):
            C0, R0, L0, R1 = para_list
            z_list = [aCaRaLRbbb(i, C0, R0, L0, R1) for i in w]
        elif serial == 3:
            #  ecm_4_003 (C0R0(C1R1))
            # def aCRaCRbb(w,C0, R0, C1, R1):
            C0, R0, C1, R1 = para_list
            z_list = [aCRaCRbb(i, C0, R0, C1, R1) for i in w]
        elif serial == 4:
            #  ecm_4_004 (C0R0(L0R1))
            # def aCRaLRbb(w, C0, R0, L0, R1):
            C0, R0, L0, R1 = para_list
            z_list = [aCRaLRbb(i, C0, R0, L0, R1) for i in w]
        elif serial == 5:
            # ecm_4_005	L0R0(C0R1)
            # def LRaCRb(w, C0, L0, R0, R1):
            C0, L0, R0, R1 = para_list
            z_list = [LRaCRb(i, C0, L0, R0, R1) for i in w]
        elif serial == 6:
            #  ecm_4_006 L0R0(Q0R1)
            # def LRaQRb(w, L0, R0, R1, Q0_pair):
            L0, R0, R1 = para_list[:3]
            Q0_pair = para_list[3:]
            z_list = [LRaQRb(i, L0, R0, R1, Q0_pair) for i in w]
        elif serial == 7:
            # ecm_4_007	R0(Q0(R1W0))
            # DPFC: ECM-4 R(Q(RW)) --> R0(Q0(R1W0))
            # def RaQaRWbb(w, R0, Q0_pair, R1, W0):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            W0 = para_list[-1]
            z_list = [RaQaRWbb(w, R0, Q0_pair, R1, W0) for i in w]
        elif serial == 8:
            # ecm_4_008	R0(C0R1)W0
            # def RaCRbW(w, R0, C0, R1, W0):
            R0, C0, R1, W0 = para_list
            z_list = [RaCRbW(w, R0, C0, R1, W0) for i in w]
        elif serial == 9:
            # ecm_4_009 R0(C0(R1W0))
            # def RaCaRWbb(w, R0, C0, R1, W0):
            R0, C0, R1, W0 = para_list
            z_list = [RaCaRWbb(w, R0, C0, R1, W0) for i in w]
        elif serial == 10:
            # ecm_4_010 R0(C0(R1T0))
            # def RaCaRTbb(w, R0, C0, R1, T0):
            R0, C0, R1, T0 = para_list
            z_list = [RaCaRTbb(w, R0, C0, R1, T0) for i in w]
        elif serial == 11:
            # ecm_4_011 R0(C0(R1O0))
            # def RaCaRObb(w, R0, C0, R1, O0)
            R0, C0, R1, O0 = para_list
            z_list = [RaCaRObb(w, R0, C0, R1, O0) for i in w]
        elif serial == 12:
            # ecm_4_012 R0(C0R1)O0
            # def RaCRbO(w, R0, C0, R1, O0):
            R0, C0, R1, O0 = para_list
            z_list = [RaCRbO(w, R0, C0, R1, O0) for i in w]
        elif serial == 13:
            # ecm_4_013 R0(C0R1)T0
            # def RaCRbT(w, R0, C0, R1, T0):
            R0, C0, R1, T0 = para_list
            z_list = [RaCRbT(w, R0, C0, R1, T0) for i in w]
        elif serial == 14:
            # ecm_4_014 R0(Q0(R1O0))
            # def RaQaRObb(w, R0, Q0_pair, R1, O0):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, O0 = para_list[3:]
            z_list = [RaQaRObb(w, R0, Q0_pair, R1, O0) for i in w]
        elif serial == 15:
            # 4_015 R0(Q0R1)W0
            # def RaQRbW(w, R0, Q0_pair, R1, W0):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, W0 = para_list[3:]
            z_list = [RaQRbW(w, R0, Q0_pair, R1, W0) for i in w]
    elif element_num == 5:
        if serial == 1:
            # ecm_5_001	L0R0(C0(Q0R1))
            # def LRaCaQRbb(w, L0, R0, C0, Q0_pair, Q1_pair, R1)
            L0, R0, C0 = para_list[:3]
            Q0_pair = para_list[3:5]
            Q1_pair = para_list[5:7]
            R1 = para_list[7]
            z_list = [LRaCaQRbb(i, L0, R0, C0, Q0_pair, Q1_pair, R1) for i in w]
        elif serial == 2:
            # ecm_5_002	L0R0(C0R1)Q0
            # def LRaCRbQ(w, L0, R0, C0, R1, Q0_pair):
            L0, R0, C0, R1 = para_list[:4]
            Q0_pair = para_list[4:]
            z_list = [LRaCRbQ(w, L0, R0, C0, R1, Q0_pair) for i in w]
        elif serial == 3:
            # ecm_5_003	L0R0Q0(C0R1)
            # def LRQaCRb(w, L0, R0, Q0_pair, C0, R1):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            C0, R1 = para_list[4:]
            z_list = [LRQaCRb(w, L0, R0, Q0_pair, C0, R1) for i in w]
        elif serial == 4:
            # ecm_5_004	L0R0Q0(Q1R1)
            # def LRQaQRb(w, L0, R0, Q0_pair, Q1_pair,R1):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            Q1_pair = para_list[4:6]
            R1 = para_list[-1]
            z_list = [LRQaQRb(w, L0, R0, Q0_pair, Q1_pair, R1) for i in w]
        elif serial == 5:
            # ecm_5_005	R0(C0(R1(C1R2)))
            # def RaCaRaCRbbb(w, R0, C0, R1, C1, R2):
            R0, C0, R1, C1, R2 = para_list
            z_list = [RaCaRaCRbbb(w, R0, C0, R1, C1, R2) for i in w]
        elif serial == 6:
            # ecm_5_006	R0(C0R1(C1R2))
            # def RaCRaCRbb(w, R0, C0, R1, C1, R2):
            R0, C0, R1, C1, R2 = para_list
            z_list = [RaCRaCRbb(w, R0, C0, R1, C1, R2) for i in w]
        elif serial == 7:
            # ecm_5_007	R0(C0R1(L0R2))
            # def RaCRaLRbb(w, R0, C0, R1, L0, R2):
            R0, C0, R1, L0, R2 = para_list
            z_list = [RaCRaLRbb(w, R0, C0, R1, L0, R2) for i in w]
        elif serial == 8:
            # ecm_5_008	R0(C0R1(Q0R2))
            # def RaCRaQRbb(w, R0, C0, R1,  Q0_pair, R2):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2 = para_list[-1]
            z_list = [RaCRaQRbb(w, R0, C0, R1, Q0_pair, R2) for i in w]
        elif serial == 9:
            # ecm_5_009	R0(C0R1)(C1R2)
            # def RaCRbaCRb(w, R0, C0, R1, C1, R2):
            R0, C0, R1, C1, R2 = para_list
            z_list = [RaCRbaCRb(w, R0, C0, R1, C1, R2) for i in w]
        elif serial == 10:
            # ecm_5_010	R0(Q0(R1(C0R2)))
            # def RaQaRaCRbbb(w, R0, Q0_pair, R1, C0, R2):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, C0, R2 = para_list[3:]
            z_list = [RaQaRaCRbbb(w, R0, Q0_pair, R1, C0, R2) for i in w]
        elif serial == 11:
            # ecm_5_011	R0(Q0(R1(Q1R2)))
            # def RaQaRaQRbbb(w, R0, Q0_pair, R1,  Q1_pair, R2):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2 = para_list[6]
            z_list = [RaQaRaQRbbb(w, R0, Q0_pair, R1, Q1_pair, R2) for i in w]
        elif serial == 12:
            # ecm_5_012	R0(Q0R1(L0R2))
            # def RaQRaLRbb(w, R0, Q0_pair, R1, L0, R2):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, L0, R2 = para_list[3:]
            z_list = [RaQRaLRbb(w, R0, Q0_pair, R1, L0, R2) for i in w]
        elif serial == 13:
            # ecm_5_013	R0(Q0R1(O0R2))
            # def RaQRaORbb(w, R0, Q0_pair, R1, O0, R2):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, O0, R2 = para_list[3:]
            z_list = [RaQRaORbb(w, R0, Q0_pair, R1, O0, R2) for i in w]
        elif serial == 14:
            # ecm_5_014	R0(Q0R1(Q1R2))
            # def RaQRaQRbb(w, R0, Q0_pair, R1, Q1_pair, R2):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2 = para_list[6]
            z_list = [RaQRaQRbb(w, R0, Q0_pair, R1, Q1_pair, R2) for i in w]
        elif serial == 15:
            # ecm_5_015	R(QR)(QR) --> R0(Q0R1)(Q1R2)
            # DPFC: ECM-2 R(QR)(QR)
            # def RaQRbaQRb(w, R0, Q0_pair, R1, Q1_pair, R2):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2 = para_list[6]
            z_list = [RaQRbaQRb(w, R0, Q0_pair, R1, Q1_pair, R2) for i in w]
        elif serial == 16:
            # 5_016 R(RC)(RC) --> R0(R1C0)(R2C1)
            # def RaRCbaRCb(w, R0, R1, C0, R2, C1):
            R0, R1, C0, R2, C1 = para_list
            z_list = [RaRCbaRCb(w, R0, R1, C0, R2, C1) for i in w]
        elif serial == 17:
            # 5_017
            # DPFC: ECM-8 R(Q(RW))Q --> R0(Q0(R1W0))Q0
            # def RaQaRWbbQ(w, R0, Q0_pair, R1, W0, Q1_pair):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, W0 = para_list[3:5]
            Q1_pair = para_list[5:]
            z_list = [RaQaRWbbQ(w, R0, Q0_pair, R1, W0, Q1_pair) for i in w]
        # elif serial == 18:
        #     z_list = [for i in w]
        # elif serial == 19:
        #     z_list = [for i in w]
        elif serial == 20:
            # 5_020 R0(R1C0)(R2W0)
            # def RaRCbaRWb(w, R0, R1, C0, R2, sigma):
            R0, R1, C0, R2, sigma = para_list
            z_list = [RaRCbaRWb(w, R0, R1, C0, R2, sigma) for i in w]
    elif element_num == 6:
        if serial == 1:
            # ecm_6_001	(C0R0(C1R1)(C2R2))
            # def aCRaCRbaCRbb(w, C0, R0, C1, R1, C2, R2):
            C0, R0, C1, R1, C2, R2, C0, R0, C1, R1, C2, R2 = para_list
            z_list = [aCRaCRbaCRbb(i, C0, R0, C1, R1, C2, R2) for i in w]
        elif serial == 2:
            # ecm_6_002	(C0R0(L0R1)(L1R2))
            # def aCRaLRbaLRbb(w, C0, R0, L0, R1, L1, R2):
            C0, R0, L0, R1, L1, R2 = para_list
            z_list = [aCRaLRbaLRbb(i, C0, R0, L0, R1, L1, R2) for i in w]
        elif serial == 3:
            # ecm_6_003	L0R0(C0(R1(C1R2)))
            # def LRaCaRaCRbbb(w, L0, R0, C0, R1, C1, R2):
            L0, R0, C0, R1, C1, R2 = para_list
            z_list = [LRaCaRaCRbbb(i, L0, R0, C0, R1, C1, R2) for i in w]
        elif serial == 4:
            # ecm_6_004	L0R0(C0R1(Q0R2))
            # def LRaCRaQRbb(w, L0, R0, C0, R1, Q0_pair, R2):
            L0, R0, C0, R1 = para_list[:4]
            Q0_pair = para_list[4:6]
            R2 = para_list[-1]
            z_list = [LRaCRaQRbb(i, L0, R0, C0, R1, Q0_pair, R2) for i in w]
        elif serial == 5:
            # ecm_6_005	L0R0(C0R1)(Q0R2)
            # def LRaCRbaQRb(w, L0, R0, C0, R1, Q0_pair, R2):
            L0, R0, C0, R1 = para_list[:4]
            Q0_pair = para_list[4:6]
            R2 = para_list[6]
            z_list = [LRaCRbaQRb(i, L0, R0, C0, R1, Q0_pair, R2) for i in w]
        elif serial == 6:
            # ecm_6_006 L0R0(Q0(R1(C0R2)))
            # def LRaQaRaCRbbb(w, L0, R0, Q0_pair, R1, C0, R2):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            R1, C0, R2 = para_list[4:]
            z_list = [LRaQaRaCRbbb(i, L0, R0, Q0_pair, R1, C0, R2) for i in w]
        elif serial == 7:
            # ecm_6_007	L0R0(Q0(R1(Q1R2)))
            # def LRaQaRaQRbbb(w, L0, R0, Q0_pair, R1, Q1_pair, R2):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            R1 = para_list[4]
            Q1_pair = para_list[5:7]
            R2 = para_list[7]
            z_list = [LRaQaRaQRbbb(i, L0, R0, Q0_pair, R1, Q1_pair, R2) for i in w]
        elif serial == 8:
            # ecm_6_008	L0R0(Q0R1(C0R2))
            # def LRaQRaCRbb(w, L0, R0, Q0_pair, R1, C0, R2):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            R1, C0, R2 = para_list[4:]
            z_list = [LRaQRaCRbb(i, L0, R0, Q0_pair, R1, C0, R2) for i in w]
        elif serial == 9:
            # ecm_6_009 L0R0(Q0R1)(C0R2)
            # def LRaQRbaCRb(w,L0, R0, Q0_pair, R1, C0, R2):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            R1, C0, R2 = para_list[4:]
            z_list = [LRaQRbaCRb(i, L0, R0, Q0_pair, R1, C0, R2) for i in w]
        elif serial == 10:
            # ecm_6_010	L0R0(Q0R1)(Q1R2)
            # def LRaQRbaQRb(w,L0, R0, Q0_pair, R1, Q1_pair, R2):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            R1 = para_list[4]
            Q1_pair = para_list[5:7]
            R2 = para_list[-1]
            z_list = [LRaQRbaQRb(i, L0, R0, Q0_pair, R1, Q1_pair, R2) for i in w]
        elif serial == 11:
            # ecm_6_011 R0(C0(R1(C1(R2O0))))
            # def RaCaRaCaRObbbb(w, R0, C0, R1, C1, R2, O0):
            R0, C0, R1, C1, R2, O0 = para_list
            z_list = [RaCaRaCaRObbbb(i, R0, C0, R1, C1, R2, O0) for i in w]
        elif serial == 12:
            # ecm_6_012 R0(C0(R1(C1(R2T0))))
            # def RaCaRaCaRTbbbb(w, R0, C0, R1, C1, R2, T0):
            R0, C0, R1, C1, R2, T0 = para_list
            z_list = [RaCaRaCaRTbbbb(i, R0, C0, R1, C1, R2, T0) for i in w]
        elif serial == 13:
            # ecm_6_013 R0(C0(R1(C1(R2W0))))
            # def RaCaRaCaRWbbbb(w, R0, C0, R1, C1, R2, W0):
            R0, C0, R1, C1, R2, W0 = para_list
            z_list = [RaCaRaCaRWbbbb(i, R0, C0, R1, C1, R2, W0) for i in w]
        elif serial == 14:
            # ecm_6_014 R0(C0(R1(Q0(R2W0))))
            # def RaCaRaQaRWbbbb(w, R0, C0, R1, Q0_pair, R2, W0):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, W0 = para_list[5:]
            z_list = [RaCaRaQaRWbbbb(i, R0, C0, R1, Q0_pair, R2, W0) for i in w]
        elif serial == 15:
            # ecm_6_015 R0(C0R1)(C1R2)O0
            # def RaCRbaCRbO(w, R0, C0, R1, C1, R2, O0):
            R0, C0, R1, C1, R2, O0 = para_list
            z_list = [RaCRbaCRbO(i, R0, C0, R1, C1, R2, O0) for i in w]
        elif serial == 16:
            # ecm_6_016 R0(C0R1)(C1R2)S0
            # def RaCRbaCRbS(w, R0, C0, R1, C1, R2, S0):
            R0, C0, R1, C1, R2, S0 = para_list
            z_list = [RaCRbaCRbS(i, R0, C0, R1, C1, R2, S0) for i in w]
        elif serial == 17:
            # ecm_6_017 R0(C0R1)(C1R2)T0
            # def RaCRbaCRbT(w, R0, C0, R1, C1, R2, T0):
            R0, C0, R1, C1, R2, T0 = para_list
            z_list = [RaCRbaCRbT(i, R0, C0, R1, C1, R2, T0) for i in w]
        elif serial == 18:
            # ecm_6_018 R0(C0R1)(C1R2)W0
            # def RaCRbaCRbW(w, R0, C0, R1, C1, R2, W0):
            R0, C0, R1, C1, R2, W0 = para_list
            z_list = [RaCRbaCRbW(i, R0, C0, R1, C1, R2, W0) for i in w]
        elif serial == 19:
            # ecm_6_019 R0(C0R1Q0(R2W0))
            # def RaCRQaRWbb(w, R0, C0, R1, Q0_pair, R2, W0):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, W0 = para_list[5:]
            z_list = [RaCRQaRWbb(i, R0, C0, R1, Q0_pair, R2, W0) for i in w]
        elif serial == 20:
            # ecm_6_020 R0(Q0(R1(Q1(R2W0))))
            # def RaQaRaQaRWbbbb(w, R0,Q0_pair, R1, Q1_pair, R2, W0):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2, W0 = para_list[6:]
            z_list = [RaQaRaQaRWbbbb(i, R0, Q0_pair, R1, Q1_pair, R2, W0) for i in w]
        elif serial == 21:
            # ecm_6_021 R0(Q0(R1W0))(C0R2)
            # def RaQaRWbbaCRb(w, R0, Q0_pair, R1, W0, C0, R2):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, W0, C0, R2 = para_list[3:]
            z_list = [RaQaRWbbaCRb(i, R0, Q0_pair, R1, W0, C0, R2) for i in w]
        elif serial == 22:
            # ecm_6_022 R0(Q0(R1W0))(Q1R2)
            # def RaQaRWbbaQRb(w, R0, Q0_pair, R1, W0, Q1_pair, R2):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, W0 = para_list[3:5]
            Q1_pair = para_list[5:7]
            R2 = para_list[7]
            z_list = [RaQaRWbbaQRb(i, R0, Q0_pair, R1, W0, Q1_pair, R2) for i in w]
        elif serial == 23:
            # ecm_6_023 R0(Q0R1)(R2(O0R3))
            # def RaQRbaRaORbb(w, R0, Q0_pair, R1, R2, O0, R3):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, R2, O0, R3 = para_list[3:]
            z_list = [RaQRbaRaORbb(i, R0, Q0_pair, R1, R2, O0, R3) for i in w]
        elif serial == 24:
            # ecm_6_024  R0(Q0R1)(Q1R2)W0
            # def RaQRbaQRbW(w, R0, Q0_pair, R1, Q1_pair, R2, W0):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2, W0 = para_list[6:]
            z_list = [RaQRbaQRbW(w, R0, Q0_pair, R1, Q1_pair, R2, W0) for i in w]
        elif serial == 25:
            # ecm_6_025 R0(Q0R1)(Q1(R2W0))
            # DPFC: ECM-6 R(QR)(Q(RW)) --> R0(Q0R1)(Q1(R2W0))
            # def RaQRbaQaRWbb(w, R0, Q0_pair, R1, Q1_pair, R2, W0):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2, W0 = para_list[6:]
            z_list = [RaQRbaQaRWbb(i, R0, Q0_pair, R1, Q1_pair, R2, W0) for i in w]
        elif serial == 26:
            # ecm_6_025ecm_6_025 R0(C0R1)(C1(R2W0))
            # DPFC: ECM-10 R(CR)(C(RW)) --> R0(C0R1)(C1(R2W0))
            # def RaCRbaCaRWbb(w, R0, C0, R1, C1, R2, W0):
            R0, C0, R1, C1, R2, W0 = para_list
            z_list = [RaCRbaCaRWbb(i, R0, C0, R1, C1, R2, W0) for i in w]
    elif element_num == 7:
        if serial == 1:
            # ecm_7_001 R0(C0(R1(C1(R2(C2R3)))))
            # def RaCaRaCaRaCRbbbbb(w, R0, C0, R1, C1, R2, C2, R3):
            R0, C0, R1, C1, R2, C2, R3 = para_list
            z_list = [RaCaRaCaRaCRbbbbb(i, R0, C0, R1, C1, R2, C2, R3) for i in w]
        elif serial == 2:
            # ecm_7_002 R0(C0(R1(Q0(R2(C1R3)))))
            # def RaCaRaQaRaCRbbbbb(w, R0, C0, R1, Q0_pair, R2, C1, R3):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, C1, R3 = para_list[5:]
            z_list = [RaCaRaQaRaCRbbbbb(i, R0, C0, R1, Q0_pair, R2, C1, R3) for i in w]
        elif serial == 3:
            # ecm_7_003 R0(C0(R1(Q0(R2(Q1R3)))))
            # def RaCaRaQaRaQRbbbbb(w, R0, C0, R1, Q0_pair, R2, Q1_pair, R3):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2 = para_list[5]
            Q1_pair = para_list[6:8]
            R3 = para_list[8]
            z_list = [RaCaRaQaRaQRbbbbb(i, R0, C0, R1, Q0_pair, R2, Q1_pair, R3) for i in w]
        elif serial == 4:
            # ecm_7_004 R(C(R(QR)))(CR) --> R0(C0(R1(Q0R2)))(C1R3)
            # def RaCaRaQRbbbaCRb(w, R0, C0, R1, Q0_pair, R2, C1, R3):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, C1, R3 = para_list[5:]
            z_list = [RaCaRaQRbbbaCRb(i, R0, C0, R1, Q0_pair, R2, C1, R3) for i in w]
        elif serial == 5:
            # ecm_7_005 R0(C0R1(C1R2)(C2R3))
            # def RaCRaCRbaCRbb(w, R0, C0, R1, C1, R2, C2, R3):
            R0, C0, R1, C1, R2, C2, R3 = para_list
            z_list = [RaCRaCRbaCRbb(i, R0, C0, R1, C1, R2, C2, R3) for i in w]
        elif serial == 6:
            # ecm_7_006 R0(C0R1(Q0R2)(Q1R3))
            # def RaCRaQRbaQRbb(w, R0, C0, R1, Q0_pair, R2, Q1_pair, R3):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2 = para_list[5]
            Q1_pair = para_list[6:8]
            R3 = para_list[-1]
            z_list = [RaCRaQRbaQRbb(i, R0, C0, R1, Q0_pair, R2, Q1_pair, R3) for i in w]
        elif serial == 7:
            # ecm_7_007 R0(Q0(R1(Q1(R2(C0R3)))))
            # def RaQaRaQaRaCRbbbbb(w, R0, R1, R2, C0, R3, Q0_pair, Q1_pair ):
            R0, R1, R2, C0, R3 = para_list[:5]
            Q0_pair = para_list[5:7]
            Q1_pair = para_list[7:9]
            z_list = [RaQaRaQaRaCRbbbbb(i, R0, R1, R2, C0, R3, Q0_pair, Q1_pair) for i in w]
        elif serial == 8:
            # ecm_7_008 R0(C0R1(Q0R2))(C1R3)
            # def RaCRaQRbbaCRb(w, R0, C0, R1, Q0_pair, R2, C1, R3 ):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, C1, R3 = para_list[5:7]
            z_list = [RaCRaQRbbaCRb(i, R0, C0, R1, Q0_pair, R2, C1, R3) for i in w]
        elif serial == 9:
            # ecm_7_009 R0(C0R1)(C1R2)(C2R3)
            # def RaCRbaCRbaCRb(w, R0, C0, R1, C1, R2, C2, R3):
            R0, C0, R1, C1, R2, C2, R3 = para_list
            z_list = [RaCRbaCRbaCRb(i, R0, C0, R1, C1, R2, C2, R3) for i in w]
        elif serial == 10:
            # ecm_7_010 R0(C0R1)(Q0R2)(C1R3)
            # def RaCRbaQRbaCRb(w, R0, C0, R1, Q0_pair, R2, C1, R3):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, C1, R3 = para_list[5:]
            z_list = [RaCRbaQRbaCRb(i, R0, C0, R1, Q0_pair, R2, C1, R3) for i in w]
        elif serial == 11:
            # ecm_7_011 R0(Q0R1(C0R2)(O0R3))
            # def RaQRaCRbaORbb(w, R0, Q0_pair, R1, C0, R2, O0, R3):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            C0, R2, O0, R3 = para_list[4:]
            z_list = [RaQRaCRbaORbb(i, R0, Q0_pair, R1, C0, R2, O0, R3) for i in w]
        elif serial == 12:
            # ecm_7_012 R0(Q0R1(C0R2)(L0R3))
            # def RaQRaCRbaLRbb(w, R0, Q0_pair, R1, C0, R2, L0, R3):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, C0, R2, L0, R3 = para_list[3:]
            z_list = [RaQRaCRbaLRbb(i, R0, Q0_pair, R1, C0, R2, L0, R3) for i in w]
        elif serial == 13:
            # ecm_7_013 R0(Q0R1)(Q1R2)(C0R3)
            # def RaQRbaQRbaCRb(w, R0, Q0_pair, R1, Q1_pair, R2, C0, R3):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2, C0, R3 = para_list[6:]
            z_list = [RaQRbaQRbaCRb(i, R0, Q0_pair, R1, Q1_pair, R2, C0, R3) for i in w]
        elif serial == 14:
            # ecm_7_014 R0(Q0R1(L0R2)(L1R3))
            # def RaQRaLRbaLRbb(w, R0, Q0_pair, R1, L0, R2, L1, R3):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, L0, R2, L1, R3 = para_list[3:]
            z_list = [RaQRaLRbaLRbb(i, R0, Q0_pair, R1, L0, R2, L1, R3) for i in w]
        elif serial == 15:
            # ecm_7_015 R0(Q0R1(Q1R2)(C0R3))
            # def RaQRaQRbaCRbb(w, R0, Q0_pair, R1, Q1_pair, R2, C0, R3):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2, C0, R3 = para_list[6:]
            z_list = [RaQRaQRbaCRbb(i, R0, Q0_pair, R1, Q1_pair, R2, C0, R3) for i in w]
        elif serial == 16:
            # ecm_7_016 R0(Q0R1(Q1R2)(O0R3))
            # def RaQRaQRbaORbb(w, R0, Q0_pair, R1, Q1_pair, R2, O0, R3):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2, O0, R3 = para_list[6:]
            z_list = [RaQRaQRbaORbb(i, R0, Q0_pair, R1, Q1_pair, R2, O0, R3) for i in w]
        elif serial == 17:
            # ecm_7_017 R0(Q0R1(C0R2)(C1R3))
            # def RaQRaCRbaCRbb(w, R0, Q0_pair, R1, C0, R2, C1, R3):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, C0, R2, C1, R3 = para_list[3:]
            z_list = [RaQRaCRbaCRbb(i, R0, Q0_pair, R1, C0, R2, C1, R3) for i in w]
        elif serial == 18:
            # 7_018 R(CR(RW))(QR)
            # DPFC: ECM-11 R0(C0R1(R2W0))(Q0R3)
            # def RaCRaRWbbaQRb(w, R0, C0, R1, R2, W0, Q0_pair, R3):
            R0, C0, R1, R2, W0 = para_list[:5]
            Q0_pair = para_list[5:7]
            R3 = para_list[7]
            z_list = [RaCRaRWbbaQRb(i, R0, C0, R1, R2, W0, Q0_pair, R3) for i in w]
    elif element_num == 8:
        if serial == 1:
            # ecm_8_001  (C0R0(C1R1)(C2R2)(C3R3))
            # def aCRaCRbaCRbaCRbb(w, C0, R0, C1, R1, C2, R2, C3, R3):
            C0, R0, C1, R1, C2, R2, C3, R3 = para_list
            z_list = [aCRaCRbaCRbaCRbb(i, C0, R0, C1, R1, C2, R2, C3, R3) for i in w]
        elif serial == 2:
            # ecm_8_002  (C0R0(L0R1)(L1R2)(L2R3))
            # def aCRaLRbaLRbaLRbb(w, C0, R0, L0, R1, L1, R2, L2, R3):
            C0, R0, L0, R1, L1, R2, L2, R3 = para_list
            z_list = [aCRaLRbaLRbaLRbb(i, C0, R0, L0, R1, L1, R2, L2, R3) for i in w]
        elif serial == 3:
            # ecm_8_003  L0R0(C0(R1(Q0R2)))(C1R3)
            # def LRaCaRaQRbbbaCRb(w, L0, R0, C0, R1, Q0_pair, R2, C1, R3):
            L0, R0, C0, R1 = para_list[:4]
            Q0_pair = para_list[4:6]
            R2, C1, R3 = para_list[6:]
            z_list = [LRaCaRaQRbbbaCRb(i, L0, R0, C0, R1, Q0_pair, R2, C1, R3) for i in w]
        elif serial == 4:
            # ecm_8_004  L0R0(C0R1(Q0R2)(R3W0))
            # def LRaCRaQRbaRWbb(w, L0, R0, C0, R1, Q0_pair, R2, R3, W0):
            L0, R0, C0, R1 = para_list[:4]
            Q0_pair = para_list[4:6]
            R2, R3, W0 = para_list[6:]
            z_list = [LRaCRaQRbaRWbb(i, L0, R0, C0, R1, Q0_pair, R2, R3, W0) for i in w]
        elif serial == 5:
            # ecm_8_005 L0R0(C0R1(Q0R2))(C1R3)
            # def LRaCRaQRbbaCRb(w, L0, R0, C0, R1, Q0_pair, R2, C1, R3):
            L0, R0, C0, R1 = para_list[:4]
            Q0_pair = para_list[4:6]
            R2, C1, R3 = para_list[6:]
            z_list = [LRaCRaQRbbaCRb(i, L0, R0, C0, R1, Q0_pair, R2, C1, R3) for i in w]
        elif serial == 6:
            # ecm_8_006 L0R0(Q0(R1(L1R2)(C0R3)))
            # def LRaQaRaLRbaCRbbb(w, L0, R0, Q0_pair, R1, L1, R2, C0, R3):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            R1, L1, R2, C0, R3 = para_list[4:]
            z_list = [LRaQaRaLRbaCRbbb(i, L0, R0, Q0_pair, R1, L1, R2, C0, R3) for i in w]
        elif serial == 7:
            # ecm_8_007 LR(QR(LR)(CR)) --> L0R0(Q0R1(L1R2)(C0R3))
            # def LRaQRaLRbaCRbb(w, L0, R0, Q0_pair, R1, L1, R2, C0, R3):
            L0, R0 = para_list[:2]
            Q0_pair = para_list[2:4]
            R1, L1, R2, C0, R3 = para_list[4:]
            z_list = [LRaQRaLRbaCRbb(i, L0, R0, Q0_pair, R1, L1, R2, C0, R3) for i in w]
        elif serial == 8:
            # ecm_8_008 R0(C0(R1(Q0(R2(C1(R3W0))))))
            # def RaCaRaQaRaCaRWbbbbbb(w, R0, C0, R1, Q0_pair, R2, C1, R3, W0):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, C1, R3, W0 = para_list[5:]
            z_list = [RaCaRaQaRaCaRWbbbbbb(i, R0, C0, R1, Q0_pair, R2, C1, R3, W0) for i in w]
        elif serial == 9:
            # ecm_8_009 R(C(R(Q(RW))))(CR) --> R0(C0(R1(Q0(R2W0))))(C1R3)
            # def RaCaRaQaRWbbbbaCRb(w, R0, C0, R1, Q0_pair, R2, W0, C1, R3):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, W0, C1, R3 = para_list[5:]
            z_list = [RaCaRaQaRWbbbbaCRb(i, R0, C0, R1, Q0_pair, R2, W0, C1, R3) for i in w]
        elif serial == 10:
            # ecm_8_010 L0R0(C0(R1(Q0(R2(C1R3)))))
            # def LRaCaRaQaRaCRbbbbb(w, L0, R0, C0, R1, Q0_pair, R2, C1, R3):
            L0, R0, C0, R1 = para_list[:4]
            Q0_pair = para_list[4:6]
            R2, C1, R3 = para_list[6:]
            z_list = [LRaCaRaQaRaCRbbbbb(i, L0, R0, C0, R1, Q0_pair, R2, C1, R3) for i in w]
        elif serial == 11:
            # ecm_8_011 R0(Q0(R1(Q1R2)(Q2(R3W0))))
            # def RaQaRaQRbaQaRWbbbb(w, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, W0):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2 = para_list[6]
            Q2_pair = para_list[7:9]
            R3, W0 = para_list[9:]
            z_list = [RaQaRaQRbaQaRWbbbb(i, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, W0) for i in w]
    elif element_num == 9:
        if serial == 1:
            # ecm_9_001  R0(Q0R1(O0R2)(L0R3)(L1R4))
            # def RaQRaORbaLRbaLRbb(w, R0, Q0_pair, R1, O0, R2, L0, R3, L1, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, O0, R2, L0, R3, L1, R4 = para_list[3:]
            z_list = [RaQRaORbaLRbaLRbb(i, R0, Q0_pair, R1, O0, R2, L0, R3, L1, R4) for i in w]
        elif serial == 2:
            # ecm_9_002 L0R0(C0(R1(Q0(R2(C1(R3W0))))))
            # def LRaCaRaQaRaCaRWbbbbbb(w, L0, R0, C0, R1, Q0_pair, R2, C1, R3, W0):
            L0, R0, C0, R1 = para_list[:4]
            Q0_pair = para_list[4:6]
            R2, C1, R3, W0 = para_list[6:]
            z_list = [LRaCaRaQaRaCaRWbbbbbb(i, L0, R0, C0, R1, Q0_pair, R2, C1, R3, W0) for i in w]
        elif serial == 3:
            # ecm_9_003 R0(C0(R1(Q0(R2(L0R3)(C1R4)))))
            # def RaCaRaQaRaLRbaCRbbbbb(w, R0, C0, R1, Q0_pair, R2, L0, R3, C1, R4):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, L0, R3, C1, R4 = para_list[5:]
            z_list = [RaCaRaQaRaLRbaCRbbbbb(i, R0, C0, R1, Q0_pair, R2, L0, R3, C1, R4) for i in w]
        elif serial == 4:
            # ecm_9_004 R0(C0R1(L0R2)(L1R3)(O0R4))
            # def RaCRaLRbaLRbaORbb(w, R0, C0, R1, L0, R2, L1, R3, O0, R4):
            R0, C0, R1, L0, R2, L1, R3, O0, R4 = para_list
            z_list = [RaCRaLRbaLRbaORbb(i, R0, C0, R1, L0, R2, L1, R3, O0, R4) for i in w]
        elif serial == 5:
            # ecm_9_005 R0(C0R1(O0R2)(L0R3)(L1R4))
            # def RaCRaORbaLRbaLRbb(w, R0, C0, R1, O0, R2, L0, R3, L1, R4):
            R0, C0, R1, O0, R2, L0, R3, L1, R4 = para_list
            z_list = [RaCRaORbaLRbaLRbb(i, R0, C0, R1, O0, R2, L0, R3, L1, R4) for i in w]
        elif serial == 6:
            # ecm_9_006 R0(C0R1(Q0R2)(L0R3)(C1R4))
            # def RaCRaQRbaLRbaCRbb(w, R0, C0, R1, Q0_pair, R2, L0, R3, C1, R4):
            R0, C0, R1 = para_list[:3]
            Q0_pair = para_list[3:5]
            R2, L0, R3, C1, R4 = para_list[5:]
            z_list = [RaCRaQRbaLRbaCRbb(i, R0, C0, R1, Q0_pair, R2, L0, R3, C1, R4) for i in w]
        elif serial == 7:
            # ecm_9_007 R0(Q0(R1(C0(R2(L0R3)(C1R4)))))
            # def RaQaRaCaRaLRbaCRbbbbb(w, R0, Q0_pair, R1, C0, R2, L0, R3, C1, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, C0, R2, L0, R3, C1, R4 = para_list[3:]
            z_list = [RaQaRaCaRaLRbaCRbbbbb(i, R0, Q0_pair, R1, C0, R2, L0, R3, C1, R4) for i in w]
        elif serial == 8:
            # ecm_9_008 R0(Q0(R1(Q1(R2(Q2(R3(C0R4)))))))
            # def RaQaRaQaRaQaRaCRbbbbbbb(w, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, C0, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2 = para_list[6]
            Q2_pair = para_list[7:9]
            R3, C0, R4 = para_list[9:]
            z_list = [RaQaRaQaRaQaRaCRbbbbbbb(i, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, C0, R4) for i in w]
        elif serial == 9:
            # ecm_9_009 R0(Q0R1(C0R2)(L0R3)(O0R4))
            # def RaQRaCRbaLRbaORbb(w, R0, Q0_pair, R1, C0, R2, L0, R3, O0, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, C0, R2, L0, R3, O0, R4 = para_list[3:]
            z_list = [RaQRaCRbaLRbaORbb(i, R0, Q0_pair, R1, C0, R2, L0, R3, O0, R4) for i in w]
        elif serial == 10:
            # ecm_9_010 R0(Q0R1(C0R2)(R3W0))(C1R4)
            # def RaQRaCRbaRWbbaCRb(w, R0, Q0_pair, R1, C0, R2, R3, W0, C1, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, C0, R2, R3, W0, C1, R4 = para_list[3:]
            z_list = [RaQRaCRbaRWbbaCRb(i, R0, Q0_pair, R1, C0, R2, R3, W0, C1, R4) for i in w]
        elif serial == 11:
            # ecm_9_011 R0(Q0R1(L0R2)(L1R3)(O0R4))
            # def RaQRaLRbaLRbaORbb(w, R0, Q0_pair, R1, L0, R2, L1, R3, O0, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, L0, R2, L1, R3, O0, R4 = para_list[3:]
            z_list = [RaQRaLRbaLRbaORbb(i, R0, Q0_pair, R1, L0, R2, L1, R3, O0, R4) for i in w]
        elif serial == 12:
            # ecm_9_012 R(QR(OR)(LR)(LR)) --> R0(Q0R1(O0R2)(L0R3)(L1R4))
            # def RaQRaORbaLRbaLRbb(w, R0, Q0_pair, R1, O0, R2, L0, R3, L1, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1, O0, R2, L0, R3, L1, R4 = para_list[3:]
            z_list = [RaQRaORbaLRbaLRbb(i, R0, Q0_pair, R1, O0, R2, L0, R3, L1, R4) for i in w]
        elif serial == 13:
            # ecm_9_013 R0(Q0R1(Q1R2)(L0R3)(O0R4))
            # def RaQRaQRbaLRbaORbb(w, R0, Q0_pair, R1, Q1_pair, R2, L0, R3, O0, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2, L0, R3, O0, R4 = para_list[6:]
            z_list = [RaQRaQRbaLRbaORbb(i, R0, Q0_pair, R1, Q1_pair, R2, L0, R3, O0, R4) for i in w]
        elif serial == 14:
            # ecm_9_014 R(QR(QR)(QR)(CR)) ---> R0(Q0R1(Q1R2)(Q2R3)(C0R4))
            # def RaQRaQRbaQRbaCRbb(w, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, C0, R4):
            R0 = para_list[0]
            Q0_pair = para_list[1:3]
            R1 = para_list[3]
            Q1_pair = para_list[4:6]
            R2 = para_list[6]
            Q2_pair = para_list[7:9]
            R3, C0, R4 = para_list[9:]
            z_list = [RaQRaQRbaQRbaCRbb(i, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, C0, R4) for i in w]
    elif element_num == 10:
        if serial == 1:
            # ecm_10_001 (C0R0(C1R1)(C2R2)(C3R3)(C4R4))
            # def aCRaCRbaCRbaCRbaCRbb(w, C0, R0, C1, R1, C2, R2, C3, R3, C4, R4):
            C0, R0, C1, R1, C2, R2, C3, R3, C4, R4 = para_list
            z_list = [aCRaCRbaCRbaCRbaCRbb(i, C0, R0, C1, R1, C2, R2, C3, R3, C4, R4) for i in w]
        elif serial == 2:
            # ecm_10_002 (C0R0(L0R1)(L1R2)(L2R3)(L3R4))
            # def aCRaLRbaLRbaLRbaLRbb(w, C0, R0, L0, R1, L1, R2, L2, R3, L3, R4):
            C0, R0, L0, R1, L1, R2, L2, R3, L3, R4 = para_list
            z_list = [aCRaLRbaLRbaLRbaLRbb(i, C0, R0, L0, R1, L1, R2, L2, R3, L3, R4) for i in w]
        elif serial == 3:
            # ecm_10_003 (QR(OR)(CR)(LR)(QR)) --> (Q0R0(O0R1)(C0R2)(L0R3)(Q1R4))
            # def aQRaORbaCRbaLRbaQRbb(w, Q0_pair, R0, O0, R1, C0, R2, L0, R3, Q1_pair, R4):
            Q0_pair = para_list[:2]
            R0, O0, R1, C0, R2, L0, R3 = para_list[2:9]
            Q1_pair = para_list[9:11]
            R4 = para_list[-1]
            z_list = [aQRaORbaCRbaLRbaQRbb(i, Q0_pair, R0, O0, R1, C0, R2, L0, R3, Q1_pair, R4) for i in w]
    elif element_num == 11:
        if serial == 1:
            # ecm_11_001 R(QR(OR)(LR)(LR)(QR)) --> R0(Q0R1(O0R2)(L0R3)(L1R4)(Q1R5))
            # def RaQRaORbaLRbaLRbaQRbb(w, R0, R1, Q0_pair, R2, O0, R3, L0, R4, L1, Q1_pair, R5):
            R0, R1 = para_list[:2]
            Q0_pair = para_list[2:4]
            R2, O0, R3, L0, R4, L1 = para_list[4:10]
            Q1_pair = para_list[10:12]
            R5 = para_list[-1]
            z_list = [RaQRaORbaLRbaLRbaQRbb(i, R0, R1, Q0_pair, R2, O0, R3, L0, R4, L1, Q1_pair, R5) for i in w]
        elif serial == 2:
            # ecm_11_002 R0(C0R1(L0R2)(L1R3)(O0R4)(Q0R5))
            # def RaCRaLRbaLRbaORbaQRbb(w, R0, C0, R1, L0, R2, L1, R3, O0, R4, Q0_pair, R5):
            R0, C0, R1, L0, R2, L1, R3, O0, R4 = para_list[:9]
            Q0_pair = para_list[9:11]
            R5 = para_list[-1]
            z_list = [RaCRaLRbaLRbaORbaQRbb(i, R0, C0, R1, L0, R2, L1, R3, O0, R4, Q0_pair, R5) for i in w]
    elif element_num == 12:
        pass
    elif element_num == 13:
        pass
    elif element_num == 6:
        pass
    elif element_num == 6:
        pass
    elif element_num == 6:
        pass
    elif element_num == 6:
        pass

    return z_list

def ecm_simulator_0(ecm_num, para_list, fre_list):
    """
    :param
        ecm_num: int
        para_list: [float, p0, p1, ...]
        fre_list: [float, f0, f1, ...]
    :return:
    """
    w_list = [2 * math.pi * f for f in fre_list]
    # ECM-0
    if ecm_num == 0:
        R0, R1, C0 = para_list
        z_list = [RaRCb(w, R0, R1, C0) for w in w_list]
    # ECM-1
    elif ecm_num == 1:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[-1]
        z_list = [RaRQb(w, R0, Q0_pair, R1) for w in w_list]
    # ECM-2
    elif ecm_num == 2:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[3]
        Q1_pair = para_list[4:6]
        R2 = para_list[-1]
        z_list = [RaQRbaQRb(w, R0, Q0_pair, R1, Q1_pair, R2) for w in w_list]
    # ECM-3
    elif ecm_num == 3:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[3]
        L0 = para_list[4]
        R2 = para_list[-1]
        z_list = [RaQRaLRbb(w, R0, Q0_pair, R1, L0, R2) for w in w_list]
    # ECM-4
    elif ecm_num == 4:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[3]
        W0 = para_list[-1]
        z_list = [RaQaRWbb(w, R0, Q0_pair, R1, W0) for w in w_list]
    # ECM-5
    elif ecm_num == 5:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[3]
        Q1_pair = para_list[4:6]
        R2 = para_list[-2]
        W0 = para_list[-1]
        z_list = [RaQRbaQRbW(w, R0, Q0_pair, R1, Q1_pair, R2, W0) for w in w_list]
    # ECM-6
    elif ecm_num == 6:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[3]
        Q1_pair = para_list[4:6]
        R2 = para_list[6]
        W0 = para_list[7]
        z_list = [RaQRbaQaRWbb(w, R0, Q0_pair, R1, Q1_pair, R2, W0) for w in w_list]
    # ECM-7
    elif ecm_num == 7:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[3]
        W0 = para_list[4]
        z_list = [RaQRbW(w, R0, Q0_pair, R1, W0) for w in w_list]
    # ECM-8
    elif ecm_num == 8:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[3]
        W0 = para_list[4]
        Q1_pair = para_list[5:7]
        z_list = [RaQaRWbbQ(w, R0, Q0_pair, R1, W0, Q1_pair) for w in w_list]
    # ECM-9
    elif ecm_num == 9:
        R0 = para_list[0]
        Q0_pair = para_list[1:3]
        R1 = para_list[3]
        Q1_pair = para_list[4:6]
        R2 = para_list[6]
        z_list = [RaQaRaQRbbb(w, R0, Q0_pair, R1, Q1_pair, R2) for w in w_list]
    # ECM-
    # elif ecm_num == :
    #     z_list = [for w in w_list]
    # ECM-
    return z_list
