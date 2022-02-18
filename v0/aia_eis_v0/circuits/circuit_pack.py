import numpy as np
from circuits.elements import ele_C as C
from circuits.elements import ele_L as L
from circuits.elements import ele_Warburg as WB
from circuits.elements import ele_Q as Q
from circuits.elements import ele_O as O
from circuits.elements import ele_T as T
from circuits.elements import ele_S as S

"""
v1
    NEW RULE: ecm_2(two element)_001(Sequence)
    ecm_serial与ECM 函数的具体对应关系记录在dpfc_src\circuits\ecm_serial.xlsx中sheet【formal】对应的 CDC
v0
    Define all the circuits used in this project
        Python 电路模型函数名 命名规则：
            ‘a’ == ‘(’; ‘b’ == ‘)’，直接用字母a替代左括号，用字母b替代右括号
        
        Circuit(ECM) No.    CDC             Function
        0                   R(CR)           RaCRb, Simplified Randles Cell
        0                                   R0aC0R1b
    
        1                   R(QR)           RaQRb
        1                   R(QR)           R0aQ0R1b
    
        2                   R(QR)(QR)       RaQRbaQRb
        2                   R(QR)(QR)       R0aQ0R1baQ1R2b
    
        3                   R(QR(LR))       RaQRaLRbb
        3                   R(QR(LR))       R0aQ0R1aL0R2bb
    
        4                   R(Q(RW))        RaQaRWbb
        4                   R(Q(RW))        R0aQ0aR1W0bb
    
        5                   R(QR)(QR)W      RaQRbaQRbW
        5                   R(QR)(QR)W      R0aQ0R1baQ1R2bW0
    
        6                   R(QR)(Q(RW))    RaQRbaQaRWbb
        6                   R(QR)(Q(RW))    R0aQ0R1baQ1aR2W0bb
    
        7                   R(QR)W          RaQRbW
        7                   R(QR)W          R0aQ0R1bW0
    
        8                   R(Q(RW))Q       RaQaRWbbQ
        8                   R(Q(RW))Q       R0aQ0aR1W0bbQ1
    
        9                   R(Q(R(QR)))     RaQaRaQRbbb
        9                   R(Q(R(QR)))     R0aQ0aR1aQ1R2bbb
        
        Q_pair = (q, n) or [q, n]
            q: CPE coefficient, Constant phase element [s^n/ohm]
            n: Constant phase elelment exponent [-]
        
        WB_sigma: warburg coefficient
"""
# ---------------------------------------- ECM has 2 elements----------------------------------------
# ecm_2_001  R0R1
def RR(w, R0, R1):
    z = R0 + R1
    return z

#ecm_2_002  (R0R1)
def aRRb(w, R0, R1):
    z = 1 / (1 / R0 + 1 / R1)
    return z

#ecm_2_003  R0L0
def RL(w, R0, L0):
    z = R0 + L(w, L0)
    return z

#ecm_2_004  (R0L0)
def aRLb(w, R0, L0):
    z = 1 / (1 / R0 + 1/(L(w, L0)))
    return z

#ecm_2_005  R0C0
def RC(w, R0, C0):
    z = R0 + C(w, C0)
    return z

#ecm_2_006  (R0C0)
def aRCb(w, R0, C0):
    z = 1 / (1 / R0 + 1 / C(w, C0))
    return z

#ecm_2_007  R0Q0
def RQ(w, R0, Q0_pair):
    z = R0 + Q(w, q=Q0_pair[0], n=Q0_pair[1])
    return z
# ---------------------------------------- ECM has 2 elements----------------------------------------


# ---------------------------------------- ECM has 3 elements----------------------------------------
#ecm_3_001  R0R1R2
def RRR(w, R0, R1, R2):
    z = R0 + R1 + R2
    return z

#ecm_3_002	R0(R1R2)
def RaRRb(w, R0, R1, R2):
    z = 1 / (1 / R1 + 1 / R2) + R0
    return z

#ecm_3_003	(R0R1R2)
def aRRRb(w, R0, R1, R2):
    z = 1 / (1 / R0 + 1 / R1 + 1 / R2)
    return z

#ecm_3_004	R0R1L0
def RRL(w, R0, R1, L0):
    z = R0 + R1 + L(w, L0)
    return z

#ecm_3_005	R0(R1L0)
def RaRLb(w, R0, R1, L0):
    z = R0 + 1 / (1 / L(w, L0) + 1 / R1)
    return z

#ecm_3_006	(R0R1)L0
def aRRbL(w, R0,R1,L0):
    z = 1 / (1 / R0 + 1 / R1) + L(w, L0)
    return z

#ecm_3_007	(R0R1L0)
def aRRLb(w,R0,R1,L0):
    z = 1 / (1 / R0 + 1 / R1 + 1 / L(w, L0))
    return z

#ecm_3_008	R0L0L1
def RLL(w,R0,L0,L1):
    z = R0 + L(w, L0) + L(w, L1)
    return z

#ecm_3_009	R0(L0L1)
def RaLLb(w,R0,L0,L1):
    z = R0 + 1/(1 / L(w, L0) + 1 / L(w, L1))
    return z

#ecm_3_010	(R0L0L1)
def aRLLb(w,R0,L0,L1):
    z = 1 / (1 / L(w, L0) + 1 / L(w, L1) + 1 / R0)
    return z

#ecm_3_011	(R0L0)L1
def aRLbL(w,R0,L0,L1):
    z = 1 / (1 / L(w, L0) + 1 / R0) + L(w, L1)
    return z

#ecm_3_012	R0R1C0
def RRC(w, R0,R1,C0):
    z = R0 + R1 + C(w, C0)
    return z

#ecm_3_013	(R0R1)C0
def aRRbC(w, R0, R1, C0):
    z = 1 / (1 / R0 + 1 / R1) + C(w, C0)
    return z

# ecm_3_014	R0(R1C0)
# DPFC: ECM-0 R(CR)
# DPFC: ECM-0 R0(C0R1)
# RaCRb == R0aC0R1b, Simplified Randles Cell
def RaRCb(w, R0, R1, C0):
    z = 1 / (1 / R1 + 1 / C(w, C0)) + R0
    return z

#ecm_3_015	(R0R1C0)
def aRRCb(w, R0,R1,C0):
    z = 1 / (1 / R0 + 1 / R1 + 1 / C(w, C0))
    return z

#ecm_3_016	R0C0C1
def RCC(w, R0, C0, C1):
    z = R0 + C(w, C0) + C(w, C1)
    return z

#ecm_3_017	(R0C0)C1
def aRCbC(w, R0, C0, C1):
    z = 1 / (1 / R0 + 1 / C(w, C0)) + 1 / C(w, C1)
    return z

#ecm_3_018	R0(C0C1)
def RaCCb(w, R0, C0, C1):
    z = R0 + 1 / (1 / C(w, C0) + 1 / C(w, C1))
    return z

#ecm_3_019	(R0C0C1)
def aRCCb(w, R0, C0, C1):
    z = 1 / (1 / R0 + 1 / C(w, C0) + 1 / C(w, C1))
    return z

#ecm_3_020	R0R1Q0
def RRQ(w, R0, R1, Q0_pair):
    z = R0 + R1 + Q(w, q = Q0_pair[0], n = Q0_pair[1])
    return z

#ecm_3_021	(R0R1)Q0
def aRRbQ(w, R0, R1, Q0_pair):
    z = 1 / (1 / R0 + 1 / R1) + Q(w, q = Q0_pair[0], n = Q0_pair[1])
    return z

#ecm_3_022	R0(R1Q0)
# DPFC: ECM-1 R(QR), already include ECM-0, when n = 1
def RaRQb(w, R0, R1, Q0_pair):
    z = R0 + 1 / (1 / R1 + 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]))
    return z

#ecm_3_023	(R0R1Q0)
def aRRQb(w, R0,R1,Q0_pair):
    z = 1 / (1 / R0 + 1 / R1 + 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]))
    return z

#ecm_3_024	RQ0Q1
def RQQ(w, R0, Q0_pair, Q1_pair):
    z = R0 + Q(w, q = Q0_pair[0], n = Q0_pair[1]) + Q(w, q = Q1_pair[0], n = Q1_pair[1])
    return z

#ecm_3_025	(R0Q0)Q1
def aRQbQ(w, R0, Q0_pair, Q1_pair):
    z = 1 / (1 / R0 + 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1])) + Q(w, q = Q1_pair[0], n = Q1_pair[1])
    return z

#ecm_3_026	R(Q0Q1)
def RaQQb(w, R0, Q0_pair, Q1_pair):
    z = R0 + 1 / (1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]))
    return z

# 3_027 C0C1Q0
def CCQ(w, C0, C1, Q0_pair):
    z = C(w, C0) + C(w, C1) + Q(w, q=Q0_pair[0],n=Q0_pair[1])
    return z

# 3_028 C0(C1Q0)
def CaCQb(w, C0, C1, Q0_pair):
    z = C(w, C0) + 1 / (1/C(w, C1) + 1/Q(w,q=Q0_pair[0],n=Q0_pair[1]))
    return z

# 3_029 (C0C1Q0)
def aCCQb(w, C0, C1, Q0_pair):
    z = 1 / (1/C(w, C0) + 1/C(w,C1)+ 1/Q(w,q=Q0_pair[0],n=Q0_pair[1]))
    return z

# 3_030 (C0C1)Q0
def aCCbQ(w, C0, C1, Q0_pair):
    z = 1 / (1/C(w, C0) + 1/C(w, C1)) + Q(w,q=Q0_pair[0],n=Q0_pair[1])
    return z

# 3_031 C0Q0Q1
def CQQ(w, C0, Q0_pair, Q1_pair):
    z = C(w, C0) + Q(w,q=Q0_pair[0],n=Q0_pair[1]) + Q(w,q=Q1_pair[0],n=Q1_pair[1])
    return z

# 3_032 C0(Q0Q1)
def CaQQb(w, C0, Q0_pair, Q1_pair):
    z = C(w, C0) + 1/(1/Q(w,q=Q0_pair[0],n=Q0_pair[1]) + 1/Q(w,q=Q1_pair[0],n=Q1_pair[1]))
    return z

# 3_033 Q0(C0Q1)
def QaCQb(w, C0, Q0_pair, Q1_pair):
    z = Q(w,q=Q0_pair[0],n=Q0_pair[1]) + 1/(1/C(w,C0) + 1/Q(w,q=Q1_pair[0],n=Q1_pair[1]))
    return z

# 3_034 (C0Q0Q1)
def aCQQb(w, C0, Q0_pair, Q1_pair):
    z = 1/(1/C(w,C0) + 1/Q(w,q=Q0_pair[0],n=Q0_pair[1]) + 1/Q(w,q=Q1_pair[0],n=Q1_pair[1]))
    return z

# 3_035 (Q0(R0Q1))
def aQaRQbb(w, Q0_pair, R0, Q1_pair):
    z = 1 / (1/Q(w,q=Q0_pair[0],n=Q0_pair[1]) + 1/(R0 + Q(w,q=Q1_pair[0],n=Q1_pair[1])))
    return z

# 3_036 (R(RO))
# 3_037
# 3_038
# ...

# 3_051 (C0(C1Q0))
def aC0aC1Q0bb(w, C0, C1, Q0_pair):
    z0 = C(w, C1) + Q(w, q=Q0_pair[0], n=Q0_pair[1])
    z = 1 / (1 / C(w, C0) + 1 / z0)
    return z

# 3_052
# 3_053
# 3_054

# 3_055 (C0(R0W0))
def aCaRWbb(w, C0, R0, W0):
    z = 1 / (1/C(w, C0) + 1/(R0 + WB(w, W0)))
    return z

# 3_056
# 3_057
# 3_058
# ---------------------------------------- ECM has 3 elements----------------------------------------


# ---------------------------------------- ECM has 4 elements----------------------------------------
# ecm_4_001	(C0(R0(C1R1)))
def aCaRaCRbbb(w, C0, R0, C1, R1):
    z = 1 / ( 1 / C(w,C0) + 1 /( R0 + 1 / (1/C(w,C1) + 1 / R1) ) )
    return z

# ecm_4_002	(C0(R0(L0R1)))
def aCaRaLRbbb(w, C0, R0, L0, R1):
    z = 1 / ( 1 / C(w,C0) + 1 /( R0 + 1 / (1 / L(w, L0) + 1 / R1 )))
    return z

# ecm_4_003	(C0R0(C1R1))
def aCRaCRbb(w,C0, R0, C1, R1):
    z = 1 / ( 1 / R0 + 1 / C(w,C0) + 1/ (C(w,C1) + R1 ))
    return z

# ecm_4_004	(C0R0(L0R1))
def aCRaLRbb(w, C0, R0, L0, R1):
    z = 1 / ( 1 / R0 + 1 / C(w,C0) + 1 / (L(w, L0) + R1))
    return z

# ecm_4_005	L0R0(C0R1)
def LRaCRb(w, C0, L0, R0, R1):
    z = L(w,L0) + R0 + 1 / ( C(w,C0) + 1 / R1)
    return z

# ecm_4_006	L0R0(Q0R1)
def LRaQRb(w, L0, R0, R1, Q0_pair):
    z = L0 + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1)
    return z

# ecm_4_007	R0(Q0(R1W0))
# DPFC: ECM-4 R(Q(RW)) --> R0(Q0(R1W0))
def RaQaRWbb(w, R0, Q0_pair, R1, W0):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 /( WB(w, sigma=W0) + R1 ) )
    return z

# ecm_4_008	R0(C0R1)W0
def RaCRbW(w, R0, C0, R1, W0):
    z = R0 + 1 / ( 1 / C(w,C0) + 1 / R1 ) + WB(w, sigma=W0)
    return z

# ecm_4_009 R0(C0(R1W0))
def RaCaRWbb(w, R0, C0, R1, W0):
    z = R0 + 1 / ( 1 / C(w,C0) + 1 / ( R1 + WB(w, sigma=W0) ) )
    return z

# ecm_4_010 R0(C0(R1T0)) ******** 待处理
def RaCaRTbb(w, R0, C0, R1, T0):
    z = R0 + 1 / ( 1 / C(w,C0) + 1 / ( R1 + T(w, T0) ) )
    return z

# ecm_4_011 R0(C0(R1O0)) ******** 待处理
def RaCaRObb(w, R0, C0, R1, O0):
    z = R0 + 1 / ( 1 / C(w,C0) + 1 / ( R1 + O(w, O0) ) )
    return z

# ecm_4_012 R0(C0R1)O0 ******** 待处理
def RaCRbO(w, R0, C0, R1, O0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + O(w, O0)
    return z

# ecm_4_013 R0(C0R1)T0 ******** 待处理
def RaCRbT(w, R0, C0, R1, T0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1) + T(w, T0)
    return z

# ecm_4_014 R0(Q0(R1O0)) ******** 待处理
def RaQaRObb(w, R0, Q0_pair, R1, O0):
    z = R0 + 1 / ( 1 / Q(w,q=Q0_pair[0],n=Q0_pair[1]) + 1 / ( R1 + O(w, O0) ) )
    return z

# 4_015 R0(Q0R1)W0
# DPFC: ECM-7 R(QR)W --> R0(Q0R1)W0
def RaQRbW(w, R0, Q0_pair, R1, W0):
    z = R0 + 1 / (1 / Q(w, q=Q0_pair[0], n=Q0_pair[1]) + 1/R1) + WB(w, sigma=W0)
    return z
# ---------------------------------------- ECM has 4 elements----------------------------------------

# ---------------------------------------- ECM has 5 elements----------------------------------------
# ecm_5_001	L0R0(C0(Q0R1))
def LRaCaQRbb(w, L0,R0,C0, Q0_pair, R1):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + Q(w, q = Q0_pair[0], n = Q0_pair[1]) ))
    return z

# ecm_5_002	L0R0(C0R1)Q0
def LRaCRbQ(w, L0, R0, C0, R1, Q0_pair):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + Q(w, q = Q0_pair[0], n = Q0_pair[1])
    return z

# ecm_5_003	L0R0Q0(C0R1)
def LRQaCRb(w, L0, R0, Q0_pair, C0, R1):
    z = L(w, L0) + R0 + Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / (1 / R1 + 1 / C(w, C0) )
    return z

# ecm_5_004	L0R0Q0(Q1R1)
def LRQaQRb(w, L0, R0, Q0_pair, Q1_pair,R1):
    z = L(w, L0) + R0 + Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( 1 / R1 + 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) )
    return z

# ecm_5_005	R0(C0(R1(C1R2)))
def RaCaRaCRbbb(w, R0, C0, R1, C1, R2):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / C(w, C1) + 1 / R2 )))
    return z

# ecm_5_006	R(CR(CR)) --> R0(C0R1(C1R2))
def RaCRaCRbb(w, R0, C0, R1, C1, R2):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( C(w, C1) + R2 ))
    return z

# ecm_5_007	R(CR(LR)) --> R0(C0R1(L0R2))
def RaCRaLRbb(w, R0, C0, R1, L0, R2):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( L(w, L0) + R2 ))
    return z

# ecm_5_008	R(CR(QR)) --> R0(C0R1(Q0R2))
def RaCRaQRbb(w, R0, C0, R1,  Q0_pair, R2):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( R2 + Q(w, q = Q0_pair[0], n = Q0_pair[1]) ))
    return z

# ecm_5_009	R(CR)(CR) --> R0(C0R1)(C1R2)
def RaCRbaCRb(w, R0, C0, R1, C1, R2):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + 1 / ( 1 / C(w, C1) + 1 / R2 )
    return z

# ecm_5_010	R(Q(R(CR))) --> R0(Q0(R1(C0R2)))
def RaQaRaCRbbb(w, R0, Q0_pair, R1, C0, R2):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / C(w, C0) + 1 / R2 )) )
    return z

# ecm_5_011	R(Q(R(QR))) --> R0(Q0(R1(Q1R2)))
# DPFC: ECM-9 R(Q(R(QR))) --> R0(Q0(R1(Q1R2)))
def RaQaRaQRbbb(w, R0, Q0_pair, R1,  Q1_pair, R2):
    z = R0 + 1 / (1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / (1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / R2 )))
    return z

# ecm_5_012	R(QR(LR)) --> R0(Q0R1(L0R2))
# DPFC: ECM-3 R(QR(LR)) --> R0(Q0R1(L0R2))
def RaQRaLRbb(w, R0, Q0_pair, R1, L0, R2):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( L(w, L0) + R2 ))
    return z

# ecm_5_013	R(QR(OR)) --> R0(Q0R1(O0R2))
def RaQRaORbb(w, R0, Q0_pair, R1, O0, R2):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( O(w, O0) + R2 ))
    return z

# ecm_5_014	R(QR(QR)) --> R0(Q0R1(Q1R2))
def RaQRaQRbb(w, R0, Q0_pair, R1, Q1_pair, R2):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( Q(w, q = Q1_pair[0], n = Q1_pair[1]) + R2 ))
    return z

# ecm_5_015	R(QR)(QR) --> R0(Q0R1)(Q1R2)
# DPFC: ECM-2 R(QR)(QR)
def RaQRbaQRb(w, R0, Q0_pair, R1, Q1_pair,R2):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 )	+ 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / R2 )
    return z

# 5_016 R(RC)(RC) --> R0(R1C0)(R2C1)
def RaRCbaRCb(w, R0, R1, C0, R2, C1):
    z = R0 + 1.0 / (1/R1 + 1/C(w, C0)) + 1.0/(1/R2 + 1/C(w, C1))
    return z

# 5_017
# DPFC: ECM-8 R(Q(RW))Q --> R0(Q0(R1W0))Q0
def RaQaRWbbQ(w, R0, Q0_pair, R1, W0, Q1_pair):
    z = R0 + 1 / ((1 / Q(w, q=Q0_pair[0],n=Q0_pair[1])) + (1 / (R1 + WB(w, sigma=W0)))) \
           + Q(w, q=Q1_pair[0], n=Q1_pair[1])
    return z

# 5_017
# 5_018
# 5_019

# 5_020 R0(R1C0)(R2W0)
def RaRCbaRWb(w, R0, R1, C0, R2, sigma):
    z = R0 + 1 / (1/R1 + 1/C(w, C0)) + 1 / (1/R2 + 1/WB(w, sigma))
    return z
# 5_021
# 5_022
# ...
# ---------------------------------------- ECM has 5 elements----------------------------------------

# ---------------------------------------- ECM has 6 elements----------------------------------------
# ecm_6_001	(C0R0(C1R1)(C2R2))
def aCRaCRbaCRbb(w, C0, R0, C1, R1, C2, R2):
    z = 1 / ( 1 / C(w,C0) + 1 / R0 + 1 / ( C(w,C1) + R1 ) + 1 / ( C(w,C2) + R2 ))
    return z

# ecm_6_002	(C0R0(L0R1)(L1R2))
def aCRaLRbaLRbb(w, C0, R0, L0, R1, L1, R2):
    z = 1 / ( 1 / C(w,C0) + 1 / R0 + 1 / ( L(w,L0) + R1 ) + 1 / ( L(w,L1) + R2 ))
    return z

#ecm_6_003	L0R0(C0(R1(C1R2)))
def LRaCaRaCRbbb(w, L0, R0, C0, R1, C1, R2):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / C(w, C1) + 1 / R2)))
    return z

#ecm_6_004	L0R0(C0R1(Q0R2))
def LRaCRaQRbb(w, L0, R0, C0, R1, Q0_pair, R2):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( Q(w, q = Q0_pair[0], n = Q0_pair[1]) + R2 ))
    return z

#ecm_6_005	L0R0(C0R1)(Q0R2)
def LRaCRbaQRb(w, L0, R0, C0, R1, Q0_pair, R2):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R2 )
    return z

#ecm_6_006 L0R0(Q0(R1(C0R2)))
def LRaQaRaCRbbb(w, L0, R0, Q0_pair, R1, C0, R2):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / C(w, C0) + 1 / R2 )))
    return z

#ecm_6_007	L0R0(Q0(R1(Q1R2)))
def LRaQaRaQRbbb(w, L0, R0, Q0_pair, R1, Q1_pair, R2):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / R2 )))
    return z

#ecm_6_008	L0R0(Q0R1(C0R2))
def LRaQRaCRbb(w, L0, R0, Q0_pair, R1, C0, R2):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( C(w, C0) + R2) )
    return z

#ecm_6_009 L0R0(Q0R1)(C0R2)
def LRaQRbaCRb(w,L0, R0, Q0_pair, R1, C0, R2):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 ) + 1 / (1 /  C(w, C0) + 1 / R2)
    return z

#ecm_6_010	L0R0(Q0R1)(Q1R2)
def LRaQRbaQRb(w,L0, R0, Q0_pair, R1, Q1_pair, R2):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 ) + 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / R2)
    return z

#ecm_6_011 R0(C0(R1(C1(R2O0))))
def RaCaRaCaRObbbb(w, R0, C0, R1, C1, R2, O0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / C(w, C1) + 1 / ( R2 + O(w, O0) ))))
    return z

#ecm_6_012 R0(C0(R1(C1(R2T0))))
def RaCaRaCaRTbbbb(w, R0, C0, R1, C1, R2, T0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / (1 / C(w, C1) + 1 / ( R2 + T(w, T0) ))))
    return z

#ecm_6_013 R0(C0(R1(C1(R2W0))))
def RaCaRaCaRWbbbb(w, R0, C0, R1, C1, R2, W0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / C(w, C1) + 1 / ( R2 + WB(w, sigma=W0) ))))
    return z

#ecm_6_014 R0(C0(R1(Q0(R2W0))))
def RaCaRaQaRWbbbb(w, R0, C0, R1, Q0_pair, R2, W0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / (1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + WB(w, sigma=W0)))))
    return z

# ecm_6_015 R0(C0R1)(C1R2)O0
def RaCRbaCRbO(w, R0, C0, R1, C1, R2, O0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + 1 / ( 1 / C(w, C1) + 1 / R2 ) + O(w, O0)
    return z

#ecm_6_016 R0(C0R1)(C1R2)S0
def RaCRbaCRbS(w, R0, C0, R1, C1, R2, S0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + 1 / ( 1 / C(w, C1) + 1 / R2 ) + S(w, S0)
    return z

#ecm_6_017 R0(C0R1)(C1R2)T0
def RaCRbaCRbT(w, R0, C0, R1, C1, R2, T0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + 1 / ( 1 / C(w, C1) + 1 / R2 ) + T0
    return z

# ecm_6_018 R0(C0R1)(C1R2)W0
def RaCRbaCRbW(w, R0, C0, R1, C1, R2, W0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + 1 / ( 1 / C(w, C1) + 1 / R2 ) + WB(w, W0)
    return z

# ecm_6_019 R0(C0R1Q0(R2W0))
def RaCRQaRWbb(w, R0, C0, R1, Q0_pair, R2, W0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + WB(w, sigma=W0) ))
    return z

# ecm_6_020 R0(Q0(R1(Q1(R2W0))))
def RaQaRaQaRWbbbb(w, R0,Q0_pair, R1, Q1_pair, R2, W0):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / (1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / ( R2 + WB(w, sigma=W0) ))))
    return z

# ecm_6_021 R0(Q0(R1W0))(C0R2)
def RaQaRWbbaCRb(w, R0, Q0_pair, R1, W0, C0, R2):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + WB(w, sigma=W0) )) + 1 / ( 1 / C(w, C0) + 1 / R2)
    return z

# ecm_6_022 R0(Q0(R1W0))(Q1R2)
def RaQaRWbbaQRb(w, R0, Q0_pair, R1, W0, Q1_pair, R2):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + WB(w, sigma=W0) )) + 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / R2)
    return z

# ecm_6_023 R0(Q0R1)(R2(O0R3))
def RaQRbaRaORbb(w, R0, Q0_pair, R1, R2, O0, R3):
    z = R0 + 1 / (1/Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1) + 1 / ( 1 / R2 + 1 / ( O(w, O0) + R3 ))
    return z

# 6_024
# DPFC: ECM-5 R(QR)(QR)W --> R0(Q0R1)(Q1R2)W0
def RaQRbaQRbW(w, R0, Q0_pair, R1, Q1_pair, R2, W0):
    z = R0 + 1 / (1 / Q(w, q=Q0_pair[0], n=Q0_pair[1]) + 1/R1) \
           + 1 / (1 / Q(w, q=Q1_pair[0], n=Q1_pair[1]) + 1/R2) \
           + WB(w, sigma=W0)
    return z

# 6_025
# DPFC: ECM-6 R(QR)(Q(RW)) --> R0(Q0R1)(Q1(R2W0))
def RaQRbaQaRWbb(w, R0, Q0_pair, R1, Q1_pair, R2, W0):
    z = R0 + 1 / (1 / Q(w, q=Q0_pair[0],n =Q0_pair[1]) + 1/R1) \
           + 1 / (1 / Q(w, q=Q1_pair[0],n=Q1_pair[1]) + (1/(R2 + WB(w, sigma=W0))))
    return z

# 6_026
# DPFC: ECM-10 R(CR)(C(RW)) --> R0(C0R1)(C1(R2W0))
def RaCRbaCaRWbb(w, R0, C0, R1, C1, R2, W0):
    z = R0 + 1/(1/R1 + 1j * w * C0) + 1 / (1j * w * C1 + 1/(R2 + WB(w, sigma=W0)))
    return z
# ---------------------------------------- ECM has 6 elements----------------------------------------

# ---------------------------------------- ECM has 7 elements----------------------------------------
# ecm_7_001 R0(C0(R1(C1(R2(C2R3)))))
def RaCaRaCaRaCRbbbbb(w, R0, C0, R1, C1, R2, C2, R3):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / C(w, C1) + 1 / ( R2 + 1 / ( 1 / C(w, C2) + 1 / R3 )))))
    return z

# ecm_7_002 R0(C0(R1(Q0(R2(C1R3)))))
def RaCaRaQaRaCRbbbbb(w, R0, C0, R1, Q0_pair, R2, C1, R3):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + 1 / ( 1 /  C(w, C1) + 1 / R3)))))
    return z

# ecm_7_003 R0(C0(R1(Q0(R2(Q1R3)))))
def RaCaRaQaRaQRbbbbb(w, R0, C0, R1, Q0_pair, R2, Q1_pair, R3):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + 1 / ( 1 /  Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / R3)))))
    return z

# ecm_7_004 R(C(R(QR)))(CR) --> R0(C0(R1(Q0R2)))(C1R3)
def RaCaRaQRbbbaCRb(w, R0, C0, R1, Q0_pair, R2, C1, R3):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R2 ))) + 1 / ( 1 / C(w, C1) + 1 / R3 )
    return z

# ecm_7_005 R0(C0R1(C1R2)(C2R3))
def RaCRaCRbaCRbb(w, R0, C0, R1, C1, R2, C2, R3):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( C(w, C1) + R2 ) + 1 / ( C(w, C2) + R3 ))
    return z

# ecm_7_006 R0(C0R1(Q0R2)(Q1R3))
def RaCRaQRbaQRbb(w, R0, C0, R1, Q0_pair, R2, Q1_pair, R3):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( Q(w, q = Q0_pair[0], n = Q0_pair[1]) + R2 ) + 1 / ( Q(w, q = Q1_pair[0], n = Q1_pair[1]) + R3 ))
    return z

# ecm_7_007 R0(Q0(R1(Q1(R2(C0R3)))))
def RaQaRaQaRaCRbbbbb(w, R0, R1, R2, C0, R3, Q0_pair, Q1_pair,):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / ( R2 + 1 / ( 1 / C(w, C0) + 1 / R3 )))))
    return z

# ecm_7_008 R0(C0R1(Q0R2))(C1R3)
def RaCRaQRbbaCRb(w, R0, C0, R1, Q0_pair, R2, C1, R3, ):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / (Q(w, q = Q0_pair[0], n = Q0_pair[1]) + R2 )) + 1 / ( 1 / C(w, C1) + 1 / R3 )
    return z

# ecm_7_009 R0(C0R1)(C1R2)(C2R3)
def RaCRbaCRbaCRb(w, R0, C0, R1, C1, R2, C2, R3):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + 1 / ( 1 / C(w, C1) + 1 / R2 ) + 1 / ( 1 / C(w, C2) + 1 / R3 )
    return z

# ecm_7_010 R0(C0R1)(Q0R2)(C1R3)
def RaCRbaQRbaCRb(w, R0, C0, R1, Q0_pair, R2, C1, R3 ):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 ) + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R2 ) + 1 / ( 1 / C(w, C1) + 1 / R3 )
    return z

# ecm_7_011 R(QR(CR)(OR)) --> R0(Q0R1(C0R2)(O0R3))
def RaQRaCRbaORbb(w,  R0, Q0_pair, R1, C0, R2, O0, R3):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( C(w, C0) + R2 ) + 1 / ( O(w, O0) + R3 ))
    return z

# ecm_7_012 R0(Q0R1(C0R2)(L0R3))
def RaQRaCRbaLRbb(w, R0, Q0_pair, R1, C0, R2, L0, R3):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( C(w, C0) + R2 ) + 1 / ( L(w, L0) + R3 ))
    return z

# ecm_7_013 R0(Q0R1)(Q1R2)(C0R3)
def RaQRbaQRbaCRb(w, R0, Q0_pair, R1, Q1_pair, R2, C0, R3):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 ) + 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / R2 ) + 1 / ( 1 / C(w, C0) + 1 / R3 )
    return z

#ecm_7_014 R0(Q0R1(L0R2)(L1R3))
def RaQRaLRbaLRbb(w,  R0, Q0_pair, R1, L0, R2, L1, R3):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( L(w, L0) + R2 ) + 1 / ( L(w, L1) + R3 ))
    return z

#ecm_7_015 R0(Q0R1(Q1R2)(C0R3))
def RaQRaQRbaCRbb(w, R0, Q0_pair, R1, Q1_pair, R2, C0, R3):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( Q(w, q = Q1_pair[0], n = Q1_pair[1]) + R2 ) + 1 / ( C(w, C0) + R3 ))
    return z

#ecm_7_016 R0(Q0R1(Q1R2)(O0R3))
def RaQRaQRbaORbb(w, R0, Q0_pair, R1, Q1_pair, R2, O0, R3):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( Q(w, q = Q1_pair[0], n = Q1_pair[1]) + R2 ) + 1 / ( O(w, O0) + R3 ))
    return z

#ecm_7_017 R0(Q0R1(C0R2)(C1R3))
def RaQRaCRbaCRbb(w, R0, Q0_pair, R1, C0, R2, C1, R3):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( C(w, C0) + R2 ) + 1 / ( C(w, C1) + R3))
    return z

# 7_018 R(CR(RW))(QR)
# DPFC: ECM-11 R0(C0R1(R2W0))(Q0R3) -->
def RaCRaRWbbaQRb(w, R0, C0, R1, R2, W0, Q0_pair, R3):
    z = R0 + 1 / (1j * w * C0 + 1/R1 + 1 / (R2 + WB(w, sigma=W0)) ) + 1 / (1 / Q(w, q=Q0_pair[0], n=Q0_pair[1]) + 1/R3)
    return z
# ---------------------------------------- ECM has 7 elements----------------------------------------

# ---------------------------------------- ECM has 8 elements----------------------------------------
#ecm_8_001 (C0R0(C1R1)(C2R2)(C3R3)) --> (C0R0(C1R1)(C2R2)(C3R3))
def aCRaCRbaCRbaCRbb(w,C0, R0, C1, R1, C2, R2, C3, R3):
    z = 1 / ( 1 / C(w, C0) + 1 / R0 + 1 / ( C(w, C1) + R1 ) + 1 / ( C(w, C2) + R2 ) + 1 / ( C(w, C3) + R3 ))
    return z

#ecm_8_002 (CR(LR)(LR)(LR)) --> (C0R0(L0R1)(L1R2)(L2R3))
def aCRaLRbaLRbaLRbb(w, C0, R0, L0, R1, L1, R2, L2, R3):
    z = 1 / ( 1 / C(w, C0) + 1 / R0 + 1 / ( L(w, L0) + R1 ) + 1 / ( L(w, L1) + R2 ) + 1 / ( L(w, L2) + R3 ))
    return z

#ecm_8_003 LR(C(R(QR)))(CR) --> L0R0(C0(R1(Q0R2)))(C1R3)
def LRaCaRaQRbbbaCRb(w, L0, R0, C0, R1, Q0_pair, R2, C1, R3):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / (1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R2))) + 1 / ( 1 / C(w, C1) + 1 / R3 )
    return z

#ecm_8_004 LR(CR(QR)(RW)) --> L0R0(C0R1(Q0R2)(R3W0))
def LRaCRaQRbaRWbb(w, L0, R0, C0, R1, Q0_pair, R2, R3, W0):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( Q(w, q = Q0_pair[0], n = Q0_pair[1]) + R2 ) + 1 / ( WB(w, sigma=W0) + R3 ))
    return z

#ecm_8_005 LR(CR(QR))(CR) --> L0R0(C0R1(Q0R2))(C1R3)
def LRaCRaQRbbaCRb(w,  L0, R0, C0, R1, Q0_pair, R2, C1, R3):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( Q(w, q = Q0_pair[0], n = Q0_pair[1]) + R2 )) + 1 / ( 1 / C(w, C1) + 1 / R3 )
    return z

#ecm_8_006 LR(Q(R(LR)(CR))) --> L0R0(Q0(R1(L1R2)(C0R3)))
def LRaQaRaLRbaCRbbb(w, L0, R0, Q0_pair, R1, L1, R2, C0, R3):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / L(w, L1) + 1 / R2 ) + 1 / ( 1 / C(w, C0) + 1 / R3 )))
    return z

#ecm_8_007 LR(QR(LR)(CR)) --> L0R0(Q0R1(L1R2)(C0R3))
def LRaQRaLRbaCRbb(w, L0, R0, Q0_pair, R1, L1, R2, C0, R3):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( L(w, L1) + R2 ) + 1 / ( C(w, C0) + R3))
    return z

#ecm_8_008 R0(C0(R1(Q0(R2(C1(R3W0))))))
def RaCaRaQaRaCaRWbbbbbb(w, R0, C0, R1, Q0_pair, R2, C1, R3, W0):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + 1 / ( 1 / C(w, C1) + 1 / ( R3 + WB(w, sigma=W0) ))))))
    return z

#ecm_8_009 R(C(R(Q(RW))))(CR) --> R0(C0(R1(Q0(R2W0))))(C1R3)
def RaCaRaQaRWbbbbaCRb(w, R0, C0, R1, Q0_pair, R2, W0, C1, R3):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + WB(w, sigma=W0) ) ))) + 1 / ( 1 / C(w, C1) + 1 / R3)
    return z

#ecm_8_010 LR(C(R(Q(R(CR))))) --> L0R0(C0(R1(Q0(R2(C1R3)))))
def LRaCaRaQaRaCRbbbbb(w,L0, R0, C0, R1, Q0_pair, R2, C1, R3):
    z = L(w, L0) + R0 + 1 / ( 1 /  C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + 1 / ( 1 / C(w, C1) + 1 / R3 )))))
    return z

#ecm_8_011 R(Q(R(QR)(Q(RW)))) --> R0(Q0(R1(Q1R2)(Q2(R3W0))))
def RaQaRaQRbaQaRWbbbb(w, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, W0):
    z = R0 + 1 / ( 1/Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / R2 ) + 1 / ( 1 / Q(w, q = Q2_pair[0], n = Q2_pair[1]) + 1 / ( R3 + WB(w, sigma=W0) ))) )
    return z
# ---------------------------------------- ECM has 8 elements----------------------------------------

# ---------------------------------------- ECM has 9 elements----------------------------------------
#ecm_9_001 R(QR(OR)(LR)(LR)) --> R0(Q0R1(O0R2)(L0R3)(L1R4))
def aQRaORbaLRbaLRbb(w, R0, Q0_pair, R1, O0, R2, L0, R3, L1, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( O(w, O0) +  R2 ) + 1 / ( L(w, L0) + R3 ) + 1 / ( L(w, L1) + R4 ))
    return z

#ecm_9_002 LR(C(R(Q(R(C(RW)))))) --> L0R0(C0(R1(Q0(R2(C1(R3W0))))))
def LRaCaRaQaRaCaRWbbbbbb(w, L0, R0, C0, R1, Q0_pair, R2, C1, R3, W0):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + 1 / ( 1 / C(w, C1) + 1 / ( R3 + WB(w, sigma=W0) ))))))
    return z

#ecm_9_003 R(C(R(Q(R(LR)(CR))))) --> R0(C0(R1(Q0(R2(L0R3)(C1R4)))))
def RaCaRaQaRaLRbaCRbbbbb(w, R0, C0, R1, Q0_pair, R2, L0, R3, C1, R4):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 +  1 / ( 1 / L(w, L0) + 1 / R3 ) + 1 / ( 1 / R4 + 1 / C(w, C1) )))))
    return z

#ecm_9_004 R(CR(LR)(LR)(OR)) --> R0(C0R1(L0R2)(L1R3)(O0R4))
def RaCRaLRbaLRbaORbb(w, R0, C0, R1, L0, R2, L1, R3, O0, R4):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( L(w, L0) + R2 ) + 1 / ( L(w, L1) + R3 ) + 1 / ( O(w, O0) + R4))
    return z

#ecm_9_005 R(CR(OR)(LR)(LR)) --> R0(C0R1(O0R2)(L0R3)(L1R4))
def RaCRaORbaLRbaLRbb(w, R0, C0, R1, O0, R2, L0, R3, L1, R4):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( O(w, O0) + R2 ) + 1 / ( L(w, L0) + R3 ) + 1 / ( L(w, L1) + R4 ))
    return z

#ecm_9_006 R(CR(QR)(LR)(CR)) --> R0(C0R1(Q0R2)(L0R3)(C1R4))
def RaCRaQRbaLRbaCRbb(w, R0, C0, R1, Q0_pair, R2, L0, R3, C1, R4):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( Q(w, q = Q0_pair[0], n = Q0_pair[1]) + R2 ) + 1 / ( L(w, L0) + R3 ) + 1 / ( C(w, C1) + R4 ))
    return z

#ecm_9_007 R(Q(R(C(R(LR)(CR))))) --> R0(Q0(R1(C0(R2(L0R3)(C1R4)))))
def RaQaRaCaRaLRbaCRbbbbb(w, R0, Q0_pair, R1, C0, R2, L0, R3, C1, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 /  C(w, C0) + 1 / ( R2 +  1 / ( 1 / L(w, L0) + 1 / R3)+ 1 / ( 1 / R4 + 1 / C(w, C1) )))))
    return z

#ecm_9_008 R(Q(R(Q(R(Q(R(CR))))))) --> R0(Q0(R1(Q1(R2(Q2(R3(C0R4)))))))
def RaQaRaQaRaQaRaCRbbbbbbb(w, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, C0, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / ( R2 + 1 / ( 1 / Q(w, q = Q2_pair[0], n = Q2_pair[1]) + 1 / ( R3 + 1 / ( 1 / C(w, C0) + 1 / R4 )))))))
    return z

#ecm_9_009 R(QR(CR)(LR)(OR)) ---> R0(Q0R1(C0R2)(L0R3)(O0R4))
def RaQRaCRbaLRbaORbb(w, R0, Q0_pair, R1, C0, R2, L0, R3, O0, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( C(w, C0) + R2 ) + 1 / ( L(w, L0) + R3 ) + 1 / ( O(w, O0) + R4 ))
    return z

#ecm_9_010 R(QR(CR)(RW))(CR) --> R0(Q0R1(C0R2)(R3W0))(C1R4)
def RaQRaCRbaRWbbaCRb(w, R0, Q0_pair, R1, C0, R2, R3, W0, C1, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) +  1 / R1 + 1 / ( C(w, C0) + R2 ) + 1 / ( R3 + WB(w, sigma=W0) )) + 1 / ( 1 / C(w, C1) + 1 / R4 )
    return z

#ecm_9_011 R(QR(LR)(LR)(OR)) --> R0(Q0R1(L0R2)(L1R3)(O0R4))
def RaQRaLRbaLRbaORbb(w, R0, Q0_pair, R1, L0, R2, L1, R3, O0, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( L(w, L0) + R2 ) + 1 / ( L(w, L1) + R3 ) + 1 / ( O(w, O0) + R4 ))
    return z

#ecm_9_012 R(QR(OR)(LR)(LR)) --> R0(Q0R1(O0R2)(L0R3)(L1R4))
def RaQRaORbaLRbaLRbb(w, R0, Q0_pair, R1, O0, R2, L0, R3, L1, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / (O(w, O0) + R2 )) + 1 / ( L(w, L0) + R3 )  + 1 / ( L(w, L1) + R4 )
    return z

#ecm_9_013 R(QR(QR)(LR)(OR)) --> R0(Q0R1(Q1R2)(L0R3)(O0R4))
def RaQRaQRbaLRbaORbb(w, R0, Q0_pair, R1, Q1_pair, R2, L0, R3, O0, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( Q(w, q = Q1_pair[0], n = Q1_pair[1]) + R2 ) + 1 / ( L(w, L0) + R3 ) + 1 / ( O(w, O0) + R4))
    return z

#ecm_9_014 R(QR(QR)(QR)(CR)) ---> R0(Q0R1(Q1R2)(Q2R3)(C0R4))
def RaQRaQRbaQRbaCRbb(w, R0, Q0_pair, R1, Q1_pair, R2, Q2_pair, R3, C0, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( Q(w, q = Q1_pair[0], n = Q1_pair[1]) + R2 ) + 1 / ( Q(w, q = Q2_pair[0], n = Q2_pair[1]) +  R3 ) + 1 / ( C(w, C0) + R4 ))
    return z
# ---------------------------------------- ECM has 9 elements----------------------------------------

# ---------------------------------------- ECM has 10 elements----------------------------------------
#ecm_10_001 (CR(CR)(CR)(CR)(CR)) --> (C0R0(C1R1)(C2R2)(C3R3)(C4R4))
def aCRaCRbaCRbaCRbaCRbb(w, C0, R0, C1, R1, C2, R2, C3, R3, C4, R4):
    z = 1 / ( 1 / C(w, C0) + 1 / R0 + 1 / ( C(w, C1) + R1 ) + 1 / ( C(w, C2) + R2 ) + 1 / ( C(w, C3) + R3 ) + 1 / ( C(w, C4) + R4 ))
    return z

#ecm_10_002 (CR(LR)(LR)(LR)(LR)) --> (C0R0(L0R1)(L1R2)(L2R3)(L3R4))
def aCRaLRbaLRbaLRbaLRbb(w, C0, R0, L0, R1, L1, R2, L2, R3, L3, R4):
    z = 1 / ( 1 / C(w, C0) + 1 / R0 + 1 / ( L(w, L0) + R1 ) + 1 / ( L(w, L1) + R2 ) + 1 / ( L(w, L2) + R3 ) + 1 / ( L(w, L3) + R4 ))
    return z

#ecm_10_003 (QR(OR)(CR)(LR)(QR)) --> (Q0R0(O0R1)(C0R2)(L0R3)(Q1R4))
def aQRaORbaCRbaLRbaQRbb(w, Q0_pair, R0, O0, R1, C0, R2, L0, R3, Q1_pair, R4):
    z = 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R0 + 1 / ( O(w, O0) + R1 ) + 1 / ( C(w, C0) + R2 ) + 1 / ( L(w, L0) + R3 ) + 1 / ( Q(w, q = Q1_pair[0], n = Q1_pair[1]) + R4 ))
    return z

#ecm_10_004 (QR(RO)(RC)(RL)(RQ)) ---> (Q0R0(R1O0)(R2C0)(R3L0)(R4Q1))
def aQRaRObaRCbaRLbaRQbb(w,Q0_pair, R0, R1, O0, R2, C0, R3, L0, R4, Q1_pair):
    z = 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R0 + 1 / ( O(w, O0) + R1) + 1 / ( C(w, C0) + R2 ) + 1 / ( R3 + L(w, L0) ) + 1 / ( R4 + Q(w, q = Q1_pair[0], n = Q1_pair[1]) ) )
    return z

#ecm_10_005 LR(C(R(Q(R(LR)(CR))))) --> L0R0(C0(R1(Q0(R2(L1R3)(C1R4)))))
def LRaCaRaQaRaLRbaCRbbbbb(w, L0, R0, C0, R1, Q0_pair, R2, L1, R3, C1, R4):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R2 + 1 / (1 / L(w, L1) + 1 / R3 ) + 1 / ( 1 / C(w, C1) + 1 / R4 )))) )
    return z

#ecm_10_006 LR(CR(QR)(LR)(CR)) ---> L0R0(C0R1(Q0R2)(L1R3)(C1R4)) ??
def LRaCRaQRbaLRbaCRbb(w, L0, R0, C0, R1, Q0_pair, R2, L1, R3, C1, R4):
    z = L(w, L0) + R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( Q(w, q = Q0_pair[0], n = Q0_pair[1]) + R2 ) + 1 / ( L(w, L1) + R3 )+ 1 / ( C(w, C1) + R4 ) )
    return z

#ecm_10_007 LR(Q(R(C(R(LR)(CR))))) ---> L0R0(Q0(R1(C0(R2(L1R3)(C1R4)))))
def LRaQaRaCaRaLRbaCRbbbbb(w, L0, R0, Q0_pair, R1, C0, R2, L1, R3, C1, R4):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / C(w, C0) + 1 / ( R2 + 1 / ( 1 /  L(w, L1) + 1 / R3 ) + 1 / ( 1 / C(w, C1) + 1 / R4 )))))
    return z

#ecm_10_008 LR(Q(R(Q(R(LR)(CR))))) --> L0R0(Q0(R1(Q1(R2(L1R3)(C0R4)))))
def LRaQaRaQaRaLRbaCRbbbbb(w, L0, R0, Q0_pair, R1, Q1_pair, R2, L1, R3, C0, R4):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / Q(w, q = Q1_pair[0], n = Q1_pair[1]) + 1 / ( R2 + 1 / ( 1 / L(w, L1) + 1 / R3 ) + 1 / ( 1 / C(w, C0) + 1 / R4 )))) )
    return z

# ecm_10_009 LR(QR(CR)(LR)(CR)) ---> L0R0(Q0R1(C0R2)(L1R3)(C1R4))
def LRaQRaCRbaLRbaCRbb(w, L0, R0, Q0_pair, R1, C0, R2, L1, R3, C1, R4):
    z = L(w, L0) + R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( C(w, C0) + R2 ) + 1 / ( L(w, L1) + R3 ) + 1 / ( C(w, C1) + R4 ))
    return z

# ecm_10_010 R(Q(R(C(R(C(RW))))))(CR) --> R0(Q0(R1(C0(R2(C1(R3W0))))))(C2R4)
def RaQaRaCaRaCaRWbbbbbbaCRb(w, R0, Q0_pair, R1, C0, R2, C1, R3, W0, C2, R4):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / ( R1 + 1 / ( 1 / C(w, C0) + 1 / ( R2 + 1 / ( 1 / C(w, C1) + 1 / ( R3 + WB(w, sigma=W0)) ))))) + 1 / ( 1 / C(w, C2) + 1 / R4 )
    return z
# ---------------------------------------- ECM has 10 elements----------------------------------------

# ---------------------------------------- ECM has 11 elements----------------------------------------
# ecm_11_001 R(QR(OR)(LR)(LR)(QR)) --> R0(Q0R1(O0R2)(L0R3)(L1R4)(Q1R5))
def RaQRaORbaLRbaLRbaQRbb(w, R0, R1, Q0_pair, R2, O0, R3, L0, R4, L1, Q1_pair, R5):
    z = R0 + 1 / ( 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]) + 1 / R1 + 1 / ( O(w, O0) + R2 ) + 1 / ( L(w, L0) + R3 ) + 1 / ( L(w, L1) + R4 ) + 1 / ( Q(w, q = Q1_pair[0], n = Q1_pair[1]) + R5 ))
    return z

#ecm_11_002 R(CR(LR)(LR)(OR)(QR)) --> R0(C0R1(L0R2)(L1R3)(O0R4)(Q0R5))
def RaCRaLRbaLRbaORbaQRbb(w, R0, C0, R1, L0, R2, L1, R3, O0, R4, Q0_pair, R5):
    z = R0 + 1 / ( 1 / C(w, C0) + 1 / R1 + 1 / ( L(w, L0) + R2 ) + 1 / ( L(w, L1) + R3 ) + 1 / ( O(w, O0) + R4 ) + 1 / ( Q(w, q = Q0_pair[0], n = Q0_pair[1]) + R5 ))
    return z
# ---------------------------------------- ECM has 11 elements----------------------------------------

def ecm_oldSeq_2_newSerial(ecm_num)-> str:
    ecm_serial = None
    if ecm_num == 1:
        # R(QR)
        ecm_serial = '3_022'
    elif ecm_num == 2:
        # ecm_5_015	R(QR)(QR) --> R0(Q0R1)(Q1R2)
        # DPFC: ECM-2 R(QR)(QR)
        ecm_serial = '5_015'
    elif ecm_num == 3:
        pass
    elif ecm_num == 4:
        # 4_015 R0(Q0R1)W0
        # DPFC: ECM-7 R(QR)W --> R0(Q0R1)W0
        ecm_serial = '4_015'
    elif ecm_num == 5:
        # ecm_5_012	R(QR(LR)) --> R0(Q0R1(L0R2))
        # DPFC: ECM-3 R(QR(LR)) --> R0(Q0R1(L0R2))
        ecm_serial = '5_012'
    elif ecm_num == 6:
        pass
    elif ecm_num == 7:
        pass
    elif ecm_num == 8:
        pass
    elif ecm_num == 9:
        # ecm_5_011	R(Q(R(QR))) --> R0(Q0(R1(Q1R2)))
        # DPFC: ECM-9 R(Q(R(QR))) --> R0(Q0(R1(Q1R2)))
        ecm_serial = '5_011'
    return ecm_serial