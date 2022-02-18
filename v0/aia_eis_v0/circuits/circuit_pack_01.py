from circuits.elements import ele_C as C
from circuits.elements import ele_L as L
from circuits.elements import ele_Warburg as WB
from circuits.elements import ele_Q as Q

"""
支惠在原始的几个电路基础上新加了很多简单的单路，之后有空再去合并
"""

"""
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

# ECM-0 R(CR)
# ECM-0 R0(C0R1)
def RaCRb(w, R0, R1, C0):
    # RaCRb == R0aC0R1b, Simplified Randles Cell
    z = R0 + 1 / (1 / R1 + 1j * w * C0)
    return z

# ECM-1 R(QR), already include ECM-0, when n = 1
def RaQRb(w, R0, Q0_pair, R1):
    z = R0 + 1 / ((1 / R1) + (1 / Q(w, q = Q0_pair[0], n = Q0_pair[1])))
    return z

# ECM-2 R(QR)(QR)
def RaQRbaQRb(w, R0, Q0_pair, R1, Q1_pair, R2):
    z = R0 \
        + 1 / ((1 / R1) + (1 / Q(w, q = Q0_pair[0], n = Q0_pair[1])))\
        + 1 / ((1 / R2) + (1 / Q(w, q = Q1_pair[0], n = Q1_pair[1])))
    return z

# ECM-3 R(QR(LR))
def RaQRaLRbb(w, R0, Q0_pair, R1, L0, R2):
    z = R0 + 1 / ((1 / Q(w, q=Q0_pair[0], n=Q0_pair[1])) + (1 / R1) + (1 / (L(w, L0) + R2)))
    return z

# ECM-4 R(Q(RW))
def RaQaRWbb(w, R0, Q0_pair, R1, W0):
    z = R0 + 1 / ((1 / Q(w, q=Q0_pair[0], n=Q0_pair[1])) + (1 / (R1 + WB(w, sigma=W0))))
    return z

# ECM-5 R(QR)(QR)W
def RaQRbaQRbW(w, R0, Q0_pair, R1, Q1_pair, R2, W0):
    z = R0 + 1 / ((1 / Q(w, q=Q0_pair[0], n=Q0_pair[1])) + (1/R1)) \
           + 1 / ((1 / Q(w, q=Q1_pair[0], n=Q1_pair[1])) + (1/R2)) \
           + WB(w, sigma=W0)
    return z

# ECM-6 R(QR)(Q(RW))
def RaQRbaQaRWbb(w, R0, Q0_pair, R1, Q1_pair, R2, W0):
    z = R0 + 1 / ((1 / Q(w, q=Q0_pair[0],n =Q0_pair[1])) + (1/R1)) \
           + 1 / ((1 / Q(w, q=Q1_pair[0],n=Q1_pair[1])) + (1/(R2 + WB(w, sigma=W0))))
    return z

# ECM-7 R(QR)W
def RaQRbW(w, R0, Q0_pair, R1, W0):
    z = R0 + 1 / ((1 / Q(w, q=Q0_pair[0], n=Q0_pair[1])) + (1/R1)) + WB(w, sigma=W0)
    return z

# ECM-8 R(Q(RW))Q
def RaQaRWbbQ(w, R0, Q0_pair, R1, W0, Q1_pair):
    z = R0 + 1 / ((1 / Q(w, q=Q0_pair[0],n=Q0_pair[1])) + (1 / (R1 + WB(w, sigma=W0)))) \
           + Q(w, q=Q1_pair[0], n=Q1_pair[1])
    return z

# ECM-9 R(Q(R(QR)))
def RaQaRaQRbbb(w, R0, Q0_pair, R1, Q1_pair, R2):
    z = R0 + 1 / ((1 / Q(w, q=Q0_pair[0], n=Q0_pair[1])) + ( 1 / ( R1 + ( 1 / ( 1/Q(w, q=Q1_pair[0],n=Q1_pair[1]) + 1/R2)))) )
    return z

# ------------------ ECMs are not numbered ------------------
# DPFC: ECM-10 R0(C0R1)(C1(R2W0))
def RaCRbaCaRWbb(w, R0, C0, R1, C1, R2, W0):
    z = R0 + 1/(1/R1 + 1j * w * C0) + 1 / (1j * w * C1 + 1/(R2 + WB(w, sigma=W0)))
    return z

# DPFC: ECM-11 R0(C0R1(R2W0))(Q0R3)
def RaCRaRWbbaQRb(w, R0, C0, R1, R2, W0, Q0_pair, R3):
    z = R0 + 1 / (1j * w * C0 + 1/R1 + 1 / (R2 + WB(w, sigma=W0)) ) + 1 / (1 / Q(w, q=Q0_pair[0], n=Q0_pair[1]) + 1/R3)
    return z
# ------------------ ECMs are not numbered ------------------

#NEW RULE: ecm_2(two element)_001(Sequence)

#ecm_2_001  R0R1
def RR(R0,R1):
    z = R0 + R1
    return z

#ecm_2_002  (R0R1)
def aRRb(R0,R1):
    z = 1 / (1 / R0 + 1 / R1)
    return z

#ecm_2_003  R0L0
def RL(w,R0,L0):
    z = R0 + L(w, L0)
    return z

#ecm_2_004  (R0L0)
def aRLb(w,R0,L0):
    z = 1 / (1 / R0 + 1/(L(w, L0) ))
    return z

#ecm_2_005  R0C0
def RC(w, R0, C0):
    z = R0 + 1 / 1j * w * C0
    return z

#ecm_2_006  (R0C0)
def aRCb(w, R0, C0):
    z = 1 / (1 / R0 + 1j * w * C0)
    return z

#ecm_3_001  R0R1R2
def RRR(R0,R1,R2):
    z = R0 + R1 + R2
    return z

#ecm_3_002	R0(R1R2)
def RaRRb(R0,R1,R2):
    z = 1 / (1 / R1 + 1 / R2) + R0
    return z

#ecm_3_003	(R0R1R2)
def aRRRb(R0,R1,R2):
    z = 1 / (1 / R0 + 1 / R1 + 1 / R2)
    return z

#ecm_3_004	R0R1L0
def RRL(w, R0,R1,L0):
    z = R0 + R1 + L(w, L0)
    return z

#ecm_3_005	R0(R1L0)
def RaRLb(w,R0,R1,L0):
    z = R0 + 1 / (1 / L(w, L0) + 1 / R1)
    return z

#ecm_3_006	(R0R1)L0
def aRRbL(w,R0,R1,L0):
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
    z = R0 + R1 + 1 / 1j * w * C0
    return z

#ecm_3_013	(R0R1)C0
def aRRbC(w, R0,R1,C0):
    z = 1 / (1 / R0 + 1 / R1) + 1 / 1j * w * C0
    return z

#ecm_3_014	R0(R1C0)
def RaRCb(w, R0,R1,C0):
    z = 1 / (1 / R1 + 1j * w * C0) + R0
    return z

#ecm_3_015	(R0R1C0)
def aRRCb(w, R0,R1,C0):
    z = 1 / (1 / R0 + 1 / R1 + 1j * w * C0)
    return z

#ecm_3_016	R0C0C1
def RCC(w, R0,C0,C1):
    z = R0 + 1 / 1j * w * C1 + 1 / 1j * w * C0
    return z

#ecm_3_017	(R0C0)C1
def aRCbC(w, R0,C0,C1):
    z = 1 / (1 / R0 + 1j * w * C0) + 1 / 1j * w * C1
    return z

#ecm_3_018	R0(C0C1)
def RaCCb(w, R0,C0,C1):
    z = R0 + 1 / (1j * w * C0 + 1j * w * C1)
    return z

#ecm_3_019	(R0C0C1)
def aRCCb(w, R0,C0,C1):
    z =  1 / (1 / R0 + 1j * w * C0 + 1j * w * C1)
    return z

#ecm_3_020	R0R1Q0
def RRQ(w, R0,R1,Q0_pair):
    z = R0 + R1 + Q(w, q = Q0_pair[0], n = Q0_pair[1])
    return z

#ecm_3_021	(R0R1)Q0
def aRRbQ(w, R0, R1, Q0_pair):
    z = 1 / (1 / R0 + 1 / R1) + Q(w, q = Q0_pair[0], n = Q0_pair[1])
    return z

#ecm_3_022	R0(R1Q0)
def RaRQb(w, R0,R1,Q0_pair):
    z = R0 + 1 / (1 / R1 + 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]))
    return z

#ecm_3_023	(R0R1Q0)
def aRRQb(w, R0,R1,Q0_pair):
    z = 1 / (1 / R0 + 1 / R1 + 1 / Q(w, q = Q0_pair[0], n = Q0_pair[1]))
    return z

#ecm_3_024	RQ0Q1
def RQQ(w, R0,Q0_pair,Q1_pair):
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

#ecm_3_027	CCQ
#ecm_3_028	C(CQ)
#ecm_3_029	(CCQ)
#ecm_3_030	(CC)Q
#ecm_3_031	CQQ
#ecm_3_032	C(QQ)
#ecm_3_033	Q(CQ)
#ecm_3_034	(CQQ)
# ------------------ ECMs are not numbered ------------------