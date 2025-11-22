import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Butler-Volmer方程
def anode_equation(E, i0, alpha_a):
    F = 96485    
    R = 8.314    
    T = 298      
    eta = E - E_eq
    eta_clipped = np.clip(eta, -0.5, 0.5)  
    return i0 * np.exp(alpha_a * F * eta_clipped / (R * T))

def cathode_equation(E, i0, alpha_c):
    F = 96485
    R = 8.314
    T = 298
    eta = E - E_eq
    eta_clipped = np.clip(eta, -0.5, 0.5)
    return i0 * np.exp(alpha_c * F * np.abs(eta_clipped) / (R * T))

df = pd.read_excel(r"自己写路径，不要照抄")

E_eq = (df["平衡电位"].iloc[0])*0.001    

E_anode = df["阳极电位"].values*0.001      
i_anode = df["阳极电流"].values/(0.002*0.006)  

E_cathode = df["阴极电位"].values*0.001    
i_cathode_abs = np.abs((df["阴极电流"].values)/(0.0075*0.0075*3.141592654))

print("平衡电位 E_eq =", E_eq)
print("阳极电位数据范围：", np.min(E_anode), "~", np.max(E_anode))
print("阴极电位数据范围：", np.min(E_cathode), "~", np.max(E_cathode))

initial_guess_anode = [1e-6, 0.5]  # [i0, alpha_a]
bounds_anode = ([1e-10, 0.1], [1e-2, 0.9])

params_anode, _ = curve_fit(
    anode_equation,
    E_anode, i_anode,
    p0=initial_guess_anode,
    bounds=bounds_anode,
    method='trf',
    max_nfev=100000
)
i0_anode, alpha_a = params_anode

initial_guess_cathode = [1e-6, 0.5]  # [i0, alpha_c]
bounds_cathode = ([1e-12, 0.1], [1e-2, 0.9])

params_cathode, _ = curve_fit(
    cathode_equation,
    E_cathode, i_cathode_abs,
    p0=initial_guess_cathode,
    bounds=bounds_cathode,
    method='trf',
    max_nfev=100000
)
i0_cathode, alpha_c = params_cathode

print("阳极拟合结果：")
print(f"交换电流密度 i0 = {i0_anode:.18e} A/m²")
print(f"阳极传递系数 α_a = {alpha_a:.18f}\n")

print("阴极拟合结果：")
print(f"交换电流密度 i0 = {i0_cathode:.18e} A/m²")
print(f"阴极传递系数 α_c = {alpha_c:.18f}")

E_fit_anode = np.linspace(np.min(E_anode), np.max(E_anode), 100)
i_fit_anode = anode_equation(E_fit_anode, i0_anode, alpha_a)

E_fit_cathode = np.linspace(np.min(E_cathode), np.max(E_cathode), 100)
i_fit_cathode = -cathode_equation(E_fit_cathode, i0_cathode, alpha_c) 

plt.scatter(E_anode, np.log10(i_anode), label="Anode experimental data", color="red")
plt.plot(E_fit_anode, np.log10(i_fit_anode), label="Anode fitting curve", linestyle="--", color="darkred")

plt.scatter(E_cathode, -np.log10(i_cathode_abs), label="Cathode experiment data", color="blue")
plt.plot(E_fit_cathode, np.log10(i_fit_cathode), label="Cathode fitting curve", linestyle="--", color="navy")

plt.xlabel("E / V")
plt.ylabel(r"$\log |i|$ (A/m²)")
plt.legend()
plt.grid(True)
plt.show()
###以下为tafel拟合###


###以下为tafel拟合###
mask_anode = (i_anode > 1e-20) & ((E_anode - E_eq) > 0.03)
E_anode_high = E_anode[mask_anode]
i_anode_high = i_anode[mask_anode]

mask_cathode = (df["阴极电流"].values < -1e-20) & ((E_cathode - E_eq) < -0.03)
E_cathode_high = E_cathode[mask_cathode]
i_cathode_high_abs = i_cathode_abs[mask_cathode]

eta_anode_tafel = E_anode_high - E_eq
logi_anode = np.log10(i_anode_high)

eta_cathode_tafel = E_cathode_high - E_eq
logi_cathode = np.log10(i_cathode_high_abs)

if len(eta_anode_tafel) < 3:
    raise ValueError("阳极数据不足，请放宽阈值或检查数据！")
if len(eta_cathode_tafel) < 3:
    raise ValueError("阴极数据不足，请放宽阈值或检查数据！")

def tafel_anode_fit(logi, a, b):
    return a + b * logi

def tafel_cathode_fit(logi, a, b):
    return a + b * logi

params_anode, _ = curve_fit(
    tafel_anode_fit,
    logi_anode,
    eta_anode_tafel,
    p0=[0.1, 0.1],
    bounds=([-np.inf, -np.inf], [np.inf, np.inf])
)
a_anode, b_anode = params_anode

params_cathode, _ = curve_fit(
    tafel_cathode_fit,
    logi_cathode,
    eta_cathode_tafel,
    p0=[-0.1, -0.1],
    bounds=([-np.inf, -np.inf], [np.inf, np.inf])
)
a_cathode, b_cathode = params_cathode

print("阳极Tafel方程：η = a + b·lg(i)")
print(f"a_anode = {a_anode:.18f} V")
print(f"b_anode = {b_anode:.18f} V/dec\n")

print("阴极Tafel方程：η = a + b·lg(|i|)")
print(f"a_cathode = {a_cathode:.18f} V")
print(f"b_cathode = {b_cathode:.18f} V/dec")

plt.figure(figsize=(10, 6))

plt.scatter(logi_anode, eta_anode_tafel, label="Anode experimental data", color="red", marker="^")
plt.plot(logi_anode, a_anode + b_anode * logi_anode, 
         "--", color="darkred", label=f"Anode fitting curve: η = {a_anode:.2f} + {b_anode:.2f}·lg(i)")

plt.scatter(logi_cathode, eta_cathode_tafel, label="Cathode experiment data", color="blue", marker="v")
plt.plot(logi_cathode, a_cathode + b_cathode * logi_cathode, 
         "--", color="navy", label=f"Cathode fitting curve: η = {a_cathode:.2f} + {b_cathode:.2f}·lg(|i|)")

plt.xlabel(r"$\log |i|$ (A/m²)")
plt.ylabel(" η / V")
plt.title("η = a + b·lg(i)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()