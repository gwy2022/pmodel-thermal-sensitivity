"""
Standalone script: Topt / Aopt (Anet-based) vs Growth Temperature,
for the "Full Acclimation" baseline plus each single-parameter-fixed
P-model acclimation experiment (Gamma*, xi, K, phi0, Eaj, Evcmax/Eac, c*).

Assembled from Figure4_v1.ipynb (cells 0,1,3,4,5,6,25,46,56,57,58,63,66,67,68,
and the configs_group1/configs_group2 definitions from cell 74), plus the
final plotting cell (49) in its already-fixed form:
  - gridspec imported locally (notebook never imported it at top level)
  - _pick_peak / _extract_points / _plot_one_pair defined locally (the
    notebook has several mutually-incompatible redefinitions of these
    same names scattered across cells)
  - _extract_points uses anet_values (net assimilation = GPP - 0.015*Vcmax),
    not gpp_values
  - equation-text annotations removed from the plots
  - panel-title fontsize set to 13 (was overflowing into the next panel at 15)
  - dead top legend row removed from the GridSpec
"""

# ============================================================
# 1. Imports and pyrealm setup
# ============================================================

from pyrealm.pmodel import (
    SubdailyScaler,
    memory_effect,
    SubdailyPModel,
    PModelEnvironment,
    PModel,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14, OptimalChiABC
from pyrealm.pmodel.functions import calc_ftemp_arrh, calc_ftemp_kphio, calc_modified_arrhenius_factor
from pyrealm.constants import PModelConst

pmodel_const = PModelConst(wang17_c=0)


# ============================================================
# 2. Common driving data (half-hourly forcing, acclimation window)
# ============================================================

# Number of half-hourly data points
n_pts = 17520  # 365 days * 48 half-hours

# Build a datetime index at 30-minute intervals
start_datetime = pd.to_datetime('2023-01-01')
datetimes = pd.date_range(start=start_datetime, periods=n_pts, freq='30T')

# Convert to a NumPy datetime64[m] array
datetimes_np = datetimes.to_numpy(dtype='datetime64[m]')

# Atmospheric pressure (Pa)
patm = np.full(n_pts, 101325)  # Standard atmospheric pressure

# VPD (vapor pressure deficit, Pa)
vpd = np.full(n_pts, 1000)  # Assume VPD = 500 Pa

# fAPAR (unitless)
fapar = np.full(n_pts, 1)  # Assume full canopy cover

# PPFD (µmol m⁻² s⁻¹)
ppfd = np.full(n_pts, 1200)  # PPFD fixed

# Atmospheric CO₂ concentration (ppm)
co2 = np.full(n_pts, 400)  # One value per half hour

# List of acclimation temperatures
acclim_temps = np.arange(10, 41, 5)

# Create SubdailyScaler and set the acclimation window
fsscaler = SubdailyScaler(datetimes=datetimes_np)
fsscaler.set_window(
    window_center=np.timedelta64(12, 'h'),   # 12:00 noon
    half_width=np.timedelta64(30, 'm'),      # ±0.5 hour
)

# Create a mask for 11:30–12:30 (acclimation window)
times = datetimes.time
start_time = pd.to_datetime('11:30').time()
end_time = pd.to_datetime('12:30').time()
acclim_mask = (times >= start_time) & (times <= end_time)


# ---- overrides used by every experiment below (from cell 6 of the notebook) ----
acclim_temps = [2.5, 5, 7.5, 10, 15, 20, 25, 30, 35]
co2 = np.full(n_pts, 420)
VP_CONSTANT = 0.5

def calc_vpd(t_celsius, vp_fixed):
    e_sat = 0.6108 * np.exp((17.27 * t_celsius) / (t_celsius + 237.3))
    return np.maximum(0, e_sat - vp_fixed) * 1000


# ============================================================
# 3. Custom pyrealm subclasses needed by the "Fixed xi" experiment
#    (quantum-yield and fixed-chi classes are defined for completeness /
#    consistency with the rest of the notebook, even though only
#    OptimalChiFixedXi is actually exercised by the experiments below)
# ============================================================

# 👇 必须先导入原始类所在模块，确保替换正确位置
from pyrealm.pmodel.quantum_yield  import QuantumYieldABC
import numpy as np
from warnings import warn

class QuantumYieldSandoval(
    QuantumYieldABC,
    method="sandoval",
    requires=["aridity_index", "mean_growth_temperature"],
    default_reference_kphio=1.0 / 9.0,   # ✅ 必须加上这个
    array_reference_kphio_ok=False
):
    """Overridden Sandoval-style quantum yield using Medlyn-style Arrhenius."""

    __experimental__: bool = True

    def peak_quantum_yield(self, aridity_index: np.ndarray) -> np.ndarray:
        phi0_theo = 1.0 / 9.0
        m = 4.090556
        n = 0.121122
        return phi0_theo / (1 + aridity_index**m)**n

    def _calculate_kphio(self) -> None:
        warn("QuantumYieldSandoval (overridden) is experimental.")
        
        ai = self.env.aridity_index
        mgdd = self.env.mean_growth_temperature
        tcleaf = self.env.tc
        tkleaf = self.env.tc + 273.15
        R = self.env.core_const.k_R

        dS0 = 3468.185
        dS_mgdd = 0.6680158
        Ha = 70885.39

        DeltaS = dS0 * mgdd**(-dS_mgdd)
        Hd = 295.0 * DeltaS

        Top = Hd / (DeltaS - R * np.log(Ha / (Hd - Ha)))

        def medlyn_peaked_arrhenius(tk, Top, Ha, Hd, dS):
            f1 = Hd * np.exp(Ha * (tk - Top) / (tk * R * Top))
            f2 = Hd - Ha * (1 - np.exp(Hd * (tk - Top) / (tk * R * Top)))
            return f1 / f2

        phi0_peak = self.peak_quantum_yield(ai)
        phi0_fT = medlyn_peaked_arrhenius(tkleaf, Top, Ha, Hd, DeltaS)

        self.kphio = phi0_peak * phi0_fT
QuantumYieldSandoval = QuantumYieldSandoval


from pyrealm.pmodel.optimal_chi import OptimalChiABC
import numpy as np

# 根据你的报错，这里使用 requires 参数
class OptimalChiFixed(OptimalChiABC, method="fixed", is_c4=False, requires=("fixed_chi",)):
    """
    自定义的固定 Chi 计算类。
    包含从 Chi 反推 Xi 的逻辑。
    """
    
    def set_beta(self) -> None:
        """设置默认 Beta 值 (仅为满足 API 需求)。"""
        self.beta = self.pmodel_const.beta_cost_ratio_prentice14

    def estimate_chi(self, xi_values=None) -> None:
        """
        使用固定的 Chi，并反向计算 Xi。
        """
        # 1. 获取固定 Chi 值
        if not hasattr(self.env, "fixed_chi"):
             raise ValueError("PModelEnvironment missing required 'fixed_chi' variable.")
             
        self.chi = getattr(self.env, "fixed_chi")

        # 2. 计算 Ci, mj, mc (这些步骤不变)
        self.ci = self.chi * self.env.ca
        self.mj = (self.ci - self.env.gammastar) / (self.ci + 2 * self.env.gammastar)
        self.mc = (self.ci - self.env.gammastar) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc
        
        # 3. 【新逻辑】反向计算 Xi
        #    公式: Xi = (Chi * sqrt(VPD)) / (1 - Chi)
        #    注意：如果 Chi 固定为 1.0，这里会发生除以零错误，需要注意
        
        vpd_sqrt = np.sqrt(self.env.vpd)
        
        # 为了防止除以0 (当 chi=1 时)，加一个极小的 epsilon 或者确保 chi < 1
        # 这里假设 chi 不会完全等于 1
        denominator = 1.0 - self.chi
        
        # 计算 Xi
        self.xi = (self.chi * vpd_sqrt) / denominator


class OptimalChiFixedXi(
    OptimalChiABC,
    method="fixed_xi",
    is_c4=False,
    requires=["xi_fixed"],   # 会自动检查 env.xi_fixed 是否存在
):
    def set_beta(self):
        self.beta = self.pmodel_const.beta_cost_ratio_prentice14

    def estimate_chi(self, xi_values=None):
        self.xi   = xi_values if xi_values is not None \
                    else np.full_like(self.env.ca, float(self.env.xi_fixed))
        self.chi  = (self.env.gammastar / self.env.ca
                     + (1.0 - self.env.gammastar / self.env.ca)
                     * self.xi / (self.xi + np.sqrt(self.env.vpd)))
        self.ci   = self.chi * self.env.ca
        self.mj   = (self.ci - self.env.gammastar) / (self.ci + 2 * self.env.gammastar)
        self.mc   = (self.ci - self.env.gammastar) / (self.ci + self.env.kmm)
        self.mjoc = self.mj / self.mc


# ============================================================
# 4.25 Full Acclimation (baseline) -> exp_low_high
# ============================================================

import copy

experiments_config = {
    'low_acclim_low_instant':  {'acclim_ppfd': 250,  'instant_ppfd': 167},
    'high_acclim_high_instant':{'acclim_ppfd': 1800, 'instant_ppfd': 1200},
    'low_acclim_high_instant': {'acclim_ppfd': 250,  'instant_ppfd': 600},
    'equal':                   {'acclim_ppfd': 250,  'instant_ppfd': 250},
}

all_results = {}

for exp_name, exp_config in experiments_config.items():
    acclim_ppfd  = exp_config['acclim_ppfd']
    instant_ppfd = exp_config['instant_ppfd']
    
    print(f"\n=== Running: {exp_name} | acclim={acclim_ppfd}, instant={instant_ppfd}, "
          f"ratio={acclim_ppfd/instant_ppfd:.3f} ===")

    results_by_light = {acclim_ppfd: []}

    for acclim_temp in acclim_temps:
        tc_1d     = np.linspace(0, 40, n_pts).copy()
        patm_use  = patm.copy()
        co2_use   = co2.copy()
        fapar_use = fapar.copy()

        ppfd_use  = np.full(n_pts, float(instant_ppfd))
        tc_1d[acclim_mask]    = acclim_temp
        vpd_use   = calc_vpd(tc_1d, VP_CONSTANT)
        vpd_use   = np.clip(vpd_use, 0, None)
        vpd_use[acclim_mask]  = 1000
        ppfd_use[acclim_mask] = acclim_ppfd

        co2_acclim  = fsscaler.get_daily_means(co2_use)
        vpd_acclim  = fsscaler.get_daily_means(vpd_use)
        patm_acclim = fsscaler.get_daily_means(patm_use)

        env = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env.aridity_index           = np.full(n_pts, 0.0)
        env.mean_growth_temperature = np.full(n_pts, 25)

        subdailyC3 = SubdailyPModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14',
            fapar=fapar_use,
            ppfd=ppfd_use,
            fs_scaler=fsscaler,
            alpha=1/15,
            allow_holdover=True,
        )

        acclim_env = PModelEnvironment(
            tc=acclim_temp,
            patm=patm_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim
        )
        acclim_env.aridity_index           = 0
        acclim_env.mean_growth_temperature = np.array([25])
        pmodel_acclim = PModel(
            env=acclim_env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=acclim_ppfd)
        if instant_ppfd == 600 and acclim_temp == 25:
            print(acclim_env.gammastar)
        R          = 8.3145
        ha_jmax25  = 43900
        ha_vcmax25 = 65330

        vcmax25_acclim = pmodel_acclim.vcmax * np.exp(
            (ha_vcmax25/R) * (1/(acclim_temp+273.15) - 1/298.15))
        jmax25_acclim  = pmodel_acclim.jmax  * np.exp(
            (ha_jmax25/R)  * (1/(acclim_temp+273.15) - 1/298.15))

        xi_real      = memory_effect(pmodel_acclim.optchi.xi, alpha=1/15)
        vcmax25_real = memory_effect(vcmax25_acclim, alpha=1/15, allow_holdover=True)
        jmax25_real  = memory_effect(jmax25_acclim,  alpha=1/15, allow_holdover=True)

        vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
        jmax25_subdaily  = fsscaler.fill_daily_to_subdaily(jmax25_real)

        vcmax_subdaily = vcmax25_subdaily * np.exp(
            (ha_vcmax25/R) * (1/298.15 - 1/(tc_1d+273.15)))
        jmax_subdaily  = jmax25_subdaily  * np.exp(
            (ha_jmax25/R)  * (1/298.15 - 1/(tc_1d+273.15)))

        xi_subdaily  = fsscaler.fill_daily_to_subdaily(xi_real)
        subdaily_chi = OptimalChiPrentice14(env=env)
        subdaily_chi.estimate_chi(xi_values=xi_subdaily)

        kmm       = env.kmm.flatten()
        gammastar = env.gammastar.flatten()
        fkphio_c3 = subdailyC3.kphio.kphio

        ac  = vcmax_subdaily * (subdaily_chi.ci - gammastar) / (subdaily_chi.ci + kmm)
        J   = 4 * fkphio_c3 * ppfd_use / np.sqrt(
                1 + (4 * fkphio_c3 * ppfd_use / jmax_subdaily)**2)
        aj  = (J / 4) * subdaily_chi.mj
        gpp = np.minimum(ac, aj)
        anet = gpp - 0.015 * vcmax_subdaily
        ci_subdaily = subdaily_chi.ci
        mask          = np.isfinite(gpp)
        instant_temps = tc_1d[mask]
        gpp_clean     = gpp[mask]
        anet_clean    = anet[mask]   # ← 新增
        ac_clean      = ac[mask]
        aj_clean      = aj[mask]
        jmax25_clean  = jmax25_subdaily[mask]
        jmax_clean    = jmax_subdaily[mask]
        vcmax_clean  = vcmax_subdaily[mask]
        vcmax25_clean = vcmax25_subdaily[mask]
        J_clean       = J[mask]
        ci_clean      = ci_subdaily[mask]

        idx_at_acclim = np.where(np.isclose(instant_temps, acclim_temp))[0]
        if len(idx_at_acclim) > 0:
            ac_at_growth_val = ac_clean[idx_at_acclim[0]]
            aj_at_growth_val = aj_clean[idx_at_acclim[0]]
            anet_at_growth_val = anet_clean[idx_at_acclim[0]]   # ← 新增
        else:
            ac_at_growth_val = np.nan
            aj_at_growth_val = np.nan

        sel          = instant_temps != acclim_temp
        final_temps  = instant_temps[sel]
        final_gpp    = gpp_clean[sel]
        final_anet   = anet_clean[sel]
        final_ac     = ac_clean[sel]
        final_aj     = aj_clean[sel]
        final_jmax25 = jmax25_clean[sel]
        final_jmax   = jmax_clean[sel]
        final_vcmax  = vcmax_clean[sel]
        final_vcmax25= vcmax25_clean[sel]
        final_J      = J_clean[sel]
        final_ci     = ci_clean[sel]

        if len(final_aj) > 0:
            idx_peak         = np.argmax(final_aj)
            topt_aj_val      = final_temps[idx_peak]
            jmax_at_topt_val = final_jmax[idx_peak]
            J_at_topt_val    = final_J[idx_peak]
            aj_peak_val      = final_aj[idx_peak]
        else:
            topt_aj_val = jmax_at_topt_val = J_at_topt_val = aj_peak_val = np.nan

        results_by_light[acclim_ppfd].append({
            'acclim_temp':   acclim_temp,
            'instant_ppfd':  instant_ppfd,
            'acclim_ppfd':   acclim_ppfd,
            'instant_temps': final_temps,
            'gpp_values':    final_gpp,
            'anet_values':   final_anet,
            'ac_values':     final_ac,
            'aj_values':     final_aj,
            'jmax25_values': final_jmax25,
            'jmax_values':   final_jmax,
            'vcmax_values':  final_vcmax,
            'vcmax25_values':final_vcmax25,
            'ci_values':       final_ci,
            'J_values':      final_J,
            'topt_aj':       topt_aj_val,
            'jmax_at_topt':  jmax_at_topt_val,
            'J_at_topt':     J_at_topt_val,
            'aj_peak':       aj_peak_val,
            'ac_at_growth':  ac_at_growth_val,
            'aj_at_growth':  aj_at_growth_val
        })

    all_results[exp_name] = copy.deepcopy(results_by_light)
    print(f"  完成！共 {len(results_by_light[acclim_ppfd])} 条记录")

# 方便访问
exp_low_low   = all_results['low_acclim_low_instant'][250]   # acclim=250,  instant=167
exp_high_high = all_results['high_acclim_high_instant'][1800] # acclim=1800, instant=1200
exp_low_high  = all_results['low_acclim_high_instant'][250]  # acclim=250,  instant=600
exp_equal     = all_results['equal'][250]                    # acclim=250,  instant=250

print("\n=== 所有实验完成 ===")
for name, config in experiments_config.items():
    print(f"{name:30s}: acclim={config['acclim_ppfd']:4d}, "
          f"instant={config['instant_ppfd']:4d}, "
          f"ratio={config['acclim_ppfd']/config['instant_ppfd']:.3f}")

# ============================================================
# 4.46 Fixed Gamma* -> exp_gamma_low_high
# ============================================================

import copy

experiments_config_gamma = {
    'low_acclim_low_instant':   {'acclim_ppfd': 250,  'instant_ppfd': 167},
    'high_acclim_high_instant': {'acclim_ppfd': 1800, 'instant_ppfd': 1200},
    'low_acclim_high_instant':  {'acclim_ppfd': 250,  'instant_ppfd': 600},
    'equal':                    {'acclim_ppfd': 250,  'instant_ppfd': 250},
}

GAMMA_FIXED = 4.332  # fixed gammastar for acclimation only

all_results_gamma = {}

for exp_name, exp_config in experiments_config_gamma.items():
    acclim_ppfd  = exp_config['acclim_ppfd']
    instant_ppfd = exp_config['instant_ppfd']

    print(f"\n=== Running gamma: {exp_name} | acclim={acclim_ppfd}, "
          f"instant={instant_ppfd}, ratio={acclim_ppfd/instant_ppfd:.3f} ===")

    results_by_light = {acclim_ppfd: []}

    for acclim_temp in acclim_temps:
        tc_1d     = np.linspace(0, 40, n_pts).copy()
        patm_use  = patm.copy()
        co2_use   = co2.copy()
        fapar_use = fapar.copy()

        ppfd_use  = np.full(n_pts, float(instant_ppfd))
        tc_1d[acclim_mask]    = acclim_temp
        vpd_use   = calc_vpd(tc_1d, VP_CONSTANT)
        vpd_use   = np.clip(vpd_use, 0, None)
        vpd_use[acclim_mask]  = 1000
        ppfd_use[acclim_mask] = acclim_ppfd

        co2_acclim  = fsscaler.get_daily_means(co2_use)
        vpd_acclim  = fsscaler.get_daily_means(vpd_use)
        patm_acclim = fsscaler.get_daily_means(patm_use)

        # ── 瞬时环境：gammastar完全温度依赖，不修改 ──
        env = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env.aridity_index           = np.full(n_pts, 0.0)
        env.mean_growth_temperature = np.full(n_pts, 25)

        # 取出原始温度依赖gammastar，用于ac瞬时计算
        gammastar_orig = env.gammastar.flatten().copy()

        # kphio来自瞬时环境（原始gammastar）
        pmodel_instant = PModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_instant.estimate_productivity(fapar=fapar_use, ppfd=ppfd_use)
        fkphio_c3 = pmodel_instant.kphio.kphio   # 温度依赖kphio ✓

        subdailyC3 = SubdailyPModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14',
            fapar=fapar_use,
            ppfd=ppfd_use,
            fs_scaler=fsscaler,
            alpha=1/15,
            allow_holdover=True,
        )

        # ── 驯化环境：gammastar固定4.33 → 影响vcmax25/jmax25/xi ──
        acclim_env = PModelEnvironment(
            tc=acclim_temp,
            patm=patm_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim
        )
        acclim_env.aridity_index           = 0
        acclim_env.mean_growth_temperature = np.array([25])
        acclim_env.gammastar = np.full_like(acclim_env.gammastar, GAMMA_FIXED)  # ← 固定 ✓

        pmodel_acclim = PModel(
            env=acclim_env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=acclim_ppfd)
        # vcmax, jmax, xi 都基于固定gammastar=4.33 ✓

        R          = 8.3145
        ha_jmax25  = 43900
        ha_vcmax25 = 65330

        vcmax25_acclim = pmodel_acclim.vcmax * np.exp(
            (ha_vcmax25/R) * (1/(acclim_temp+273.15) - 1/298.15))
        jmax25_acclim  = pmodel_acclim.jmax  * np.exp(
            (ha_jmax25/R)  * (1/(acclim_temp+273.15) - 1/298.15))

        # memory effect（基于固定gammastar驯化出的xi）
        xi_real      = memory_effect(pmodel_acclim.optchi.xi, alpha=1/15)
        vcmax25_real = memory_effect(vcmax25_acclim, alpha=1/15, allow_holdover=True)
        jmax25_real  = memory_effect(jmax25_acclim,  alpha=1/15, allow_holdover=True)

        vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
        jmax25_subdaily  = fsscaler.fill_daily_to_subdaily(jmax25_real)

        # 瞬时温度调整（温度依赖，不涉及gammastar）
        vcmax_subdaily = vcmax25_subdaily * np.exp(
            (ha_vcmax25/R) * (1/298.15 - 1/(tc_1d+273.15)))
        jmax_subdaily  = jmax25_subdaily  * np.exp(
            (ha_jmax25/R)  * (1/298.15 - 1/(tc_1d+273.15)))

        # ── chi：只有acclim_mask用固定gammastar，其余温度依赖 ──
        xi_subdaily     = fsscaler.fill_daily_to_subdaily(xi_real)
        gammastar_mixed = gammastar_orig.copy()
        gammastar_mixed[acclim_mask] = GAMMA_FIXED   # 只改acclim时间步 ✓

        env_chi = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env_chi.aridity_index           = np.full(n_pts, 0.0)
        env_chi.mean_growth_temperature = np.full(n_pts, 25)
        env_chi.gammastar               = gammastar_mixed

        subdaily_chi = OptimalChiPrentice14(env=env_chi)
        subdaily_chi.estimate_chi(xi_values=xi_subdaily)

        kmm = env.kmm.flatten()

        # ── ac：瞬时gammastar_orig（温度依赖）✓ ──
        ac  = vcmax_subdaily * (subdaily_chi.ci - gammastar_orig) / (subdaily_chi.ci + kmm)

        # ── J/Aj：不涉及gammastar ──
        J   = 4 * fkphio_c3 * ppfd_use / np.sqrt(
                1 + (4 * fkphio_c3 * ppfd_use / jmax_subdaily)**2)
        aj  = (J / 4) * subdaily_chi.mj

        gpp  = np.minimum(ac, aj)
        anet = gpp - 0.015 * vcmax_subdaily

        # ── mask ──
        mask          = np.isfinite(gpp)
        instant_temps = tc_1d[mask]
        gpp_clean     = gpp[mask]
        anet_clean    = anet[mask]
        ac_clean      = ac[mask]
        aj_clean      = aj[mask]
        jmax25_clean  = jmax25_subdaily[mask]
        jmax_clean    = jmax_subdaily[mask]
        vcmax25_clean = vcmax25_subdaily[mask]
        vcmax_clean   = vcmax_subdaily[mask]
        J_clean       = J[mask]

        # at growth temp
        idx_at_acclim = np.where(np.isclose(instant_temps, acclim_temp))[0]
        if len(idx_at_acclim) > 0:
            i_acc              = idx_at_acclim[0]
            ac_at_growth_val   = ac_clean[i_acc]
            aj_at_growth_val   = aj_clean[i_acc]
            anet_at_growth_val = anet_clean[i_acc]
        else:
            ac_at_growth_val   = np.nan
            aj_at_growth_val   = np.nan
            anet_at_growth_val = np.nan

        # remove acclim window
        sel          = instant_temps != acclim_temp
        final_temps  = instant_temps[sel]
        final_gpp    = gpp_clean[sel]
        final_anet   = anet_clean[sel]
        final_ac     = ac_clean[sel]
        final_aj     = aj_clean[sel]
        final_jmax25 = jmax25_clean[sel]
        final_jmax   = jmax_clean[sel]
        final_vcmax25 = vcmax25_clean[sel]
        final_vcmax   = vcmax_clean[sel]
        final_J      = J_clean[sel]

        # peak of Aj
        if len(final_aj) > 0:
            idx_peak         = np.argmax(final_aj)
            topt_aj_val      = final_temps[idx_peak]
            jmax_at_topt_val = final_jmax[idx_peak]
            J_at_topt_val    = final_J[idx_peak]
            aj_peak_val      = final_aj[idx_peak]
        else:
            topt_aj_val = jmax_at_topt_val = J_at_topt_val = aj_peak_val = np.nan

        results_by_light[acclim_ppfd].append({
            'acclim_temp':     acclim_temp,
            'instant_ppfd':    instant_ppfd,
            'acclim_ppfd':     acclim_ppfd,
            'instant_temps':   final_temps,
            'gpp_values':      final_gpp,
            'anet_values':     final_anet,
            'ac_values':       final_ac,
            'aj_values':       final_aj,
            'jmax25_values':   final_jmax25,
            'jmax_values':     final_jmax,
            'vcmax_values':    final_vcmax,
            'vcmax25_values':  final_vcmax25,
            'J_values':        final_J,
            'topt_aj':         topt_aj_val,
            'jmax_at_topt':    jmax_at_topt_val,
            'J_at_topt':       J_at_topt_val,
            'aj_peak':         aj_peak_val,
            'ac_at_growth':    ac_at_growth_val,
            'aj_at_growth':    aj_at_growth_val,
            'anet_at_growth':  anet_at_growth_val,
        })

    all_results_gamma[exp_name] = copy.deepcopy(results_by_light)
    print(f"  完成！共 {len(results_by_light[acclim_ppfd])} 条记录")

# ── 方便访问 ──
exp_gamma_low_low   = all_results_gamma['low_acclim_low_instant'][250]
exp_gamma_high_high = all_results_gamma['high_acclim_high_instant'][1800]
exp_gamma_low_high  = all_results_gamma['low_acclim_high_instant'][250]
exp_gamma_equal     = all_results_gamma['equal'][250]

print("\n=== Gamma 实验完成 ===")
for name, config in experiments_config_gamma.items():
    print(f"{name:30s}: acclim={config['acclim_ppfd']:4d}, "
          f"instant={config['instant_ppfd']:4d}, "
          f"ratio={config['instant_ppfd']/config['acclim_ppfd']:.3f}")

# ── 表格 ──


# ============================================================
# 4.57 Fixed xi -> exp_xi_low_high
# ============================================================

import copy
import matplotlib.pyplot as plt
import numpy as np

experiments_config_xi = {
    'low_acclim_low_instant':   {'acclim_ppfd': 250,  'instant_ppfd': 167},
    'high_acclim_high_instant': {'acclim_ppfd': 1800, 'instant_ppfd': 1200},
    'low_acclim_high_instant':  {'acclim_ppfd': 250,  'instant_ppfd': 600},
    'equal':                    {'acclim_ppfd': 250,  'instant_ppfd': 250},
}

XI_FIXED    = 82.82
CHECK_TEMP  = 20.0   # 只对这个温度做详细检验图
CHECK_EXP   = 'equal'  # 只对这个实验做检验图

all_results_xi  = {}
validation_log  = []   # 收集所有检验结果

for exp_name, exp_config in experiments_config_xi.items():
    acclim_ppfd  = exp_config['acclim_ppfd']
    instant_ppfd = exp_config['instant_ppfd']

    print(f"\n=== Running xi: {exp_name} | acclim={acclim_ppfd}, "
          f"instant={instant_ppfd}, ratio={acclim_ppfd/instant_ppfd:.3f} ===")

    results_by_light = {acclim_ppfd: []}

    for acclim_temp in acclim_temps:
        tc_1d     = np.linspace(0, 40, n_pts).copy()
        patm_use  = patm.copy()
        co2_use   = co2.copy()
        fapar_use = fapar.copy()

        ppfd_use  = np.full(n_pts, float(instant_ppfd))
        tc_1d[acclim_mask]    = acclim_temp
        vpd_use   = calc_vpd(tc_1d, VP_CONSTANT)
        vpd_use   = np.clip(vpd_use, 0, None)
        vpd_use[acclim_mask]  = 1000
        ppfd_use[acclim_mask] = acclim_ppfd

        co2_acclim  = fsscaler.get_daily_means(co2_use)
        vpd_acclim  = fsscaler.get_daily_means(vpd_use)
        patm_acclim = fsscaler.get_daily_means(patm_use)

        # ── 瞬时环境：完全温度依赖 ──
        env = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env.aridity_index           = np.full(n_pts, 0.0)
        env.mean_growth_temperature = np.full(n_pts, 25)

        gammastar_orig = env.gammastar.flatten().copy()
        kmm_orig       = env.kmm.flatten().copy()

        pmodel_instant = PModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_instant.estimate_productivity(fapar=fapar_use, ppfd=ppfd_use)
        fkphio_c3 = pmodel_instant.kphio.kphio

        subdailyC3 = SubdailyPModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14',
            fapar=fapar_use,
            ppfd=ppfd_use,
            fs_scaler=fsscaler,
            alpha=1/15,
            allow_holdover=True,
        )

        # ── 驯化环境：正常运行后覆盖xi ──
        acclim_env = PModelEnvironment(
            tc=acclim_temp,
            patm=patm_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim
        )
        acclim_env.aridity_index           = 0
        acclim_env.mean_growth_temperature = np.array([25])
        acclim_env.xi_fixed = XI_FIXED  # 直接把固定xi作为环境属性，方便后续检验
        pmodel_acclim = PModel(
            env=acclim_env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='fixed_xi'
        )
        xi_before_fix = float(np.mean(pmodel_acclim.optchi.xi))

        # ── 覆盖xi为固定值 ──
        pmodel_acclim.optchi.xi = np.full_like(pmodel_acclim.optchi.xi, XI_FIXED)
        xi_after_fix = float(np.mean(pmodel_acclim.optchi.xi))
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=acclim_ppfd)

        R          = 8.3145
        ha_jmax25  = 43900
        ha_vcmax25 = 65330
        optimal_chi = OptimalChiPrentice14(env=acclim_env)
        optimal_chi.estimate_chi(xi_values=pmodel_acclim.optchi.xi)
        ci_fix = optimal_chi.ci.flatten().copy()

        jmax25_acclim  = (4*pmodel_acclim.kphio.kphio*250)/np.sqrt(1/(1-(0.41*(ci_fix+2*acclim_env.gammastar)/(pmodel_acclim.optchi.ci-acclim_env.gammastar))**(2.0/3))-1) * np.exp((ha_jmax25/R) * (1/(acclim_temp+273.15) - 1/298.15))
        vcmax25_acclim = pmodel_acclim.vcmax * np.exp((ha_vcmax25/R) * (1/(acclim_temp+273.15) - 1/298.15))

        # memory effect：输入xi已是固定值 ✓
        xi_real      = memory_effect(pmodel_acclim.optchi.xi, alpha=1/15)
        vcmax25_real = memory_effect(vcmax25_acclim, alpha=1/15, allow_holdover=True)
        jmax25_real  = memory_effect(jmax25_acclim,  alpha=1/15, allow_holdover=True)

        vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
        jmax25_subdaily  = fsscaler.fill_daily_to_subdaily(jmax25_real)

        vcmax_subdaily = vcmax25_subdaily * np.exp(
            (ha_vcmax25/R) * (1/298.15 - 1/(tc_1d+273.15)))
        jmax_subdaily  = jmax25_subdaily  * np.exp(
            (ha_jmax25/R)  * (1/298.15 - 1/(tc_1d+273.15)))

        xi_subdaily = fsscaler.fill_daily_to_subdaily(xi_real)

        # env保持原始温度依赖（kmm/gammastar不变）
        subdaily_chi = OptimalChiPrentice14(env=env)
        subdaily_chi.estimate_chi(xi_values=xi_subdaily)
        ci_subdaily = subdaily_chi.ci.flatten().copy()

        # ── ac：瞬时温度依赖 ──
        ac  = vcmax_subdaily * (subdaily_chi.ci - gammastar_orig) / (subdaily_chi.ci + kmm_orig)
        J = 4 * fkphio_c3 * ppfd_use / np.sqrt(
            1 + (4 * fkphio_c3 * ppfd_use / jmax_subdaily) ** 2
        )
        aj  = (J / 4) * subdaily_chi.mj
        gpp  = np.minimum(ac, aj)
        anet = gpp - 0.015 * vcmax_subdaily

        # ── mask ──
        mask          = np.isfinite(gpp)
        instant_temps = tc_1d[mask]
        gpp_clean     = gpp[mask]
        anet_clean    = anet[mask]
        ac_clean      = ac[mask]
        aj_clean      = aj[mask]
        jmax25_clean  = jmax25_subdaily[mask]
        jmax_clean    = jmax_subdaily[mask]
        J_clean       = J[mask]
        vcmax_clean   = vcmax_subdaily[mask]
        vcmax25_clean  = vcmax25_subdaily[mask]
        ci_clean       = ci_subdaily[mask]

        idx_at_acclim = np.where(np.isclose(instant_temps, acclim_temp))[0]
        if len(idx_at_acclim) > 0:
            i_acc              = idx_at_acclim[0]
            ac_at_growth_val   = ac_clean[i_acc]
            aj_at_growth_val   = aj_clean[i_acc]
            anet_at_growth_val = anet_clean[i_acc]
        else:
            ac_at_growth_val   = np.nan
            aj_at_growth_val   = np.nan
            anet_at_growth_val = np.nan

        sel          = instant_temps != acclim_temp
        final_temps  = instant_temps[sel]
        final_gpp    = gpp_clean[sel]
        final_anet   = anet_clean[sel]
        final_ac     = ac_clean[sel]
        final_aj     = aj_clean[sel]
        final_jmax25 = jmax25_clean[sel]
        final_jmax   = jmax_clean[sel]
        final_J      = J_clean[sel]
        final_vcmax  = vcmax_clean[sel]
        final_vcmax25 = vcmax25_clean[sel]
        final_ci     = ci_clean[sel]

        if len(final_aj) > 0:
            idx_peak         = np.argmax(final_aj)
            topt_aj_val      = final_temps[idx_peak]
            jmax_at_topt_val = final_jmax[idx_peak]
            J_at_topt_val    = final_J[idx_peak]
            aj_peak_val      = final_aj[idx_peak]
        else:
            topt_aj_val = jmax_at_topt_val = J_at_topt_val = aj_peak_val = np.nan

        results_by_light[acclim_ppfd].append({
            'acclim_temp':     acclim_temp,
            'instant_ppfd':    instant_ppfd,
            'acclim_ppfd':     acclim_ppfd,
            'instant_temps':   final_temps,
            'gpp_values':      final_gpp,
            'anet_values':     final_anet,
            'ac_values':       final_ac,
            'aj_values':       final_aj,
            'jmax25_values':   final_jmax25,
            'jmax_values':     final_jmax,
            'vcmax_values':    final_vcmax,
            'vcmax25_values': final_vcmax25,
            'ci_values':       final_ci,
            'J_values':        final_J,
            'topt_aj':         topt_aj_val,
            'jmax_at_topt':    jmax_at_topt_val,
            'J_at_topt':       J_at_topt_val,
            'aj_peak':         aj_peak_val,
            'ac_at_growth':    ac_at_growth_val,
            'aj_at_growth':    aj_at_growth_val,
            'anet_at_growth':  anet_at_growth_val,
        })

    all_results_xi[exp_name] = copy.deepcopy(results_by_light)
    print(f"  完成！共 {len(results_by_light[acclim_ppfd])} 条记录")

# ── 方便访问 ──
exp_xi_low_low   = all_results_xi['low_acclim_low_instant'][250]
exp_xi_high_high = all_results_xi['high_acclim_high_instant'][1800]
exp_xi_low_high  = all_results_xi['low_acclim_high_instant'][250]
exp_xi_equal     = all_results_xi['equal'][250]


# ============================================================
# 4.58 Fixed K (kmm) -> exp_kmm_low_high
# ============================================================

import copy

experiments_config_kmm = {
    'low_acclim_low_instant':   {'acclim_ppfd': 250,  'instant_ppfd': 167},
    'high_acclim_high_instant': {'acclim_ppfd': 1800, 'instant_ppfd': 1200},
    'low_acclim_high_instant':  {'acclim_ppfd': 250,  'instant_ppfd': 600},
    'equal':                    {'acclim_ppfd': 250,  'instant_ppfd': 250},
}

KMM_FIXED = 70.84225158

all_results_kmm = {}

for exp_name, exp_config in experiments_config_kmm.items():
    acclim_ppfd  = exp_config['acclim_ppfd']
    instant_ppfd = exp_config['instant_ppfd']

    print(f"\n=== Running kmm: {exp_name} | acclim={acclim_ppfd}, "
          f"instant={instant_ppfd}, ratio={acclim_ppfd/instant_ppfd:.3f} ===")

    results_by_light = {acclim_ppfd: []}

    for acclim_temp in acclim_temps:
        tc_1d     = np.linspace(0, 40, n_pts).copy()
        patm_use  = patm.copy()
        co2_use   = co2.copy()
        fapar_use = fapar.copy()

        ppfd_use  = np.full(n_pts, float(instant_ppfd))
        tc_1d[acclim_mask]    = acclim_temp
        vpd_use   = calc_vpd(tc_1d, VP_CONSTANT)
        vpd_use   = np.clip(vpd_use, 0, None)
        vpd_use[acclim_mask]  = 1000
        ppfd_use[acclim_mask] = acclim_ppfd

        co2_acclim  = fsscaler.get_daily_means(co2_use)
        vpd_acclim  = fsscaler.get_daily_means(vpd_use)
        patm_acclim = fsscaler.get_daily_means(patm_use)

        # ── 瞬时环境：完全温度依赖 ──
        env = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env.aridity_index           = np.full(n_pts, 0.0)
        env.mean_growth_temperature = np.full(n_pts, 25)

        gammastar_orig = env.gammastar.flatten().copy()
        kmm_orig       = env.kmm.flatten().copy()

        pmodel_instant = PModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_instant.estimate_productivity(fapar=fapar_use, ppfd=ppfd_use)
        fkphio_c3 = pmodel_instant.kphio.kphio

        subdailyC3 = SubdailyPModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14',
            fapar=fapar_use,
            ppfd=ppfd_use,
            fs_scaler=fsscaler,
            alpha=1/15,
            allow_holdover=True,
        )

        # ── 驯化环境：kmm固定256 → 影响vcmax25/jmax25/xi ──
        acclim_env = PModelEnvironment(
            tc=acclim_temp,
            patm=patm_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim
        )
        acclim_env.aridity_index           = 0
        acclim_env.mean_growth_temperature = np.array([25])
        acclim_env.kmm = np.full_like(acclim_env.kmm, KMM_FIXED)  # ← 只固定acclim ✓

        pmodel_acclim = PModel(
            env=acclim_env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=acclim_ppfd)

        R          = 8.3145
        ha_jmax25  = 43900
        ha_vcmax25 = 65330

        vcmax25_acclim = pmodel_acclim.vcmax * np.exp(
            (ha_vcmax25/R) * (1/(acclim_temp+273.15) - 1/298.15))
        jmax25_acclim  = pmodel_acclim.jmax  * np.exp(
            (ha_jmax25/R)  * (1/(acclim_temp+273.15) - 1/298.15))

        xi_real      = memory_effect(pmodel_acclim.optchi.xi, alpha=1/15)
        vcmax25_real = memory_effect(vcmax25_acclim, alpha=1/15, allow_holdover=True)
        jmax25_real  = memory_effect(jmax25_acclim,  alpha=1/15, allow_holdover=True)

        vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
        jmax25_subdaily  = fsscaler.fill_daily_to_subdaily(jmax25_real)

        vcmax_subdaily = vcmax25_subdaily * np.exp(
            (ha_vcmax25/R) * (1/298.15 - 1/(tc_1d+273.15)))
        jmax_subdaily  = jmax25_subdaily  * np.exp(
            (ha_jmax25/R)  * (1/298.15 - 1/(tc_1d+273.15)))

        # ── chi：kmm不参与chi计算，直接用原始env ──
        xi_subdaily  = fsscaler.fill_daily_to_subdaily(xi_real)
        subdaily_chi = OptimalChiPrentice14(env=env)
        subdaily_chi.estimate_chi(xi_values=xi_subdaily)

        # ── ac：瞬时kmm_orig和gammastar_orig（温度依赖）✓ ──
        ac  = vcmax_subdaily * (subdaily_chi.ci - gammastar_orig) / (subdaily_chi.ci + kmm_orig)
        J   = 4 * fkphio_c3 * ppfd_use / np.sqrt(
                1 + (4 * fkphio_c3 * ppfd_use / jmax_subdaily)**2)
        aj  = (J / 4) * subdaily_chi.mj
        gpp  = np.minimum(ac, aj)
        anet = gpp - 0.015 * vcmax_subdaily

        # ── mask ──
        mask          = np.isfinite(gpp)
        instant_temps = tc_1d[mask]
        gpp_clean     = gpp[mask]
        anet_clean    = anet[mask]
        ac_clean      = ac[mask]
        aj_clean      = aj[mask]
        jmax25_clean  = jmax25_subdaily[mask]
        jmax_clean    = jmax_subdaily[mask]
        J_clean       = J[mask]
        vcmax_clean   = vcmax_subdaily[mask]

        idx_at_acclim = np.where(np.isclose(instant_temps, acclim_temp))[0]
        if len(idx_at_acclim) > 0:
            i_acc              = idx_at_acclim[0]
            ac_at_growth_val   = ac_clean[i_acc]
            aj_at_growth_val   = aj_clean[i_acc]
            anet_at_growth_val = anet_clean[i_acc]
        else:
            ac_at_growth_val   = np.nan
            aj_at_growth_val   = np.nan
            anet_at_growth_val = np.nan

        sel          = instant_temps != acclim_temp
        final_temps  = instant_temps[sel]
        final_gpp    = gpp_clean[sel]
        final_anet   = anet_clean[sel]
        final_ac     = ac_clean[sel]
        final_aj     = aj_clean[sel]
        final_jmax25 = jmax25_clean[sel]
        final_jmax   = jmax_clean[sel]
        final_J      = J_clean[sel]
        final_vcmax  = vcmax_clean[sel]

        if len(final_aj) > 0:
            idx_peak         = np.argmax(final_aj)
            topt_aj_val      = final_temps[idx_peak]
            jmax_at_topt_val = final_jmax[idx_peak]
            J_at_topt_val    = final_J[idx_peak]
            aj_peak_val      = final_aj[idx_peak]
        else:
            topt_aj_val = jmax_at_topt_val = J_at_topt_val = aj_peak_val = np.nan

        results_by_light[acclim_ppfd].append({
            'acclim_temp':     acclim_temp,
            'instant_ppfd':    instant_ppfd,
            'acclim_ppfd':     acclim_ppfd,
            'instant_temps':   final_temps,
            'gpp_values':      final_gpp,
            'anet_values':     final_anet,
            'ac_values':       final_ac,
            'aj_values':       final_aj,
            'jmax25_values':   final_jmax25,
            'jmax_values':     final_jmax,
            'vcmax_values':    final_vcmax,
            'J_values':        final_J,
            'topt_aj':         topt_aj_val,
            'jmax_at_topt':    jmax_at_topt_val,
            'J_at_topt':       J_at_topt_val,
            'aj_peak':         aj_peak_val,
            'ac_at_growth':    ac_at_growth_val,
            'aj_at_growth':    aj_at_growth_val,
            'anet_at_growth':  anet_at_growth_val,
        })

    all_results_kmm[exp_name] = copy.deepcopy(results_by_light)
    print(f"  完成！共 {len(results_by_light[acclim_ppfd])} 条记录")

# ── 方便访问 ──
exp_kmm_low_low   = all_results_kmm['low_acclim_low_instant'][250]
exp_kmm_high_high = all_results_kmm['high_acclim_high_instant'][1800]
exp_kmm_low_high  = all_results_kmm['low_acclim_high_instant'][250]
exp_kmm_equal     = all_results_kmm['equal'][250]

print("\n=== Kmm 实验完成 ===")
for name, config in experiments_config_kmm.items():
    print(f"{name:30s}: acclim={config['acclim_ppfd']:4d}, "
          f"instant={config['instant_ppfd']:4d}, "
          f"ratio={config['instant_ppfd']/config['acclim_ppfd']:.3f}")

# ── 结果表格 ──


# ============================================================
# 4.63 Fixed phi0 -> exp_phi_low_high
# ============================================================

import copy

experiments_config_phi = {
    'low_acclim_low_instant':   {'acclim_ppfd': 250,  'instant_ppfd': 167},
    'high_acclim_high_instant': {'acclim_ppfd': 1800, 'instant_ppfd': 1200},
    'low_acclim_high_instant':  {'acclim_ppfd': 250,  'instant_ppfd': 600},
    'equal':                    {'acclim_ppfd': 250,  'instant_ppfd': 250},
}

all_results_phi = {}

for exp_name, exp_config in experiments_config_phi.items():
    acclim_ppfd  = exp_config['acclim_ppfd']
    instant_ppfd = exp_config['instant_ppfd']

    print(f"\n=== Running phi: {exp_name} | acclim={acclim_ppfd}, "
          f"instant={instant_ppfd}, ratio={acclim_ppfd/instant_ppfd:.3f} ===")

    results_by_light = {acclim_ppfd: []}

    for acclim_temp in acclim_temps:
        tc_1d     = np.linspace(0, 40, n_pts).copy()
        patm_use  = patm.copy()
        co2_use   = co2.copy()
        fapar_use = fapar.copy()

        ppfd_use  = np.full(n_pts, float(instant_ppfd))
        tc_1d[acclim_mask]    = acclim_temp
        vpd_use   = calc_vpd(tc_1d, VP_CONSTANT)
        vpd_use   = np.clip(vpd_use, 0, None)
        vpd_use[acclim_mask]  = 1000
        ppfd_use[acclim_mask] = acclim_ppfd

        co2_acclim  = fsscaler.get_daily_means(co2_use)
        vpd_acclim  = fsscaler.get_daily_means(vpd_use)
        patm_acclim = fsscaler.get_daily_means(patm_use)

        # ── 瞬时环境：完全温度依赖，kphio用sandoval ──
        env = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env.aridity_index           = np.full(n_pts, 0.0)
        env.mean_growth_temperature = np.full(n_pts, 25)

        gammastar_orig = env.gammastar.flatten().copy()
        kmm_orig       = env.kmm.flatten().copy()

        # 瞬时kphio：sandoval（温度依赖）
        pmodel_instant = PModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_instant.estimate_productivity(fapar=fapar_use, ppfd=ppfd_use)
        fkphio_c3 = pmodel_instant.kphio.kphio   # 温度依赖 ✓

        subdailyC3 = SubdailyPModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14',
            fapar=fapar_use,
            ppfd=ppfd_use,
            fs_scaler=fsscaler,
            alpha=1/15,
            allow_holdover=True,
        )

        # ── 驯化环境：kphio固定（method_kphio='fixed'）──
        acclim_env = PModelEnvironment(
            tc=acclim_temp,
            patm=patm_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim
        )
        acclim_env.aridity_index           = 0
        acclim_env.mean_growth_temperature = np.array([25])

        pmodel_acclim = PModel(
            env=acclim_env,
            method_kphio='fixed',       # ← 固定kphio ✓
            reference_kphio=1/8,
            method_optchi='prentice14'
        )
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=acclim_ppfd)
        # vcmax, jmax, xi 都基于固定kphio=1/8 ✓

        R          = 8.3145
        ha_jmax25  = 43900
        ha_vcmax25 = 65330

        vcmax25_acclim = pmodel_acclim.vcmax * np.exp(
            (ha_vcmax25/R) * (1/(acclim_temp+273.15) - 1/298.15))
        jmax25_acclim  = pmodel_acclim.jmax  * np.exp(
            (ha_jmax25/R)  * (1/(acclim_temp+273.15) - 1/298.15))

        xi_real      = memory_effect(pmodel_acclim.optchi.xi, alpha=1/15)
        vcmax25_real = memory_effect(vcmax25_acclim, alpha=1/15, allow_holdover=True)
        jmax25_real  = memory_effect(jmax25_acclim,  alpha=1/15, allow_holdover=True)

        vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
        jmax25_subdaily  = fsscaler.fill_daily_to_subdaily(jmax25_real)

        vcmax_subdaily = vcmax25_subdaily * np.exp(
            (ha_vcmax25/R) * (1/298.15 - 1/(tc_1d+273.15)))
        jmax_subdaily  = jmax25_subdaily  * np.exp(
            (ha_jmax25/R)  * (1/298.15 - 1/(tc_1d+273.15)))

        # ── chi：kphio不参与chi计算，直接用原始env ──
        xi_subdaily  = fsscaler.fill_daily_to_subdaily(xi_real)
        subdaily_chi = OptimalChiPrentice14(env=env)
        subdaily_chi.estimate_chi(xi_values=xi_subdaily)
        ci_subdaily = subdaily_chi.ci.flatten().copy()
        # ── ac：瞬时kmm_orig和gammastar_orig ──
        ac  = vcmax_subdaily * (subdaily_chi.ci - gammastar_orig) / (subdaily_chi.ci + kmm_orig)

        # ── J：瞬时kphio（sandoval，温度依赖）✓ ──
        J   = 4 * fkphio_c3 * ppfd_use / np.sqrt(
                1 + (4 * fkphio_c3 * ppfd_use / jmax_subdaily)**2)
        aj  = (J / 4) * subdaily_chi.mj
        gpp  = np.minimum(ac, aj)
        anet = gpp - 0.015 * vcmax_subdaily

        # ── mask ──
        mask          = np.isfinite(gpp)
        instant_temps = tc_1d[mask]
        gpp_clean     = gpp[mask]
        anet_clean    = anet[mask]
        ac_clean      = ac[mask]
        aj_clean      = aj[mask]
        jmax25_clean  = jmax25_subdaily[mask]
        jmax_clean    = jmax_subdaily[mask]
        J_clean       = J[mask]
        vcmax_clean   = vcmax_subdaily[mask]
        vcmax25_clean = vcmax25_subdaily[mask]
        ci_clean      = ci_subdaily[mask]

        idx_at_acclim = np.where(np.isclose(instant_temps, acclim_temp))[0]
        if len(idx_at_acclim) > 0:
            i_acc              = idx_at_acclim[0]
            ac_at_growth_val   = ac_clean[i_acc]
            aj_at_growth_val   = aj_clean[i_acc]
            anet_at_growth_val = anet_clean[i_acc]
        else:
            ac_at_growth_val   = np.nan
            aj_at_growth_val   = np.nan
            anet_at_growth_val = np.nan

        sel          = instant_temps != acclim_temp
        final_temps  = instant_temps[sel]
        final_gpp    = gpp_clean[sel]
        final_anet   = anet_clean[sel]
        final_ac     = ac_clean[sel]
        final_aj     = aj_clean[sel]
        final_jmax25 = jmax25_clean[sel]
        final_vcmax25 = vcmax25_clean[sel]
        final_ci      = ci_clean[sel]
        final_jmax    = jmax_clean[sel]
        final_J       = J_clean[sel]
        final_vcmax   = vcmax_clean[sel]

        if len(final_aj) > 0:
            idx_peak         = np.argmax(final_aj)
            topt_aj_val      = final_temps[idx_peak]
            jmax_at_topt_val = final_jmax[idx_peak]
            J_at_topt_val    = final_J[idx_peak]
            aj_peak_val      = final_aj[idx_peak]
        else:
            topt_aj_val = jmax_at_topt_val = J_at_topt_val = aj_peak_val = np.nan

        results_by_light[acclim_ppfd].append({
            'acclim_temp':     acclim_temp,
            'instant_ppfd':    instant_ppfd,
            'acclim_ppfd':     acclim_ppfd,
            'instant_temps':   final_temps,
            'gpp_values':      final_gpp,
            'anet_values':     final_anet,
            'ac_values':       final_ac,
            'aj_values':       final_aj,
            'jmax25_values':   final_jmax25,
            'vcmax25_values':  final_vcmax25,
            'jmax_values':     final_jmax,
            'vcmax_values':    final_vcmax,
            'J_values':        final_J,
            'ci_values':      final_ci,
            'topt_aj':         topt_aj_val,
            'jmax_at_topt':    jmax_at_topt_val,
            'J_at_topt':       J_at_topt_val,
            'aj_peak':         aj_peak_val,
            'ac_at_growth':    ac_at_growth_val,
            'aj_at_growth':    aj_at_growth_val,
            'anet_at_growth':  anet_at_growth_val,
        })

    all_results_phi[exp_name] = copy.deepcopy(results_by_light)
    print(f"  完成！共 {len(results_by_light[acclim_ppfd])} 条记录")

# ── 方便访问 ──
exp_phi_low_low   = all_results_phi['low_acclim_low_instant'][250]
exp_phi_high_high = all_results_phi['high_acclim_high_instant'][1800]
exp_phi_low_high  = all_results_phi['low_acclim_high_instant'][250]
exp_phi_equal     = all_results_phi['equal'][250]

print("\n=== Phi 实验完成 ===")
for name, config in experiments_config_phi.items():
    print(f"{name:30s}: acclim={config['acclim_ppfd']:4d}, "
          f"instant={config['instant_ppfd']:4d}, "
          f"ratio={config['instant_ppfd']/config['acclim_ppfd']:.3f}")

# ── 结果表格 ──


# ============================================================
# 4.66 Fixed Evcmax/Eac -> exp_evcmax_low_high
# ============================================================

import copy

experiments_config_energy_vcmax = {
    'low_acclim_low_instant':   {'acclim_ppfd': 250,  'instant_ppfd': 167},
    'high_acclim_high_instant': {'acclim_ppfd': 1800, 'instant_ppfd': 1200},
    'low_acclim_high_instant':  {'acclim_ppfd': 250,  'instant_ppfd': 600},
    'equal':                    {'acclim_ppfd': 250,  'instant_ppfd': 250},
}

HA_VCMAX25_ACCLIM = 0       # ← 关键：vcmax acclimation无温度依赖
HA_VCMAX25_INSTANT = 65330  # 瞬时温度响应保持正常
HA_JMAX25 = 43900
R = 8.3145

all_results_energy_vcmax = {}

for exp_name, exp_config in experiments_config_energy_vcmax.items():
    acclim_ppfd  = exp_config['acclim_ppfd']
    instant_ppfd = exp_config['instant_ppfd']

    print(f"\n=== Running energy_vcmax: {exp_name} | acclim={acclim_ppfd}, "
          f"instant={instant_ppfd}, ratio={acclim_ppfd/instant_ppfd:.3f} ===")

    results_by_light = {acclim_ppfd: []}

    for acclim_temp in acclim_temps:
        tc_1d     = np.linspace(0, 40, n_pts).copy()
        patm_use  = patm.copy()
        co2_use   = co2.copy()
        fapar_use = fapar.copy()

        ppfd_use  = np.full(n_pts, float(instant_ppfd))
        tc_1d[acclim_mask]    = acclim_temp
        vpd_use   = calc_vpd(tc_1d, VP_CONSTANT)
        vpd_use   = np.clip(vpd_use, 0, None)
        vpd_use[acclim_mask]  = 1000
        ppfd_use[acclim_mask] = acclim_ppfd

        co2_acclim  = fsscaler.get_daily_means(co2_use)
        vpd_acclim  = fsscaler.get_daily_means(vpd_use)
        patm_acclim = fsscaler.get_daily_means(patm_use)

        # ── 瞬时环境：完全温度依赖 ──
        env = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env.aridity_index           = np.full(n_pts, 0.0)
        env.mean_growth_temperature = np.full(n_pts, 25)

        gammastar_orig = env.gammastar.flatten().copy()
        kmm_orig       = env.kmm.flatten().copy()

        pmodel_instant = PModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_instant.estimate_productivity(fapar=fapar_use, ppfd=ppfd_use)
        fkphio_c3 = pmodel_instant.kphio.kphio

        subdailyC3 = SubdailyPModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14',
            fapar=fapar_use,
            ppfd=ppfd_use,
            fs_scaler=fsscaler,
            alpha=1/15,
            allow_holdover=True,
        )

        # ── 驯化环境：正常 ──
        acclim_env = PModelEnvironment(
            tc=acclim_temp,
            patm=patm_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim
        )
        acclim_env.aridity_index           = 0
        acclim_env.mean_growth_temperature = np.array([25])

        pmodel_acclim = PModel(
            env=acclim_env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=acclim_ppfd)

        # ── 关键：vcmax25用ha=0（无温度依赖acclimation），jmax25正常 ──
        vcmax25_acclim = pmodel_acclim.vcmax * np.exp(
            (HA_VCMAX25_ACCLIM/R) * (1/(acclim_temp+273.15) - 1/298.15))  # ha=0 → exp(0)=1
        jmax25_acclim  = pmodel_acclim.jmax  * np.exp(
            (HA_JMAX25/R)         * (1/(acclim_temp+273.15) - 1/298.15))

        xi_real      = memory_effect(pmodel_acclim.optchi.xi, alpha=1/15)
        vcmax25_real = memory_effect(vcmax25_acclim, alpha=1/15, allow_holdover=True)
        jmax25_real  = memory_effect(jmax25_acclim,  alpha=1/15, allow_holdover=True)

        vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
        jmax25_subdaily  = fsscaler.fill_daily_to_subdaily(jmax25_real)

        # ── 瞬时温度响应：vcmax用正常ha_vcmax25 ──
        vcmax_subdaily = vcmax25_subdaily * np.exp(
            (HA_VCMAX25_INSTANT/R) * (1/298.15 - 1/(tc_1d+273.15)))
        jmax_subdaily  = jmax25_subdaily  * np.exp(
            (HA_JMAX25/R)          * (1/298.15 - 1/(tc_1d+273.15)))

        # ── chi：直接用原始env ──
        xi_subdaily  = fsscaler.fill_daily_to_subdaily(xi_real)
        subdaily_chi = OptimalChiPrentice14(env=env)
        subdaily_chi.estimate_chi(xi_values=xi_subdaily)

        # ── ac/J/aj/gpp ──
        ac  = vcmax_subdaily * (subdaily_chi.ci - gammastar_orig) / (subdaily_chi.ci + kmm_orig)
        J   = 4 * fkphio_c3 * ppfd_use / np.sqrt(
                1 + (4 * fkphio_c3 * ppfd_use / jmax_subdaily)**2)
        aj  = (J / 4) * subdaily_chi.mj
        gpp  = np.minimum(ac, aj)
        anet = gpp - 0.015 * vcmax_subdaily

        # ── mask ──
        mask          = np.isfinite(gpp)
        instant_temps = tc_1d[mask]
        gpp_clean     = gpp[mask]
        anet_clean    = anet[mask]
        ac_clean      = ac[mask]
        aj_clean      = aj[mask]
        jmax25_clean  = jmax25_subdaily[mask]
        jmax_clean    = jmax_subdaily[mask]
        J_clean       = J[mask]
        vcmax_clean   = vcmax_subdaily[mask]
        vcmax25_clean = vcmax25_subdaily[mask]

        idx_at_acclim = np.where(np.isclose(instant_temps, acclim_temp))[0]
        if len(idx_at_acclim) > 0:
            i_acc              = idx_at_acclim[0]
            ac_at_growth_val   = ac_clean[i_acc]
            aj_at_growth_val   = aj_clean[i_acc]
            anet_at_growth_val = anet_clean[i_acc]
        else:
            ac_at_growth_val   = np.nan
            aj_at_growth_val   = np.nan
            anet_at_growth_val = np.nan

        sel          = instant_temps != acclim_temp
        final_temps  = instant_temps[sel]
        final_gpp    = gpp_clean[sel]
        final_anet   = anet_clean[sel]
        final_ac     = ac_clean[sel]
        final_aj     = aj_clean[sel]
        final_jmax25 = jmax25_clean[sel]
        final_jmax   = jmax_clean[sel]
        final_J      = J_clean[sel]
        final_vcmax25 = vcmax25_clean[sel]
        final_vcmax  = vcmax_clean[sel]

        if len(final_aj) > 0:
            idx_peak         = np.argmax(final_aj)
            topt_aj_val      = final_temps[idx_peak]
            jmax_at_topt_val = final_jmax[idx_peak]
            J_at_topt_val    = final_J[idx_peak]
            aj_peak_val      = final_aj[idx_peak]
        else:
            topt_aj_val = jmax_at_topt_val = J_at_topt_val = aj_peak_val = np.nan

        results_by_light[acclim_ppfd].append({
            'acclim_temp':     acclim_temp,
            'instant_ppfd':    instant_ppfd,
            'acclim_ppfd':     acclim_ppfd,
            'instant_temps':   final_temps,
            'gpp_values':      final_gpp,
            'anet_values':     final_anet,
            'ac_values':       final_ac,
            'aj_values':       final_aj,
            'jmax25_values':   final_jmax25,
            'jmax_values':     final_jmax,
            'vcmax_values':    final_vcmax,
            'J_values':        final_J,
            'topt_aj':         topt_aj_val,
            'jmax_at_topt':    jmax_at_topt_val,
            'J_at_topt':       J_at_topt_val,
            'aj_peak':         aj_peak_val,
            'ac_at_growth':    ac_at_growth_val,
            'aj_at_growth':    aj_at_growth_val,
            'anet_at_growth':  anet_at_growth_val,
        })

    all_results_energy_vcmax[exp_name] = copy.deepcopy(results_by_light)
    print(f"  完成！共 {len(results_by_light[acclim_ppfd])} 条记录")

# ── 方便访问 ──
exp_evcmax_low_low   = all_results_energy_vcmax['low_acclim_low_instant'][250]
exp_evcmax_high_high = all_results_energy_vcmax['high_acclim_high_instant'][1800]
exp_evcmax_low_high  = all_results_energy_vcmax['low_acclim_high_instant'][250]
exp_evcmax_equal     = all_results_energy_vcmax['equal'][250]

print("\n=== Energy Vcmax 实验完成 ===")
for name, config in experiments_config_energy_vcmax.items():
    print(f"{name:30s}: acclim={config['acclim_ppfd']:4d}, "
          f"instant={config['instant_ppfd']:4d}, "
          f"ratio={config['instant_ppfd']/config['acclim_ppfd']:.3f}")

# ── 结果表格 ──


# ============================================================
# 4.67 Fixed Ejmax/Eaj -> exp_ejmax_low_high
# ============================================================

import copy

experiments_config_energy_jmax = {
    'low_acclim_low_instant':   {'acclim_ppfd': 250,  'instant_ppfd': 167},
    'high_acclim_high_instant': {'acclim_ppfd': 1800, 'instant_ppfd': 1200},
    'low_acclim_high_instant':  {'acclim_ppfd': 250,  'instant_ppfd': 600},
    'equal':                    {'acclim_ppfd': 250,  'instant_ppfd': 250},
}

HA_JMAX25_ACCLIM  = 0      # ← 关键：jmax acclimation无温度依赖
HA_JMAX25_INSTANT = 43900  # 瞬时温度响应保持正常
HA_VCMAX25        = 65330
R                 = 8.3145

all_results_energy_jmax = {}

for exp_name, exp_config in experiments_config_energy_jmax.items():
    acclim_ppfd  = exp_config['acclim_ppfd']
    instant_ppfd = exp_config['instant_ppfd']

    print(f"\n=== Running energy_jmax: {exp_name} | acclim={acclim_ppfd}, "
          f"instant={instant_ppfd}, ratio={acclim_ppfd/instant_ppfd:.3f} ===")

    results_by_light = {acclim_ppfd: []}

    for acclim_temp in acclim_temps:
        tc_1d     = np.linspace(0, 40, n_pts).copy()
        patm_use  = patm.copy()
        co2_use   = co2.copy()
        fapar_use = fapar.copy()

        ppfd_use  = np.full(n_pts, float(instant_ppfd))
        tc_1d[acclim_mask]    = acclim_temp
        vpd_use   = calc_vpd(tc_1d, VP_CONSTANT)
        vpd_use   = np.clip(vpd_use, 0, None)
        vpd_use[acclim_mask]  = 1000
        ppfd_use[acclim_mask] = acclim_ppfd

        co2_acclim  = fsscaler.get_daily_means(co2_use)
        vpd_acclim  = fsscaler.get_daily_means(vpd_use)
        patm_acclim = fsscaler.get_daily_means(patm_use)

        # ── 瞬时环境：完全温度依赖 ──
        env = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env.aridity_index           = np.full(n_pts, 0.0)
        env.mean_growth_temperature = np.full(n_pts, 25)

        gammastar_orig = env.gammastar.flatten().copy()
        kmm_orig       = env.kmm.flatten().copy()

        pmodel_instant = PModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_instant.estimate_productivity(fapar=fapar_use, ppfd=ppfd_use)
        fkphio_c3 = pmodel_instant.kphio.kphio

        subdailyC3 = SubdailyPModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14',
            fapar=fapar_use,
            ppfd=ppfd_use,
            fs_scaler=fsscaler,
            alpha=1/15,
            allow_holdover=True,
        )

        # ── 驯化环境：正常 ──
        acclim_env = PModelEnvironment(
            tc=acclim_temp,
            patm=patm_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim
        )
        acclim_env.aridity_index           = 0
        acclim_env.mean_growth_temperature = np.array([25])

        pmodel_acclim = PModel(
            env=acclim_env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=acclim_ppfd)

        # ── 关键：jmax25用ha=0，vcmax25正常 ──
        vcmax25_acclim = pmodel_acclim.vcmax * np.exp(
            (HA_VCMAX25/R) * (1/(acclim_temp+273.15) - 1/298.15))
        jmax25_acclim  = pmodel_acclim.jmax  * np.exp(
            (HA_JMAX25_ACCLIM/R) * (1/(acclim_temp+273.15) - 1/298.15))  # ha=0 → exp(0)=1 ✓

        xi_real      = memory_effect(pmodel_acclim.optchi.xi, alpha=1/15)
        vcmax25_real = memory_effect(vcmax25_acclim, alpha=1/15, allow_holdover=True)
        jmax25_real  = memory_effect(jmax25_acclim,  alpha=1/15, allow_holdover=True)

        vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
        jmax25_subdaily  = fsscaler.fill_daily_to_subdaily(jmax25_real)

        # ── 瞬时温度响应：jmax用正常ha_jmax25 ──
        vcmax_subdaily = vcmax25_subdaily * np.exp(
            (HA_VCMAX25/R)        * (1/298.15 - 1/(tc_1d+273.15)))
        jmax_subdaily  = jmax25_subdaily  * np.exp(
            (HA_JMAX25_INSTANT/R) * (1/298.15 - 1/(tc_1d+273.15)))  # 瞬时响应正常 ✓

        # ── chi：直接用原始env ──
        xi_subdaily  = fsscaler.fill_daily_to_subdaily(xi_real)
        subdaily_chi = OptimalChiPrentice14(env=env)
        subdaily_chi.estimate_chi(xi_values=xi_subdaily)

        ac  = vcmax_subdaily * (subdaily_chi.ci - gammastar_orig) / (subdaily_chi.ci + kmm_orig)
        J   = 4 * fkphio_c3 * ppfd_use / np.sqrt(
                1 + (4 * fkphio_c3 * ppfd_use / jmax_subdaily)**2)
        aj  = (J / 4) * subdaily_chi.mj
        gpp  = np.minimum(ac, aj)
        anet = gpp - 0.015 * vcmax_subdaily
        # ── mask ──
        mask          = np.isfinite(gpp)
        instant_temps = tc_1d[mask]
        gpp_clean     = gpp[mask]
        anet_clean    = anet[mask]
        ac_clean      = ac[mask]
        aj_clean      = aj[mask]
        jmax25_clean  = jmax25_subdaily[mask]
        jmax_clean    = jmax_subdaily[mask]
        J_clean       = J[mask]
        vcmax25_clean = vcmax25_subdaily[mask]
        vcmax_clean   = vcmax_subdaily[mask]

        idx_at_acclim = np.where(np.isclose(instant_temps, acclim_temp))[0]
        if len(idx_at_acclim) > 0:
            i_acc              = idx_at_acclim[0]
            ac_at_growth_val   = ac_clean[i_acc]
            aj_at_growth_val   = aj_clean[i_acc]
            anet_at_growth_val = anet_clean[i_acc]
        else:
            ac_at_growth_val   = np.nan
            aj_at_growth_val   = np.nan
            anet_at_growth_val = np.nan

        sel          = instant_temps != acclim_temp
        final_temps  = instant_temps[sel]
        final_gpp    = gpp_clean[sel]
        final_anet   = anet_clean[sel]
        final_ac     = ac_clean[sel]
        final_aj     = aj_clean[sel]
        final_jmax25 = jmax25_clean[sel]
        final_jmax   = jmax_clean[sel]
        final_J      = J_clean[sel]
        final_vcmax25 = vcmax25_clean[sel]
        final_vcmax  = vcmax_clean[sel]

        if len(final_aj) > 0:
            idx_peak         = np.argmax(final_aj)
            topt_aj_val      = final_temps[idx_peak]
            jmax_at_topt_val = final_jmax[idx_peak]
            J_at_topt_val    = final_J[idx_peak]
            aj_peak_val      = final_aj[idx_peak]
        else:
            topt_aj_val = jmax_at_topt_val = J_at_topt_val = aj_peak_val = np.nan

        results_by_light[acclim_ppfd].append({
            'acclim_temp':     acclim_temp,
            'instant_ppfd':    instant_ppfd,
            'acclim_ppfd':     acclim_ppfd,
            'instant_temps':   final_temps,
            'gpp_values':      final_gpp,
            'anet_values':     final_anet,
            'ac_values':       final_ac,
            'aj_values':       final_aj,
            'jmax25_values':   final_jmax25,
            'jmax_values':     final_jmax,
            'vcmax_values':    final_vcmax,
            'vcmax25_values': final_vcmax25,
            'J_values':        final_J,
            'topt_aj':         topt_aj_val,
            'jmax_at_topt':    jmax_at_topt_val,
            'J_at_topt':       J_at_topt_val,
            'aj_peak':         aj_peak_val,
            'ac_at_growth':    ac_at_growth_val,
            'aj_at_growth':    aj_at_growth_val,
            'anet_at_growth':  anet_at_growth_val,
        })

    all_results_energy_jmax[exp_name] = copy.deepcopy(results_by_light)
    print(f"  完成！共 {len(results_by_light[acclim_ppfd])} 条记录")

# ── 方便访问 ──
exp_ejmax_low_low   = all_results_energy_jmax['low_acclim_low_instant'][250]
exp_ejmax_high_high = all_results_energy_jmax['high_acclim_high_instant'][1800]
exp_ejmax_low_high  = all_results_energy_jmax['low_acclim_high_instant'][250]
exp_ejmax_equal     = all_results_energy_jmax['equal'][250]

print("\n=== Energy Jmax 实验完成 ===")
for name, config in experiments_config_energy_jmax.items():
    print(f"{name:30s}: acclim={config['acclim_ppfd']:4d}, "
          f"instant={config['instant_ppfd']:4d}, "
          f"ratio={config['instant_ppfd']/config['acclim_ppfd']:.3f}")

# ── 结果表格 ──


# ============================================================
# 4.68 Fixed c* -> exp_cstar_low_high
# ============================================================

import copy

experiments_config_cstar = {
    'low_acclim_low_instant':   {'acclim_ppfd': 250,  'instant_ppfd': 167},
    'high_acclim_high_instant': {'acclim_ppfd': 1800, 'instant_ppfd': 1200},
    'low_acclim_high_instant':  {'acclim_ppfd': 250,  'instant_ppfd': 600},
    'equal':                    {'acclim_ppfd': 250,  'instant_ppfd': 250},
}

HA_VCMAX25 = 65330
HA_JMAX25  = 43900
R          = 8.3145

all_results_cstar = {}

for exp_name, exp_config in experiments_config_cstar.items():
    acclim_ppfd  = exp_config['acclim_ppfd']
    instant_ppfd = exp_config['instant_ppfd']

    print(f"\n=== Running cstar: {exp_name} | acclim={acclim_ppfd}, "
          f"instant={instant_ppfd}, ratio={acclim_ppfd/instant_ppfd:.3f} ===")

    results_by_light = {acclim_ppfd: []}

    for acclim_temp in acclim_temps:
        tc_1d     = np.linspace(0, 40, n_pts).copy()
        patm_use  = patm.copy()
        co2_use   = co2.copy()
        fapar_use = fapar.copy()

        ppfd_use  = np.full(n_pts, float(instant_ppfd))
        tc_1d[acclim_mask]    = acclim_temp
        vpd_use   = calc_vpd(tc_1d, VP_CONSTANT)
        vpd_use   = np.clip(vpd_use, 0, None)
        vpd_use[acclim_mask]  = 1000
        ppfd_use[acclim_mask] = acclim_ppfd

        co2_acclim  = fsscaler.get_daily_means(co2_use)
        vpd_acclim  = fsscaler.get_daily_means(vpd_use)
        patm_acclim = fsscaler.get_daily_means(patm_use)

        # ── 瞬时环境：完全正常，不修改pmodel_const ──
        env = PModelEnvironment(tc=tc_1d, patm=patm_use, vpd=vpd_use, co2=co2_use)
        env.aridity_index           = np.full(n_pts, 0.0)
        env.mean_growth_temperature = np.full(n_pts, 25)

        gammastar_orig = env.gammastar.flatten().copy()
        kmm_orig       = env.kmm.flatten().copy()

        pmodel_instant = PModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_instant.estimate_productivity(fapar=fapar_use, ppfd=ppfd_use)
        fkphio_c3 = pmodel_instant.kphio.kphio

        subdailyC3 = SubdailyPModel(
            env=env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14',
            fapar=fapar_use,
            ppfd=ppfd_use,
            fs_scaler=fsscaler,
            alpha=1/15,
            allow_holdover=True,
        )

        # ── 驯化环境：使用自定义pmodel_const ──
        acclim_env = PModelEnvironment(
            tc=acclim_temp,
            patm=patm_acclim,
            vpd=vpd_acclim,
            co2=co2_acclim
        )
        acclim_env.aridity_index           = 0
        acclim_env.mean_growth_temperature = np.array([25])
        acclim_env.pmodel_const = pmodel_const   # ← 关键：自定义常数 ✓

        pmodel_acclim = PModel(
            env=acclim_env,
            method_kphio='sandoval',
            reference_kphio=1/9,
            method_optchi='prentice14'
        )
        pmodel_acclim.estimate_productivity(fapar=1, ppfd=acclim_ppfd)

        vcmax25_acclim = pmodel_acclim.vcmax * np.exp(
            (HA_VCMAX25/R) * (1/(acclim_temp+273.15) - 1/298.15))
        jmax25_acclim  = pmodel_acclim.jmax  * np.exp(
            (HA_JMAX25/R)  * (1/(acclim_temp+273.15) - 1/298.15))

        xi_real      = memory_effect(pmodel_acclim.optchi.xi, alpha=1/15)
        vcmax25_real = memory_effect(vcmax25_acclim, alpha=1/15, allow_holdover=True)
        jmax25_real  = memory_effect(jmax25_acclim,  alpha=1/15, allow_holdover=True)

        vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
        jmax25_subdaily  = fsscaler.fill_daily_to_subdaily(jmax25_real)

        vcmax_subdaily = vcmax25_subdaily * np.exp(
            (HA_VCMAX25/R) * (1/298.15 - 1/(tc_1d+273.15)))
        jmax_subdaily  = jmax25_subdaily  * np.exp(
            (HA_JMAX25/R)  * (1/298.15 - 1/(tc_1d+273.15)))

        # ── chi ──
        xi_subdaily  = fsscaler.fill_daily_to_subdaily(xi_real)
        subdaily_chi = OptimalChiPrentice14(env=env)
        subdaily_chi.estimate_chi(xi_values=xi_subdaily)
        ac  = vcmax_subdaily * (subdaily_chi.ci - gammastar_orig) / (subdaily_chi.ci + kmm_orig)
        J   = 4 * fkphio_c3 * ppfd_use / np.sqrt(
                1 + (4 * fkphio_c3 * ppfd_use / jmax_subdaily)**2)
        aj  = (J / 4) * subdaily_chi.mj
        gpp  = np.minimum(ac, aj)
        anet = gpp - 0.015 * vcmax_subdaily

        # ── mask ──
        mask          = np.isfinite(gpp)
        instant_temps = tc_1d[mask]
        gpp_clean     = gpp[mask]
        anet_clean    = anet[mask]
        ac_clean      = ac[mask]
        aj_clean      = aj[mask]
        jmax25_clean  = jmax25_subdaily[mask]
        jmax_clean    = jmax_subdaily[mask]
        J_clean       = J[mask]
        vcmax_clean   = vcmax_subdaily[mask]

        idx_at_acclim = np.where(np.isclose(instant_temps, acclim_temp))[0]
        if len(idx_at_acclim) > 0:
            i_acc              = idx_at_acclim[0]
            ac_at_growth_val   = ac_clean[i_acc]
            aj_at_growth_val   = aj_clean[i_acc]
            anet_at_growth_val = anet_clean[i_acc]
        else:
            ac_at_growth_val   = np.nan
            aj_at_growth_val   = np.nan
            anet_at_growth_val = np.nan

        sel          = instant_temps != acclim_temp
        final_temps  = instant_temps[sel]
        final_gpp    = gpp_clean[sel]
        final_anet   = anet_clean[sel]
        final_ac     = ac_clean[sel]
        final_aj     = aj_clean[sel]
        final_jmax25 = jmax25_clean[sel]
        final_jmax   = jmax_clean[sel]
        final_J      = J_clean[sel]
        final_vcmax  = vcmax_clean[sel]

        if len(final_aj) > 0:
            idx_peak         = np.argmax(final_aj)
            topt_aj_val      = final_temps[idx_peak]
            jmax_at_topt_val = final_jmax[idx_peak]
            J_at_topt_val    = final_J[idx_peak]
            aj_peak_val      = final_aj[idx_peak]
        else:
            topt_aj_val = jmax_at_topt_val = J_at_topt_val = aj_peak_val = np.nan

        results_by_light[acclim_ppfd].append({
            'acclim_temp':     acclim_temp,
            'instant_ppfd':    instant_ppfd,
            'acclim_ppfd':     acclim_ppfd,
            'instant_temps':   final_temps,
            'gpp_values':      final_gpp,
            'anet_values':     final_anet,
            'ac_values':       final_ac,
            'aj_values':       final_aj,
            'jmax25_values':   final_jmax25,
            'jmax_values':     final_jmax,
            'vcmax_values':    final_vcmax,
            'J_values':        final_J,
            'topt_aj':         topt_aj_val,
            'jmax_at_topt':    jmax_at_topt_val,
            'J_at_topt':       J_at_topt_val,
            'aj_peak':         aj_peak_val,
            'ac_at_growth':    ac_at_growth_val,
            'aj_at_growth':    aj_at_growth_val,
            'anet_at_growth':  anet_at_growth_val,
        })

    all_results_cstar[exp_name] = copy.deepcopy(results_by_light)
    print(f"  完成！共 {len(results_by_light[acclim_ppfd])} 条记录")

# ── 方便访问 ──
exp_cstar_low_low   = all_results_cstar['low_acclim_low_instant'][250]
exp_cstar_high_high = all_results_cstar['high_acclim_high_instant'][1800]
exp_cstar_low_high  = all_results_cstar['low_acclim_high_instant'][250]
exp_cstar_equal     = all_results_cstar['equal'][250]

print("\n=== Cstar 实验完成 ===")
for name, config in experiments_config_cstar.items():
    print(f"{name:30s}: acclim={config['acclim_ppfd']:4d}, "
          f"instant={config['instant_ppfd']:4d}, "
          f"ratio={config['instant_ppfd']/config['acclim_ppfd']:.3f}")

# ── 结果表格 ──


# ============================================================
# 5. Experiment groupings used by the figure
# ============================================================
configs_group1 = [
    {'exp_name': 'Full Acclimation',  'data': exp_low_high},
    {'exp_name': r'Fixed $\Gamma^*$', 'data': exp_gamma_low_high},
    {'exp_name': r'Fixed $\xi$',      'data': exp_xi_low_high},
    {'exp_name': r'Fixed $K$',        'data': exp_kmm_low_high},
    {'exp_name': r'Fixed $\phi_0$',   'data': exp_phi_low_high},
]

configs_group2 = [
    {'exp_name': r'Fixed $E_{aj}$', 'data': exp_ejmax_low_high},
    {'exp_name': r'Fixed $E_{ac}$', 'data': exp_evcmax_low_high},
    {'exp_name': r'Fixed $c^*$',    'data': exp_cstar_low_high},
]


# ============================================================
# 6. Plotting (cell 49, already fixed)
# ============================================================

# -- Self-contained helpers (local to this cell, do not rely on any other
#    cell having been run first -- this notebook has several incompatible
#    redefinitions of these same function names elsewhere) --
def _pick_peak(x, y):
    if y is None:
        return None, np.nan, np.nan
    y, x = np.asarray(y), np.asarray(x)
    if y.size == 0:
        return None, np.nan, np.nan
    idx = int(np.nanargmax(y))
    return idx, float(x[idx]), float(y[idx])

def _extract_points(records):
    tg_list, topt_list, Aopt_list, lim_list = [], [], [], []
    for rec in sorted(records, key=lambda r: r['acclim_temp']):
        tg  = float(rec['acclim_temp'])
        x   = np.asarray(rec['instant_temps'])
        A   = np.asarray(rec['anet_values'])
        Ac  = np.asarray(rec.get('ac_values', []))
        Aj  = np.asarray(rec.get('aj_values', []))
        idx, toptA, Aopt = _pick_peak(x, A)
        if idx is None: continue
        lim = None
        if Ac.size > 0 and Aj.size > 0:
            lim = "Aj" if Aj[idx] < Ac[idx] else "Ac"
        tg_list.append(tg); topt_list.append(toptA)
        Aopt_list.append(Aopt); lim_list.append(lim)
    return np.asarray(tg_list), np.asarray(topt_list), np.asarray(Aopt_list), lim_list

def _plot_one_pair(ax_t, ax_a, tg_arr, toptA_arr, Aopt_arr, lim_at_topt):
    if tg_arr.size == 0: return
    tg_fit  = np.linspace(tg_arr.min(), tg_arr.max(), 100)
    mask_Ac = np.array([l == "Ac" for l in lim_at_topt], dtype=bool)
    mask_Aj = np.array([l == "Aj" for l in lim_at_topt], dtype=bool)
    mask_na = ~(mask_Ac | mask_Aj)

    slope, intercept = np.polyfit(tg_arr, toptA_arr, 1)
    ax_t.plot(tg_fit, slope*tg_fit + intercept,
              color="gray", linestyle="--", linewidth=1.5, zorder=1)
    A_norm = Aopt_arr / np.nanmax(Aopt_arr)
    p = np.polyfit(tg_arr, A_norm, 2)
    a2, b2, c2 = p
    ax_a.plot(tg_fit, np.polyval(p, tg_fit),
              color="tab:orange", alpha=0.7, linewidth=2.0, zorder=1)

    for mask, fc, ec, mk in [
        (mask_Ac, 'white',    'tab:red', 'o'),
        (mask_Aj, 'tab:blue', 'white',   '^'),
        (mask_na, 'white',    'gray',    'o'),
    ]:
        if np.any(mask):
            ax_t.scatter(tg_arr[mask], toptA_arr[mask],
                         s=60, c=fc, edgecolors=ec, marker=mk, zorder=3)
            ax_a.scatter(tg_arr[mask], A_norm[mask],
                         s=60, c=fc, edgecolors=ec, marker=mk, zorder=3)

def _make_topt_aopt_fig(configs, title, filename, table_filename=None):
    n_cols = len(configs)

    fig_w = max(2.6 * n_cols + 1.0, 8.0)
    fig_h = 3.0 * 2 + 0.6          # 比之前略矮，因为 legend 移到顶部
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        2, n_cols,
        hspace=0.35, wspace=0.2,
    )

    axes = np.empty((2, n_cols), dtype=object)
    for c in range(n_cols):
        sharey_t = axes[0, 0] if c > 0 else None
        axes[0, c] = fig.add_subplot(gs[0, c], sharey=sharey_t)
        sharey_a = axes[1, 0] if c > 0 else None
        axes[1, c] = fig.add_subplot(gs[1, c],
                                     sharex=axes[0, c],
                                     sharey=sharey_a)

    panel_labels = [chr(ord('a') + i) for i in range(2 * n_cols)]
    fit_records  = []

    for col, cfg in enumerate(configs):
        ax_t, ax_a = axes[0, col], axes[1, col]
        tg, topt, Aopt, lim = _extract_points(cfg['data'])
        fits = _plot_one_pair(ax_t, ax_a, tg, topt, Aopt, lim)
        if fits is not None:
            fit_records.append({'experiment': cfg['exp_name'], **fits})

        # ↓↓↓ 这一行 ↓↓↓
        ax_t.text(-0.10, 1.10, f"({panel_labels[col]}) {cfg['exp_name']}",
                  transform=ax_t.transAxes, fontweight="bold", fontsize=13)

        ax_t.set_ylim(15, 31)
        ax_a.set_ylim(0.0, 1.2)

        if col == 0:
            ax_t.set_ylabel(r"$T_{opt}$ (°C)", fontsize=15)
            ax_a.set_ylabel(r"Norm. $A_{opt}$", fontsize=15)
        else:
            plt.setp(ax_t.get_yticklabels(), visible=False)
            plt.setp(ax_a.get_yticklabels(), visible=False)

        plt.setp(ax_t.get_xticklabels(), visible=False)

        ax_a.text(-0.10, 1.10, f"({panel_labels[n_cols + col]})",
                  transform=ax_a.transAxes, fontweight="bold", fontsize=13)

        for ax in (ax_t, ax_a):
            ax.tick_params(axis='both', labelsize=15)
            ax.spines[['top', 'right']].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=r'$A_c$-limited',
               markerfacecolor='white', markeredgecolor='tab:red',
               markersize=12, markeredgewidth=1.6),
        Line2D([0], [0], marker='^', color='w', label=r'$A_j$-limited',
               markerfacecolor='tab:blue', markeredgecolor='white',
               markersize=13),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.0,
               label=r'Linear fit ($T_{opt}$ vs $T_g$)'),
        Line2D([0], [0], color='tab:orange', linewidth=2.4,
               label=r'Quadratic fit'),
    ]


    if title:
        plt.suptitle(title, fontsize=15, fontweight='bold')
    fig.legend(handles=legend_elements, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, -0.1),
               fontsize=14, ncol=4,
               handletextpad=0.7, columnspacing=2.0)
    fig.supxlabel("Growth Temperature (°C)", fontsize=17)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved figure: {filename}")

_make_topt_aopt_fig(configs_group1, "",
                    "topt_aopt_group1_lowhigh.tiff",
                    "topt_aopt_group1_fits.csv")
_make_topt_aopt_fig(configs_group2, "",
                    "topt_aopt_group2_lowhigh.tiff",
                    "topt_aopt_group2_fits.csv")

# ============================================================
# 7. Vcmax at T_growth / T_opt two-panel plot (cell 31)
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

color_map = {
    'Full Acclimation':  'black',
    r'Fixed $\Gamma^*$': '#E64B35',
    r'Fixed $\xi$':      '#4DBBD5',
    r'Fixed $K$':        '#00A087',
    r'Fixed $\phi_0$':   '#F39B7F',
    r'Fixed $E_{aj}$':   '#8491B4',
    r'Fixed $E_{ac}$':   '#91D1C2',
    r'Fixed $c^*$':      '#B09C85',
}

marker_map = {
    'Full Acclimation':  'o',
    r'Fixed $\Gamma^*$': 's',
    r'Fixed $\xi$':      '^',
    r'Fixed $K$':        'v',
    r'Fixed $\phi_0$':   'D',
    r'Fixed $E_{aj}$':   'P',
    r'Fixed $E_{ac}$':   'X',
    r'Fixed $c^*$':      '*',
}

configs = [
    {'label': 'Full Acclimation',  'data': exp_low_high},
    {'label': r'Fixed $\Gamma^*$', 'data': exp_gamma_low_high},
    {'label': r'Fixed $\xi$',      'data': exp_xi_low_high},
    {'label': r'Fixed $K$',        'data': exp_kmm_low_high},
    {'label': r'Fixed $\phi_0$',   'data': exp_phi_low_high},
    {'label': r'Fixed $E_{aj}$',   'data': exp_ejmax_low_high},
    {'label': r'Fixed $E_{ac}$',   'data': exp_evcmax_low_high},
    {'label': r'Fixed $c^*$',      'data': exp_cstar_low_high},
]

acclim_temps = sorted({rec['acclim_temp'] for rec in configs[0]['data']})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

for cfg in configs:
    label   = cfg['label']
    data    = cfg['data']
    color   = color_map.get(label, '#333333')
    marker  = marker_map.get(label, 'o')
    is_base = (label == 'Full Acclimation')
    style   = dict(lw=2.5 if is_base else 1.5, marker=marker,
                   ms=7 if marker == '*' else 5, alpha=1.0,
                   zorder=5 if is_base else 2, color=color)

    vcmax_at_tg   = []   # panel 1: Vcmax at T_growth
    vcmax_at_topt = []   # panel 2: Vcmax at Topt

    for T_acc in acclim_temps:
        rec = next((r for r in data if r['acclim_temp'] == T_acc), None)
        if rec is None:
            vcmax_at_tg.append(np.nan)
            vcmax_at_topt.append(np.nan)
            continue

        t    = np.asarray(rec['instant_temps'])
        vc   = np.asarray(rec['vcmax_values'])
        gpp  = np.asarray(rec['gpp_values'])

        # Panel 1: Vcmax at T_inst = T_growth
        vcmax_at_tg.append(np.interp(T_acc, t, vc))

        # Panel 2: Vcmax at Topt (where GPP is max)
        if gpp.size > 0:
            topt = t[np.nanargmax(gpp)]
            vcmax_at_topt.append(np.interp(topt, t, vc))
        else:
            vcmax_at_topt.append(np.nan)

    ax1.plot(acclim_temps, vcmax_at_tg,   '-', label=label, **style)
    ax2.plot(acclim_temps, vcmax_at_topt, '-', label=label, **style)
    ax1.text(-0.08, 1.04, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax2.text(-0.08, 1.04, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold')

for ax, title, ylabel in [
    (ax1, r'$V_{cmax}$ at $T_{growth}$',  r'$V_{cmax}$ (µmol m⁻² s⁻¹)'),
    (ax2, r'$V_{cmax}$ at $T_{opt}$',     r'$V_{cmax}$ (µmol m⁻² s⁻¹)'),
]:
    ax.set_xlabel("Growth Temperature (°C)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0,200)

ax1.legend(fontsize=9, frameon=False, loc='best')
plt.tight_layout()
plt.savefig("acclimated_vcmax_tg_topt.png", dpi=600, bbox_inches='tight')
plt.show()
