import itertools
from typing import Optional

import numpy as np
from neuron import h

from myogen import RANDOM_GENERATOR, SEED
from myogen.simulator.neuron._cython._poisson_process_generator import (
    _PoissonProcessGenerator__Cython,
)


class _Cell:
    _gid__iterator = itertools.count(0)

    def __init__(self, class_ID: int, pool_ID: int | None = None):
        self.global_ID = next(self._gid__iterator)
        self.class_ID = class_ID
        self.pool_ID = pool_ID

        self._create_sections()
        self._build_topology()
        self._define_geometry()
        self._define_biophysics()

        self.create_axon()
        self.reclist = []
        self.synlist = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} [global ID: {self.global_ID}, class ID: {self.class_ID}, pool ID: {self.pool_ID}]"

    def _create_sections(self):
        pass

    def _build_topology(self):
        pass

    def _define_geometry(self):
        pass

    def _define_biophysics(self):
        pass

    def create_axon(self, length__m: float = 0, velcon__m_per_s: float = 50):
        self.len = length__m  # m, length
        self.velcon = velcon__m_per_s  # m/s, conduction velocity
        self.axonDelay = self.len / self.velcon * 1e3  # [ms]

    def plot_voltage(self):
        """Plot the recorded traces"""
        import pylab as plt

        for vec in self.reclist:
            plt.plot(self.t_vec, vec[0], label=str(vec[1]) + "(" + str(vec[2]) + ")")
        plt.legend()
        plt.ylabel("mV")

    def create_synapses(self, loc, e=0, tau1=0.2, tau2=0.3):
        """Add an exponentially decaying synapse
        exp2Syn synapses with tau1=0.2, tau2=0.3 and e= 70 were
        obtained from modals(Sanches, 2018). The weigth of the
        synapses used were weigth = 0.6
        Segev et al. (1990) uses alpha function to model synapses with
        the following param.  Esyn = vHold + 70

        """
        assert tau2 > tau1
        syn = h.Exp2Syn(loc(0.5))
        syn.tau1 = tau1
        syn.tau2 = tau2
        if hasattr(self, "model") and self.model == "ModALS":
            syn.e = 70
        else:
            syn.e = e
        self.synlist.append(syn)  # synlist is defined in Cell
        return syn

    def create_netcon(self, target, threshold=-10, delay=1, weight=1):
        """created netcon to record spikes"""
        nc = h.NetCon(self.soma(0.5)._ref_v, target, sec=self.soma)
        nc.threshold = threshold
        nc.delay = delay
        nc.weight[0] = weight
        self.nclist.append(nc)

    def set_recording(self, sec, loc=0.5):
        v_vec = h.Vector()
        self.reclist.append([v_vec.record(sec(loc)._ref_v), sec, loc])
        if "self.t_vec" not in locals():
            self.t_vec = h.Vector()  # Time stamp vector
            self.t_vec.record(h._ref_t)

    def add_current_stim(self, sec, loc=0.5, delay=5, amp=0.3, dur=1):
        self.stim = h.IClamp(sec(loc))
        self.stim.amp = amp  # input current in nA
        self.stim.delay = delay  # turn on after this time in ms
        self.stim.dur = dur  # duration of 1 ms


# INTERNEURONS
class INgII(_Cell):
    _ids2 = itertools.count(0)

    def __init__(self, class_ID: Optional[int] = None, pool_ID: int | None = None):
        super().__init__(
            class_ID if class_ID is not None else next(self._ids2), pool_ID
        )
        self.create_synapses(self.soma)

    def _create_sections(self):
        """Create the sections of the cell."""
        self.soma = h.Section(name="soma", cell=self)

    def _define_geometry(self):
        """Set the 3D geometry of the cell."""
        # Data from Bui et al., 2003 Ia Inhibitory interneurons
        Amu = 81390 + 3113
        Aci = 1.96 * (891.5 + 46.141) / np.sqrt(8)
        i = 0
        A = [Amu - Aci, Amu + Aci]
        D = np.sqrt(A[i] / np.pi)
        self.soma.L = D
        self.soma.diam = D
        self.soma.Ra = 70
        self.soma.cm = 1
        self.soma.nseg = 1

    def _define_biophysics(self):
        self.soma.insert("pas")
        self.soma.g_pas = 5e-5  # Rin, Tau
        self.soma.e_pas = -71.0

        for mech in ["na3rp", "kdrRL", "gh", "mAHP"]:
            self.soma.insert(mech)

        self.soma.ghbar_gh = 2.5e-5  # Rin, Tau, AHPdur
        self.soma.half_gh = -70.0  # AHPdur
        self.soma.ek = -80.0
        self.soma.gbar_na3rp = 0.003  # APamp, Ir (direct) [0.01,0.022]
        self.soma.qinf_na3rp = 8.0  # AP amp (inverse)
        self.soma.sh_na3rp = 1.0  # Rheobase (direct)
        self.soma.thinf_na3rp = -50
        self.soma.gMax_kdrRL = 0.015  # Rheobase, AHPdur, AHPmag (direct!)
        self.soma.gcamax_mAHP = 3e-6  # AHP MAG (direc prop.)
        self.soma.gkcamax_mAHP = 0.0005  # AHP MAG (direc prop.)
        self.soma.taur_mAHP = 70.0  # AHP DURATION (direct prop)


class INgIb(INgII):
    _ids2 = itertools.count(0)

    def __init__(self, pool_ID: int | None = None):
        super().__init__(next(self._ids2), pool_ID)


# AFFERENTS AND DESCENDING TRACTS
class DD(_Cell, _PoissonProcessGenerator__Cython):
    _ids2 = itertools.count(0)

    def __init__(self, N, dt, pool_ID: int | None = None):
        self.ns = h.DUMMY()  # Dummy cell
        _Cell.__init__(self, next(self._ids2), pool_ID)
        _PoissonProcessGenerator__Cython.__init__(
            self, SEED + (self.class_ID + 1) * (self.global_ID + 1), N, dt
        )

    def __repr__(self) -> str:
        return _Cell.__repr__(self)

    def integrate(self, y):
        return self.compute(y) if y > 0 else 0


class AffIa(_Cell, _PoissonProcessGenerator__Cython):
    _ids2 = itertools.count(0)

    def __init__(
        self,
        RT,
        N,
        dt,
        initN=0,
        class_ID: Optional[int] = None,
        pool_ID: int | None = None,
    ):
        self.ns = h.DUMMY()  # Dummy cell

        self.RT = RT  # Recruitment Threshold
        self.IFR = RANDOM_GENERATOR.normal(5, 2.5)  # Variability

        _Cell.__init__(
            self, class_ID if class_ID is not None else next(self._ids2), pool_ID
        )
        _PoissonProcessGenerator__Cython.__init__(
            self,
            seed=SEED + (self.class_ID + 1) * (self.global_ID + 1),
            N=N,
            dt=dt,
            Ninit=initN,
        )

    def integrate(self, y):
        act = y - self.RT + self.IFR
        return self.compute(act) if act > 0 else 0


class AffII(AffIa):
    _ids2 = itertools.count(0)

    def __init__(self, RT, N, pool_ID: int | None = None, *args, **kwargs):
        super().__init__(
            RT, N, class_ID=next(self._ids2), pool_ID=pool_ID, *args, **kwargs
        )


class AffIb(AffIa):
    _ids2 = itertools.count(0)

    def __init__(self, RT, N, pool_ID: int | None = None, *args, **kwargs):
        super().__init__(
            RT, N, class_ID=next(self._ids2), pool_ID=pool_ID, *args, **kwargs
        )


# MOTORNEURON
class AlphaMN(_Cell):
    _ids2 = itertools.count(0)

    """HH cell: A soma with active channels"""

    def __init__(
        self,
        nseg=1,  # Number of segments
        mode="active",  # 'active' or 'pas' model
        n_dend=4,  # Number of dendrites
        model="Powers2017",
        rid=None,
        pool_ID: int | None = None,
        *args,
        **kwargs,
    ):
        if self.__class__.__name__ == "AlphaMN" and rid is None:
            self.rid = next(self._ids2)
        if rid is not None:
            self.rid = rid
        self.n_dend = n_dend
        self.nseg = nseg
        self.mode = mode
        self.model = model
        super().__init__(class_ID=self.rid, pool_ID=pool_ID, *args, **kwargs)

        for d in self.dend:
            self.create_synapses(d)

    def _create_sections(self):
        """Create the sections of the cell."""
        self.soma = h.Section(name="soma", cell=self)
        self.dend = [h.Section(name="dend", cell=self) for _ in range(self.n_dend)]

    def _define_geometry(self):
        """Set the 3D geometry of the cell."""
        self.soma.L = 2952
        self.soma.diam = 22
        self.soma.Ra = 0.001
        self.soma.cm = 1.35546
        self.soma.nseg = self.nseg

        for d in self.dend:
            if self.n_dend == 4:
                d.L = 1794.13
                d.diam = 8.73071
            if self.n_dend == 1:
                d.L = 2848
                d.diam = 22
            d.Ra = 51.038
            d.cm = 0.867781
            d.nseg = self.nseg

    def _build_topology(self):
        for i, d in enumerate(self.dend):
            d.connect(self.soma(i % 2))

    def insert_Gfluctdv(self):
        for d in self.dend:
            d.insert("Gfluctdv")
            # h.new_seed(seed)
            d.g_e0_Gfluctdv = 1e-5
            d.std_e_Gfluctdv = 1.2e-5  # 6e-6
            d.tau_e_Gfluctdv = 20  # 0.5
            d.g_i0_Gfluctdv = 1e-5
            d.std_i_Gfluctdv = 1.2e-5  # 6e-6
            d.tau_i_Gfluctdv = 20  # 2

    def _define_biophysics(self, gamma=1):
        """Assign the membrane properties across the cell."""
        if self.model == "ModALS":
            if self.mode == "active":
                self.soma.insert("napp")
                self.soma.insert("Constant")
                self.dend[0].insert("caL")
                self.dend[0].insert("Constant")
            else:
                print("ModALS passive model is not defined yet.")
                print("try active mode.")

        if self.model == "Powers2017":
            # Channel types and their roles:
            # PIC Channels:
            #   - L_CA_inact: L-type Calcium channels
            #   - nas: Na slow inactivation Channel
            # AP Channels:
            #   - na3rp: Na current
            #   - kdrRL: Potassium Delayed Rectifier Channel
            # AHP Channel:
            #   - mAHP: Calcium-dependent potassium Channel
            # Passive Channels:
            #   - gh: Hodgkin-Huxley Potassium h channel
            #   - pas: passive mechanisms

            # Soma passive properties
            self.soma.insert("pas")
            self.soma.g_pas = 8.109e-05
            self.soma.e_pas = -71.0

            # Active soma mechanisms
            if self.mode == "active":
                for mech in ["na3rp", "naps", "kdrRL", "mAHP", "gh"]:
                    self.soma.insert(mech)

                # Sodium channels (na3rp)
                self.soma.gbar_na3rp = 0.01
                self.soma.sh_na3rp = 1.0
                self.soma.ar_na3rp = 1.0
                self.soma.qinf_na3rp = 8.0
                self.soma.thinf_na3rp = -50

                # Persistent sodium channels (naps)
                self.soma.gbar_naps = 2.6e-05
                self.soma.sh_naps = 5.0
                self.soma.ar_naps = 1.0

                # Potassium delayed rectifier (kdrRL)
                self.soma.gMax_kdrRL = 0.015

                # Calcium-dependent potassium (mAHP)
                self.soma.gcamax_mAHP = 6.4e-06
                self.soma.gkcamax_mAHP = 0.00045
                self.soma.taur_mAHP = 90.0
                self.soma.ek = -80.0

                # H-current (gh)
                self.soma.ghbar_gh = 3e-05
                self.soma.half_gh = -77.0

            # Global channel parameters
            h.vslope_naps = 5  # activation slope for persistent sodium channels (mV)
            h.asvh_naps = (
                -90
            )  # slow inactivation voltage half-point for persistent sodium (mV)
            h.bsvh_naps = (
                -22
            )  # slow inactivation voltage parameter for persistent sodium (mV)
            h.mvhalfca_mAHP = (
                -22
            )  # calcium activation voltage half-point for Ca-dependent K channels (mV)
            h.mtauca_mAHP = 2  # calcium time constant for Ca-dependent K channels (ms)
            h.tau_m_L_Ca_inact = (
                40  # activation time constant for L-type calcium channels (ms)
            )
            h.tau_h_L_Ca_inact = (
                2500.0  # inactivation time constant for L-type calcium channels (ms)
            )
            h.kappa_h_L_Ca_inact = (
                5.0  # inactivation slope factor for L-type calcium channels
            )
            h.mVh_kdrRL = (
                -21.0
            )  # half-activation voltage for K delayed rectifier channels (mV)
            h.tmin_kdrRL = (
                0.8  # minimum time constant for K delayed rectifier channels (ms)
            )
            h.taumax_kdrRL = (
                20.0  # maximum time constant for K delayed rectifier channels (ms)
            )
            h.htau_gh = 30.0  # time constant for h-current channels (ms)

            # Dendritic mechanisms
            if self.mode == "pas":
                dendMechs = ["pas"]
            elif self.mode == "active":
                dendMechs = ["pas", "L_Ca_inact", "gh"]
                if self.n_dend == 1:
                    gca = [1.05e-4]
                else:
                    gca = [8.5e-05, 9.5e-5, 1e-4, 1.15e-4]

            # Apply dendritic properties
            for i, d in enumerate(self.dend):
                for mech in dendMechs:
                    d.insert(mech)
                d.g_pas = 7.93445e-05
                d.e_pas = -71.0

                if self.mode == "active":
                    d.gcabar_L_Ca_inact = gca[i] * gamma
                    d.ghbar_gh = 3e-05
                    d.half_gh = -77.0
                    d.theta_m_L_Ca_inact = -42.0
                    d.theta_h_L_Ca_inact = 10.0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import neo
    import quantities as pq
    from elephant import kernels, statistics

    from myogen import setup_myogen

    setup_myogen()

    # Sim parameters
    tstop = 20000  # [ms]
    dt = 0.05  # [ms]
    t = np.arange(0, tstop + dt, dt)
    S = np.zeros(len(t), dtype=np.intc)
    # Modulating signals
    fs = 0.5  # Modulating frequency [Hz]
    # y = 50 * np.sin(2 * np.pi * fs * t / 1000) + 10  # Sinusoidal
    # y = 10 + 5*(t/1000)
    # y = 20*np.ones(len(t))
    y = np.interp(t, [0, tstop / 2, tstop], [0, 50, 0])
    x = AffII(RT=25, N=25, dt=dt)
    for i in range(len(t)):
        if y[i] >= x.RT:
            S[i] = x.integrate(y[i])
    window = 250

    yest = statistics.instantaneous_rate(
        neo.SpikeTrain(np.argwhere(S)[:, 0] * dt * pq.ms, tstop),
        sampling_period=dt * pq.ms,
        # kernel=kernels.RectangularKernel(sigma=window * pq.ms),
        border_correction=True,
    )[:, 0]

    plt.plot(t, y)
    plt.plot(t, S)
    plt.plot(yest.times, yest.magnitude)
    plt.show()
