/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__Gfluctdv
#define _nrn_initial _nrn_initial__Gfluctdv
#define nrn_cur _nrn_cur__Gfluctdv
#define _nrn_current _nrn_current__Gfluctdv
#define nrn_jacob _nrn_jacob__Gfluctdv
#define nrn_state _nrn_state__Gfluctdv
#define _net_receive _net_receive__Gfluctdv 
#define new_seed new_seed__Gfluctdv 
#define oup oup__Gfluctdv 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg(int);
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define E_e _p[0]
#define E_e_columnindex 0
#define E_i _p[1]
#define E_i_columnindex 1
#define g_e0 _p[2]
#define g_e0_columnindex 2
#define g_i0 _p[3]
#define g_i0_columnindex 3
#define std_e _p[4]
#define std_e_columnindex 4
#define std_i _p[5]
#define std_i_columnindex 5
#define tau_e _p[6]
#define tau_e_columnindex 6
#define tau_i _p[7]
#define tau_i_columnindex 7
#define i _p[8]
#define i_columnindex 8
#define g_e _p[9]
#define g_e_columnindex 9
#define g_i _p[10]
#define g_i_columnindex 10
#define g_e1 _p[11]
#define g_e1_columnindex 11
#define g_i1 _p[12]
#define g_i1_columnindex 12
#define D_e _p[13]
#define D_e_columnindex 13
#define D_i _p[14]
#define D_i_columnindex 14
#define exp_e _p[15]
#define exp_e_columnindex 15
#define exp_i _p[16]
#define exp_i_columnindex 16
#define amp_e _p[17]
#define amp_e_columnindex 17
#define amp_i _p[18]
#define amp_i_columnindex 18
#define _g _p[19]
#define _g_columnindex 19
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_new_seed(void);
 static void _hoc_oup(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_Gfluctdv", _hoc_setdata,
 "new_seed_Gfluctdv", _hoc_new_seed,
 "oup_Gfluctdv", _hoc_oup,
 0, 0
};
 /* declare global and static user variables */
#define multin multin_Gfluctdv
 double multin = 0;
#define multex multex_Gfluctdv
 double multex = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "E_e_Gfluctdv", "mV",
 "E_i_Gfluctdv", "mV",
 "g_e0_Gfluctdv", "S/cm2",
 "g_i0_Gfluctdv", "S/cm2",
 "std_e_Gfluctdv", "S/cm2",
 "std_i_Gfluctdv", "S/cm2",
 "tau_e_Gfluctdv", "ms",
 "tau_i_Gfluctdv", "ms",
 "i_Gfluctdv", "mA/cm2",
 "g_e_Gfluctdv", "S/cm2",
 "g_i_Gfluctdv", "S/cm2",
 "g_e1_Gfluctdv", "S/cm2",
 "g_i1_Gfluctdv", "S/cm2",
 "D_e_Gfluctdv", "umho umho /ms",
 "D_i_Gfluctdv", "umho umho /ms",
 0,0
};
 static double delta_t = 1;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "multex_Gfluctdv", &multex_Gfluctdv,
 "multin_Gfluctdv", &multin_Gfluctdv,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Gfluctdv",
 "E_e_Gfluctdv",
 "E_i_Gfluctdv",
 "g_e0_Gfluctdv",
 "g_i0_Gfluctdv",
 "std_e_Gfluctdv",
 "std_i_Gfluctdv",
 "tau_e_Gfluctdv",
 "tau_i_Gfluctdv",
 0,
 "i_Gfluctdv",
 "g_e_Gfluctdv",
 "g_i_Gfluctdv",
 "g_e1_Gfluctdv",
 "g_i1_Gfluctdv",
 "D_e_Gfluctdv",
 "D_i_Gfluctdv",
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 20, _prop);
 	/*initialize range parameters*/
 	E_e = 0;
 	E_i = -75;
 	g_e0 = 0.0001;
 	g_i0 = 0.0005;
 	std_e = 3e-05;
 	std_i = 6e-05;
 	tau_e = 2.728;
 	tau_i = 10.49;
 	_prop->param = _p;
 	_prop->param_size = 20;
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Gfluctdv_reg() {
	int _vectorized = 0;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 20, 0);
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Gfluctdv /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/Gfluctdv.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Fluctuating conductances";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int new_seed(double);
static int oup();
 
static int  oup (  ) {
   if ( tau_e  != 0.0 ) {
     amp_e = sqrt ( multex ) * std_e * sqrt ( ( 1.0 - exp ( - 2.0 * dt / tau_e ) ) ) ;
     g_e1 = exp_e * g_e1 + amp_e * normrand ( 0.0 , 1.0 ) ;
     }
   if ( tau_i  != 0.0 ) {
     amp_i = sqrt ( multin ) * std_i * sqrt ( ( 1.0 - exp ( - 2.0 * dt / tau_i ) ) ) ;
     g_i1 = exp_i * g_i1 + amp_i * normrand ( 0.0 , 1.0 ) ;
     }
    return 0; }
 
static void _hoc_oup(void) {
  double _r;
   _r = 1.;
 oup (  );
 hoc_retpushx(_r);
}
 
static int  new_seed (  double _lseed ) {
   set_seed ( _lseed ) ;
   
/*VERBATIM*/
	  	printf("Setting random generator with seed = %g\n", _lseed);
  return 0; }
 
static void _hoc_new_seed(void) {
  double _r;
   _r = 1.;
 new_seed (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("Gfluctdv", "cannot be used with CVODE"); return 0;}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
 {
   g_e1 = 0.0 ;
   g_i1 = 0.0 ;
   if ( tau_e  != 0.0 ) {
     D_e = 2.0 * std_e * std_e / tau_e ;
     exp_e = exp ( - dt / tau_e ) ;
     amp_e = sqrt ( multex ) * std_e * sqrt ( ( 1.0 - exp ( - 2.0 * dt / tau_e ) ) ) ;
     }
   if ( tau_i  != 0.0 ) {
     D_i = 2.0 * std_i * std_i / tau_i ;
     exp_i = exp ( - dt / tau_i ) ;
     amp_i = sqrt ( multin ) * std_i * sqrt ( ( 1.0 - exp ( - 2.0 * dt / tau_i ) ) ) ;
     }
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   if ( tau_e  == 0.0 ) {
     g_e = std_e * normrand ( 0.0 , 1.0 ) ;
     }
   if ( tau_i  == 0.0 ) {
     g_i = std_i * normrand ( 0.0 , 1.0 ) ;
     }
   g_e = multex * g_e0 + g_e1 ;
   if ( g_e < 0.0 ) {
     g_e = 0.0 ;
     }
   g_i = multin * g_i0 + g_i1 ;
   if ( g_i < 0.0 ) {
     g_i = 0.0 ;
     }
   i = g_e * ( v - E_e ) + g_i * ( v - E_i ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 { error =  oup();
 if(error){fprintf(stderr,"at line 153 in file Gfluctdv.mod:\n	SOLVE oup\n"); nrn_complain(_p); abort_run(error);}
 }}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/Gfluctdv.mod";
static const char* nmodl_file_text = 
  "TITLE Fluctuating conductances\n"
  "\n"
  "COMMENT\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "	Fluctuating conductance model for synaptic bombardment\n"
  "	======================================================\n"
  "\n"
  "THEORY\n"
  "\n"
  "  Synaptic bombardment is represented by a stochastic model containing\n"
  "  two fluctuating conductances g_e(t) and g_i(t) descibed by:\n"
  "\n"
  "     Isyn = g_e(t) * [V - E_e] + g_i(t) * [V - E_i]\n"
  "     d g_e / dt = -(g_e - g_e0) / tau_e + sqrt(D_e) * Ft\n"
  "     d g_i / dt = -(g_i - g_i0) / tau_i + sqrt(D_i) * Ft\n"
  "\n"
  "  where E_e, E_i are the reversal potentials, g_e0, g_i0 are the average\n"
  "  conductances, tau_e, tau_i are time constants, D_e, D_i are noise diffusion\n"
  "  coefficients and Ft is a gaussian white noise of unit standard deviation.\n"
  "\n"
  "  g_e and g_i are described by an Ornstein-Uhlenbeck (OU) stochastic process\n"
  "  where tau_e and tau_i represent the \"correlation\" (if tau_e and tau_i are \n"
  "  zero, g_e and g_i are white noise).  The estimation of OU parameters can\n"
  "  be made from the power spectrum:\n"
  "\n"
  "     S(w) =  2 * D * tau^2 / (1 + w^2 * tau^2)\n"
  "\n"
  "  and the diffusion coeffient D is estimated from the variance:\n"
  "\n"
  "     D = 2 * sigma^2 / tau\n"
  "\n"
  "\n"
  "NUMERICAL RESOLUTION\n"
  "\n"
  "  The numerical scheme for integration of OU processes takes advantage \n"
  "  of the fact that these processes are gaussian, which led to an exact\n"
  "  update rule independent of the time step dt (see Gillespie DT, Am J Phys \n"
  "  64: 225, 1996):\n"
  "\n"
  "     x(t+dt) = x(t) * exp(-dt/tau) + A * N(0,1)\n"
  "\n"
  "  where A = sqrt( D*tau/2 * (1-exp(-2*dt/tau)) ) and N(0,1) is a normal\n"
  "  random number (avg=0, sigma=1)\n"
  "\n"
  "\n"
  "IMPLEMENTATION\n"
  "\n"
  "  This version has changed from point process nonspecific current to density\n"
  "\n"
  "\n"
  "PARAMETERS\n"
  "\n"
  "  The mechanism takes the following parameters:\n"
  "\n"
  "     E_e = 0  (mV)		: reversal potential of excitatory conductance\n"
  "     E_i = -75 (mV)		: reversal potential of inhibitory conductance\n"
  "\n"
  "     g_e0 = 0.0001 (S/cm2)	: average excitatory conductance\n"
  "     g_i0 = 0.0005 (S/cm2)	: average inhibitory conductance\n"
  "\n"
  "     std_e = 3e-5 (S/cm2)	: standard dev of excitatory conductance\n"
  "     std_i = 6e-5 (S/cm2)	: standard dev of inhibitory conductance\n"
  "\n"
  "     tau_e = 2.728 (ms)		: time constant of excitatory conductance\n"
  "     tau_i = 10.49 (ms)		: time constant of inhibitory conductance\n"
  "\n"
  "\n"
  "Gfluct2: conductance cannot be negative\n"
  "\n"
  "\n"
  "REFERENCE\n"
  "\n"
  "  Destexhe, A., Rudolph, M., Fellous, J-M. and Sejnowski, T.J.  \n"
  "  Fluctuating synaptic conductances recreate in-vivo--like activity in\n"
  "  neocortical neurons. Neuroscience 107: 13-24 (2001).\n"
  "\n"
  "  (electronic copy available at http://cns.iaf.cnrs-gif.fr)\n"
  "\n"
  "\n"
  "  A. Destexhe, 1999\n"
  "Modified 04/09/08 by RKP so that current can be varied continuously over the course of a simulation\n"
  "-----------------------------------------------------------------------------\n"
  "ENDCOMMENT\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX Gfluctdv\n"
  "	RANGE g_e, g_i, E_e, E_i, g_e0, g_i0, g_e1, g_i1\n"
  "	RANGE std_e, std_i, tau_e, tau_i, D_e, D_i\n"
  "	RANGE new_seed\n"
  "	GLOBAL multex,multin\n"
  "	NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mV) = (millivolt)\n"
  "	(mA) = (milliamp)\n"
  "   	(S) = (siemens)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	dt					(ms)\n"
  "\n"
  "	E_e	= 		0 		(mV)	: reversal potential of excitatory conductance\n"
  "	E_i	= 		-75 	(mV)	: reversal potential of inhibitory conductance\n"
  "\n"
  "	g_e0 = 		0.0001 	(S/cm2)	: average excitatory conductance\n"
  "	g_i0 = 		0.0005 	(S/cm2)	: average inhibitory conductance\n"
  "\n"
  "	std_e = 	3e-5 	(S/cm2)	: standard dev of excitatory conductance\n"
  "	std_i = 	6e-5 	(S/cm2)	: standard dev of inhibitory conductance\n"
  "\n"
  "	tau_e = 	2.728	(ms)	: time constant of excitatory conductance\n"
  "	tau_i = 	10.49	(ms)	: time constant of inhibitory conductance\n"
  "\n"
  "	multex = 	0\n"
  "	multin = 	0\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v					(mV)			: membrane voltage\n"
  "	i 					(mA/cm2)		: fluctuating current\n"
  "	g_e					(S/cm2)			: total excitatory conductance\n"
  "	g_i					(S/cm2)			: total inhibitory conductance\n"
  "	g_e1				(S/cm2)			: fluctuating excitatory conductance\n"
  "	g_i1				(S/cm2)			: fluctuating inhibitory conductance\n"
  "	D_e					(umho umho /ms) : excitatory diffusion coefficient\n"
  "	D_i					(umho umho /ms) : inhibitory diffusion coefficient\n"
  "	exp_e\n"
  "	exp_i\n"
  "	amp_e				(umho)\n"
  "	amp_i				(umho)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	g_e1 = 0\n"
  "	g_i1 = 0\n"
  "	if(tau_e != 0) {\n"
  "		D_e = 2 * std_e * std_e / tau_e\n"
  "		exp_e = exp(-dt/tau_e)\n"
  "		amp_e = sqrt(multex)*std_e * sqrt( (1-exp(-2*dt/tau_e)) )\n"
  "	}\n"
  "	if(tau_i != 0) {\n"
  "		D_i = 2 * std_i * std_i / tau_i\n"
  "		exp_i = exp(-dt/tau_i)\n"
  "		amp_i = sqrt(multin)*std_i * sqrt( (1-exp(-2*dt/tau_i)) )\n"
  "	}\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE oup\n"
  "	if(tau_e==0) {\n"
  "	   g_e = std_e * normrand(0,1)\n"
  "	}\n"
  "	if(tau_i==0) {\n"
  "	   g_i = std_i * normrand(0,1)\n"
  "	}\n"
  "	g_e = multex*g_e0 + g_e1\n"
  "	if(g_e < 0) { g_e = 0 }\n"
  "	g_i = multin* g_i0 + g_i1\n"
  "	if(g_i < 0) { g_i = 0 }\n"
  "	i = g_e * (v - E_e) + g_i * (v - E_i)\n"
  "}\n"
  "\n"
  "\n"
  "PROCEDURE oup() {	: use Scop function normrand(mean, std_dev)\n"
  "   if(tau_e!=0) {\n"
  "		amp_e = sqrt(multex)*std_e * sqrt( (1-exp(-2*dt/tau_e)) )\n"
  "		g_e1 =  exp_e * g_e1 + amp_e * normrand(0,1)\n"
  "   }\n"
  "   if(tau_i!=0) {\n"
  "		amp_i = sqrt(multin)*std_i * sqrt( (1-exp(-2*dt/tau_i)) )\n"
  "		g_i1 =  exp_i * g_i1 + amp_i * normrand(0,1)\n"
  "   }\n"
  "}\n"
  "\n"
  "\n"
  "PROCEDURE new_seed(seed) {	: procedure to set the seed\n"
  "	set_seed(seed)\n"
  "	VERBATIM\n"
  "	  	printf(\"Setting random generator with seed = %g\\n\", _lseed);\n"
  "	ENDVERBATIM\n"
  "}\n"
  "\n"
  ;
#endif
