/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
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
 
#define nrn_init _nrn_init__napp
#define _nrn_initial _nrn_initial__napp
#define nrn_cur _nrn_cur__napp
#define _nrn_current _nrn_current__napp
#define nrn_jacob _nrn_jacob__napp
#define nrn_state _nrn_state__napp
#define _net_receive _net_receive__napp 
#define rates rates__napp 
#define states states__napp 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg(int);
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gnabar _p[0]
#define gnabar_columnindex 0
#define gnapbar _p[1]
#define gnapbar_columnindex 1
#define gkfbar _p[2]
#define gkfbar_columnindex 2
#define gksbar _p[3]
#define gksbar_columnindex 3
#define gl _p[4]
#define gl_columnindex 4
#define el _p[5]
#define el_columnindex 5
#define vtraub _p[6]
#define vtraub_columnindex 6
#define mact _p[7]
#define mact_columnindex 7
#define rinact _p[8]
#define rinact_columnindex 8
#define m_alpha_A _p[9]
#define m_alpha_A_columnindex 9
#define m_alpha_v_offset _p[10]
#define m_alpha_v_offset_columnindex 10
#define m_alpha_k _p[11]
#define m_alpha_k_columnindex 11
#define m_beta_A _p[12]
#define m_beta_A_columnindex 12
#define m_beta_v_offset _p[13]
#define m_beta_v_offset_columnindex 13
#define m_beta_k _p[14]
#define m_beta_k_columnindex 14
#define h_alpha_A _p[15]
#define h_alpha_A_columnindex 15
#define h_alpha_v_offset _p[16]
#define h_alpha_v_offset_columnindex 16
#define h_alpha_tau _p[17]
#define h_alpha_tau_columnindex 17
#define h_beta_A _p[18]
#define h_beta_A_columnindex 18
#define h_beta_v_offset _p[19]
#define h_beta_v_offset_columnindex 19
#define h_beta_k _p[20]
#define h_beta_k_columnindex 20
#define p_alpha_A _p[21]
#define p_alpha_A_columnindex 21
#define p_alpha_v_offset _p[22]
#define p_alpha_v_offset_columnindex 22
#define p_alpha_k _p[23]
#define p_alpha_k_columnindex 23
#define p_beta_A _p[24]
#define p_beta_A_columnindex 24
#define p_beta_v_offset _p[25]
#define p_beta_v_offset_columnindex 25
#define p_beta_k _p[26]
#define p_beta_k_columnindex 26
#define n_alpha_A _p[27]
#define n_alpha_A_columnindex 27
#define n_alpha_v_offset _p[28]
#define n_alpha_v_offset_columnindex 28
#define n_alpha_k _p[29]
#define n_alpha_k_columnindex 29
#define n_beta_A _p[30]
#define n_beta_A_columnindex 30
#define n_beta_v_offset _p[31]
#define n_beta_v_offset_columnindex 31
#define n_beta_tau _p[32]
#define n_beta_tau_columnindex 32
#define r_alpha_A _p[33]
#define r_alpha_A_columnindex 33
#define r_alpha_v_offset _p[34]
#define r_alpha_v_offset_columnindex 34
#define r_alpha_k _p[35]
#define r_alpha_k_columnindex 35
#define gna _p[36]
#define gna_columnindex 36
#define gnap _p[37]
#define gnap_columnindex 37
#define gkf _p[38]
#define gkf_columnindex 38
#define gks _p[39]
#define gks_columnindex 39
#define inap _p[40]
#define inap_columnindex 40
#define inaf _p[41]
#define inaf_columnindex 41
#define ikf _p[42]
#define ikf_columnindex 42
#define iks _p[43]
#define iks_columnindex 43
#define il _p[44]
#define il_columnindex 44
#define m _p[45]
#define m_columnindex 45
#define h _p[46]
#define h_columnindex 46
#define p _p[47]
#define p_columnindex 47
#define n _p[48]
#define n_columnindex 48
#define r _p[49]
#define r_columnindex 49
#define Dm _p[50]
#define Dm_columnindex 50
#define Dh _p[51]
#define Dh_columnindex 51
#define Dp _p[52]
#define Dp_columnindex 52
#define Dn _p[53]
#define Dn_columnindex 53
#define Dr _p[54]
#define Dr_columnindex 54
#define ena _p[55]
#define ena_columnindex 55
#define ek _p[56]
#define ek_columnindex 56
#define ina _p[57]
#define ina_columnindex 57
#define ik _p[58]
#define ik_columnindex 58
#define v _p[59]
#define v_columnindex 59
#define _g _p[60]
#define _g_columnindex 60
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
#define _ion_ek	*_ppvar[3]._pval
#define _ion_ik	*_ppvar[4]._pval
#define _ion_dikdv	*_ppvar[5]._pval
 
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_rates(void);
 static void _hoc_vtrap(void);
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
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_napp", _hoc_setdata,
 "rates_napp", _hoc_rates,
 "vtrap_napp", _hoc_vtrap,
 0, 0
};
#define vtrap vtrap_napp
 extern double vtrap( _threadargsprotocomma_ double , double );
 /* declare global and static user variables */
 static int _thread1data_inuse = 0;
static double _thread1data[10];
#define _gth 0
#define htau_napp _thread1data[0]
#define htau _thread[_gth]._pval[0]
#define hinf_napp _thread1data[1]
#define hinf _thread[_gth]._pval[1]
#define mtau_napp _thread1data[2]
#define mtau _thread[_gth]._pval[2]
#define minf_napp _thread1data[3]
#define minf _thread[_gth]._pval[3]
#define ntau_napp _thread1data[4]
#define ntau _thread[_gth]._pval[4]
#define ninf_napp _thread1data[5]
#define ninf _thread[_gth]._pval[5]
#define ptau_napp _thread1data[6]
#define ptau _thread[_gth]._pval[6]
#define pinf_napp _thread1data[7]
#define pinf _thread[_gth]._pval[7]
#define rtau_napp _thread1data[8]
#define rtau _thread[_gth]._pval[8]
#define rinf_napp _thread1data[9]
#define rinf _thread[_gth]._pval[9]
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gl_napp", 0, 1e+09,
 "gksbar_napp", 0, 1e+09,
 "gkfbar_napp", 0, 1e+09,
 "gnapbar_napp", 0, 1e+09,
 "gnabar_napp", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "mtau_napp", "ms",
 "htau_napp", "ms",
 "ptau_napp", "ms",
 "ntau_napp", "ms",
 "rtau_napp", "ms",
 "gnabar_napp", "S/cm2",
 "gnapbar_napp", "S/cm2",
 "gkfbar_napp", "S/cm2",
 "gksbar_napp", "S/cm2",
 "gl_napp", "S/cm2",
 "el_napp", "mV",
 "vtraub_napp", "mV",
 "mact_napp", "mV",
 "rinact_napp", "/ms",
 "m_alpha_v_offset_napp", "mV",
 "m_alpha_k_napp", "mV",
 "m_beta_v_offset_napp", "mV",
 "m_beta_k_napp", "mV",
 "h_alpha_v_offset_napp", "mV",
 "h_alpha_tau_napp", "mV",
 "h_beta_v_offset_napp", "mV",
 "h_beta_k_napp", "mV",
 "p_alpha_v_offset_napp", "mV",
 "p_alpha_k_napp", "mV",
 "p_beta_v_offset_napp", "mV",
 "p_beta_k_napp", "mV",
 "n_alpha_v_offset_napp", "mV",
 "n_alpha_k_napp", "mV",
 "n_beta_v_offset_napp", "mV",
 "n_beta_tau_napp", "mV",
 "r_alpha_v_offset_napp", "mV",
 "r_alpha_k_napp", "mV",
 "gna_napp", "S/cm2",
 "gnap_napp", "S/cm2",
 "gkf_napp", "S/cm2",
 "gks_napp", "S/cm2",
 "inap_napp", "mA/cm2",
 "inaf_napp", "mA/cm2",
 "ikf_napp", "mA/cm2",
 "iks_napp", "mA/cm2",
 "il_napp", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double n0 = 0;
 static double p0 = 0;
 static double r0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "minf_napp", &minf_napp,
 "hinf_napp", &hinf_napp,
 "pinf_napp", &pinf_napp,
 "ninf_napp", &ninf_napp,
 "rinf_napp", &rinf_napp,
 "mtau_napp", &mtau_napp,
 "htau_napp", &htau_napp,
 "ptau_napp", &ptau_napp,
 "ntau_napp", &ntau_napp,
 "rtau_napp", &rtau_napp,
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
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[6]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"napp",
 "gnabar_napp",
 "gnapbar_napp",
 "gkfbar_napp",
 "gksbar_napp",
 "gl_napp",
 "el_napp",
 "vtraub_napp",
 "mact_napp",
 "rinact_napp",
 "m_alpha_A_napp",
 "m_alpha_v_offset_napp",
 "m_alpha_k_napp",
 "m_beta_A_napp",
 "m_beta_v_offset_napp",
 "m_beta_k_napp",
 "h_alpha_A_napp",
 "h_alpha_v_offset_napp",
 "h_alpha_tau_napp",
 "h_beta_A_napp",
 "h_beta_v_offset_napp",
 "h_beta_k_napp",
 "p_alpha_A_napp",
 "p_alpha_v_offset_napp",
 "p_alpha_k_napp",
 "p_beta_A_napp",
 "p_beta_v_offset_napp",
 "p_beta_k_napp",
 "n_alpha_A_napp",
 "n_alpha_v_offset_napp",
 "n_alpha_k_napp",
 "n_beta_A_napp",
 "n_beta_v_offset_napp",
 "n_beta_tau_napp",
 "r_alpha_A_napp",
 "r_alpha_v_offset_napp",
 "r_alpha_k_napp",
 0,
 "gna_napp",
 "gnap_napp",
 "gkf_napp",
 "gks_napp",
 "inap_napp",
 "inaf_napp",
 "ikf_napp",
 "iks_napp",
 "il_napp",
 0,
 "m_napp",
 "h_napp",
 "p_napp",
 "n_napp",
 "r_napp",
 0,
 0};
 static Symbol* _na_sym;
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 61, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.03;
 	gnapbar = 3.3e-05;
 	gkfbar = 0.016;
 	gksbar = 0.004;
 	gl = 0.0003;
 	el = -54.3;
 	vtraub = 50;
 	mact = 15;
 	rinact = 0.05;
 	m_alpha_A = 0.64;
 	m_alpha_v_offset = 15;
 	m_alpha_k = 4;
 	m_beta_A = 0.56;
 	m_beta_v_offset = 40;
 	m_beta_k = 5;
 	h_alpha_A = 0.928;
 	h_alpha_v_offset = 17;
 	h_alpha_tau = 18;
 	h_beta_A = 9;
 	h_beta_v_offset = 40;
 	h_beta_k = 5;
 	p_alpha_A = 0.64;
 	p_alpha_v_offset = 5;
 	p_alpha_k = 4;
 	p_beta_A = 0.56;
 	p_beta_v_offset = 30;
 	p_beta_k = 5;
 	n_alpha_A = 0.08;
 	n_alpha_v_offset = 15;
 	n_alpha_k = 7;
 	n_beta_A = 2;
 	n_beta_v_offset = 10;
 	n_beta_tau = 40;
 	r_alpha_A = 3.5;
 	r_alpha_v_offset = 55;
 	r_alpha_k = 4;
 	_prop->param = _p;
 	_prop->param_size = 61;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[3]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[4]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[5]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _napp_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	ion_reg("k", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 2);
  _extcall_thread = (Datum*)ecalloc(1, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
  _thread1data_inuse = 0;
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 61, 7);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 napp /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/napp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "napp.mod   squid sodium, potassium, and leak channels";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[5], _dlist1[5];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   Dp = ( pinf - p ) / ptau ;
   Dn = ( ninf - n ) / ntau ;
   Dr = ( rinf - r ) / rtau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
 Dp = Dp  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ptau )) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ntau )) ;
 Dr = Dr  / (1. - dt*( ( ( ( - 1.0 ) ) ) / rtau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
    p = p + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ptau)))*(- ( ( ( pinf ) ) / ptau ) / ( ( ( ( - 1.0 ) ) ) / ptau ) - p) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ntau)))*(- ( ( ( ninf ) ) / ntau ) / ( ( ( ( - 1.0 ) ) ) / ntau ) - n) ;
    r = r + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / rtau)))*(- ( ( ( rinf ) ) / rtau ) / ( ( ( ( - 1.0 ) ) ) / rtau ) - r) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   double _lalpha , _lbeta , _lsum , _lv2 ;
  _lv2 = _lv - vtraub ;
   _lalpha = m_alpha_A * vtrap ( _threadargscomma_ m_alpha_v_offset - _lv2 , m_alpha_k ) ;
   _lbeta = m_beta_A * vtrap ( _threadargscomma_ _lv2 - m_beta_v_offset , m_beta_k ) ;
   _lsum = _lalpha + _lbeta ;
   mtau = 1.0 / _lsum ;
   minf = _lalpha / _lsum ;
   _lalpha = h_alpha_A * exp ( ( h_alpha_v_offset - _lv2 ) / h_alpha_tau ) ;
   _lbeta = h_beta_A / ( exp ( ( h_beta_v_offset - _lv2 ) / h_beta_k ) + 1.0 ) ;
   _lsum = _lalpha + _lbeta ;
   htau = 1.0 / _lsum ;
   hinf = _lalpha / _lsum ;
   _lalpha = p_alpha_A * vtrap ( _threadargscomma_ p_alpha_v_offset - _lv2 , p_alpha_k ) ;
   _lbeta = p_beta_A * vtrap ( _threadargscomma_ _lv2 - p_beta_v_offset , p_beta_k ) ;
   _lsum = _lalpha + _lbeta ;
   ptau = 1.0 / _lsum ;
   pinf = _lalpha / _lsum ;
   _lalpha = n_alpha_A * vtrap ( _threadargscomma_ n_alpha_v_offset - _lv2 , n_alpha_k ) ;
   _lbeta = n_beta_A * exp ( ( n_beta_v_offset - _lv2 ) / n_beta_tau ) ;
   _lsum = _lalpha + _lbeta ;
   ntau = 1.0 / _lsum ;
   ninf = _lalpha / _lsum ;
   _lalpha = ( r_alpha_A ) / ( exp ( ( r_alpha_v_offset - _lv2 ) / r_alpha_k ) + 1.0 ) ;
   _lbeta = rinact ;
   _lsum = _lalpha + _lbeta ;
   rtau = 1.0 / _lsum ;
   rinf = _lalpha / _lsum ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double vtrap ( _threadargsprotocomma_ double _lx , double _ly ) {
   double _lvtrap;
 if ( fabs ( _lx / _ly ) < 1e-6 ) {
     _lvtrap = _ly * ( 1.0 - _lx / _ly / 2.0 ) ;
     }
   else {
     _lvtrap = _lx / ( exp ( _lx / _ly ) - 1.0 ) ;
     }
   
return _lvtrap;
 }
 
static void _hoc_vtrap(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  vtrap ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 5;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
  ek = _ion_ek;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
   }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 5; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
  if (_thread1data_inuse) {_thread[_gth]._pval = (double*)ecalloc(10, sizeof(double));
 }else{
 _thread[_gth]._pval = _thread1data; _thread1data_inuse = 1;
 }
 }
 
static void _thread_cleanup(Datum* _thread) {
  if (_thread[_gth]._pval == _thread1data) {
   _thread1data_inuse = 0;
  }else{
   free((void*)_thread[_gth]._pval);
  }
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
   nrn_update_ion_pointer(_k_sym, _ppvar, 3, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 4, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 5, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
  n = n0;
  p = p0;
  r = r0;
 {
   rates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   p = pinf ;
   n = ninf ;
   r = rinf ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
  ena = _ion_ena;
  ek = _ion_ek;
 initmodel(_p, _ppvar, _thread, _nt);
  }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gna = gnabar * m * m * m * h ;
   gnap = gnapbar * p * p * p ;
   inaf = gna * ( v - ena ) ;
   inap = gnap * ( v - ena ) ;
   ina = inaf + inap ;
   gkf = gkfbar * n * n * n * n ;
   gks = gksbar * r * r ;
   ikf = ( gkf ) * ( v - ek ) ;
   iks = ( gks ) * ( v - ek ) ;
   ik = ikf + iks ;
   il = gl * ( v - el ) ;
   }
 _current += ina;
 _current += ik;
 _current += il;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
  ena = _ion_ena;
  ek = _ion_ek;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dik;
 double _dina;
  _dina = ina;
  _dik = ik;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
  ena = _ion_ena;
  ek = _ion_ek;
 {   states(_p, _ppvar, _thread, _nt);
  }  }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
 _slist1[2] = p_columnindex;  _dlist1[2] = Dp_columnindex;
 _slist1[3] = n_columnindex;  _dlist1[3] = Dn_columnindex;
 _slist1[4] = r_columnindex;  _dlist1[4] = Dr_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/napp.mod";
static const char* nmodl_file_text = 
  "\n"
  "TITLE napp.mod   squid sodium, potassium, and leak channels\n"
  "\n"
  "COMMENT\n"
  " This is the original Hodgkin-Huxley treatment for the set of sodium,\n"
  "  potassium, and leakage channels found in the squid giant axon membrane.\n"
  "  (\"A quantitative description of membrane current and its application\n"
  "  conduction and excitation in nerve\" J.Physiol. (Lond.) 117:500-544 (1952).)\n"
  " Membrane voltage is in absolute mV and has been reversed in polarity\n"
  "  from the original HH convention and shifted to reflect a resting potential\n"
  "  of -65 mV.\n"
  " Remember to set celsius=6.3 (or whatever) in your HOC file.\n"
  " See squid.hoc for an example of a simulation using this model.\n"
  " SW Jaslove  6 March, 1992\n"
  "ENDCOMMENT\n"
  "\n"
  "UNITS {\n"
  "        (mA) = (milliamp)\n"
  "        (mV) = (millivolt)\n"
  "	(S) = (siemens)\n"
  "}\n"
  "\n"
  "? interface\n"
  "NEURON {\n"
  "    SUFFIX napp\n"
  "    USEION na READ ena WRITE ina\n"
  "    USEION k READ ek WRITE ik\n"
  "    NONSPECIFIC_CURRENT il\n"
  "    RANGE gnabar,gnapbar, gkfbar, gksbar, gl, el, gna, gnap, gkf, gks, vtraub, mact, rinact, inap, inaf, ikf, iks, ek, ena\n"
  "    RANGE m_alpha_A, m_alpha_v_offset, m_alpha_k, m_beta_A, m_beta_v_offset, m_beta_k\n"
  "    RANGE h_alpha_A, h_alpha_v_offset, h_alpha_tau, h_beta_A, h_beta_v_offset, h_beta_k\n"
  "    RANGE p_alpha_A, p_alpha_v_offset, p_alpha_k, p_beta_A, p_beta_v_offset, p_beta_k\n"
  "    RANGE n_alpha_A, n_alpha_v_offset, n_alpha_k, n_beta_A, n_beta_v_offset, n_beta_tau\n"
  "    RANGE r_alpha_A, r_alpha_v_offset, r_alpha_k\n"
  "    GLOBAL minf, hinf, pinf, ninf, rinf, mtau, htau, ptau, ntau, rtau\n"
  "    THREADSAFE : assigned GLOBALs will be per thread\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    gnabar =            .030 (S/cm2) <0,1e9>\n"
  "    gnapbar =           .000033 (S/cm2) <0,1e9>\n"
  "    gkfbar =            .016 (S/cm2)	<0,1e9>\n"
  "    gksbar =            .004 (S/cm2)	<0,1e9>\n"
  "\n"
  "    gl =                .0003 (S/cm2)	<0,1e9>\n"
  "    el =                -54.3 (mV)\n"
  "    vtraub =            50.0 (mV)\n"
  "    mact =              15.0 (mV)\n"
  "    rinact =            0.05 (/ms)\n"
  "\n"
  "    : Alpha and beta parameters for m gate (sodium activation)\n"
  "    m_alpha_A =         0.64\n"
  "    m_alpha_v_offset =  15.0 (mV)\n"
  "    m_alpha_k =         4.0 (mV)\n"
  "    m_beta_A =          0.56\n"
  "    m_beta_v_offset =   40.0 (mV)\n"
  "    m_beta_k =          5.0 (mV)\n"
  "\n"
  "    : Alpha and beta parameters for h gate (sodium inactivation)\n"
  "    h_alpha_A =         0.928\n"
  "    h_alpha_v_offset =  17.0 (mV)\n"
  "    h_alpha_tau =       18.0 (mV)\n"
  "    h_beta_A =          9.0\n"
  "    h_beta_v_offset =   40.0 (mV)\n"
  "    h_beta_k =          5.0 (mV)\n"
  "\n"
  "    : Alpha and beta parameters for p gate (persistent sodium activation)\n"
  "    p_alpha_A =         0.64\n"
  "    p_alpha_v_offset =  5.0 (mV)\n"
  "    p_alpha_k =         4.0 (mV)\n"
  "    p_beta_A =          0.56\n"
  "    p_beta_v_offset =   30.0 (mV)\n"
  "    p_beta_k =          5.0 (mV)\n"
  "    : Alpha and beta parameters for n gate (fast potassium activation)\n"
  "    n_alpha_A =         0.08\n"
  "    n_alpha_v_offset =  15.0 (mV)\n"
  "    n_alpha_k =         7.0 (mV)\n"
  "    n_beta_A =          2.0\n"
  "    n_beta_v_offset =   10.0 (mV)\n"
  "    n_beta_tau =        40.0 (mV)\n"
  "\n"
  "    : Alpha and beta parameters for r gate (slow potassium activation)\n"
  "    r_alpha_A =         3.5\n"
  "    r_alpha_v_offset =  55.0 (mV)\n"
  "    r_alpha_k =         4.0 (mV)\n"
  "\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    m h p n r \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    v (mV)\n"
  "    celsius (degC)\n"
  "    \n"
  "    gna (S/cm2)\n"
  "    gnap (S/cm2)\n"
  "    gkf (S/cm2)\n"
  "    gks (S/cm2)\n"
  "    ena (mV)\n"
  "    ek (mV)\n"
  "    ina (mA/cm2)\n"
  "    inap (mA/cm2)\n"
  "    inaf (mA/cm2)\n"
  "    ik (mA/cm2)\n"
  "    ikf (mA/cm2)\n"
  "    iks (mA/cm2)\n"
  "    il (mA/cm2)\n"
  "    minf hinf pinf ninf rinf\n"
  "    mtau (ms) htau (ms) ptau (ms) ntau (ms) rtau (ms)\n"
  "}\n"
  "\n"
  "? currents\n"
  "BREAKPOINT {\n"
  "    SOLVE states METHOD cnexp\n"
  "    gna = gnabar*m*m*m*h\n"
  "    gnap = gnapbar*p*p*p\n"
  "    inaf = gna*(v-ena)\n"
  "    inap = gnap*(v-ena)\n"
  "    ina = inaf+inap\n"
  "    gkf = gkfbar*n*n*n*n\n"
  "    gks = gksbar*r*r\n"
  "    ikf = (gkf)*(v-ek)\n"
  "    iks = (gks)*(v-ek)\n"
  "    ik = ikf+iks\n"
  "    il = gl*(v - el)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    rates(v)\n"
  "    m = minf\n"
  "    h = hinf\n"
  "    p = pinf\n"
  "    n = ninf\n"
  "    r = rinf\n"
  "}\n"
  "\n"
  "? states\n"
  "DERIVATIVE states {\n"
  "    rates(v)\n"
  "    m' =  (minf-m)/mtau\n"
  "    h' = (hinf-h)/htau\n"
  "    p' = (pinf-p)/ptau\n"
  "    n' = (ninf-n)/ntau\n"
  "    r' = (rinf-r)/rtau\n"
  "}\n"
  "\n"
  "? rates\n"
  "PROCEDURE rates(v(mV)) {  \n"
  "    :Computes rate and other constants at current v.\n"
  "    :Call once from HOC to initialize inf at resting v.\n"
  "    LOCAL alpha, beta, sum, v2\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "    v2 = v - vtraub\n"
  "\n"
  "    :\"m\" sodium activation system\n"
  "    alpha = m_alpha_A * vtrap(m_alpha_v_offset-v2, m_alpha_k)\n"
  "    beta = m_beta_A * vtrap(v2-m_beta_v_offset, m_beta_k)\n"
  "    sum = alpha + beta\n"
  "    mtau = 1/sum\n"
  "    minf = alpha/sum\n"
  "    :\"h\" sodium inactivation system\n"
  "    alpha = h_alpha_A * exp((h_alpha_v_offset-v2)/h_alpha_tau)\n"
  "    beta = h_beta_A / (exp((h_beta_v_offset-v2)/h_beta_k) + 1)\n"
  "    sum = alpha + beta\n"
  "    htau = 1/sum\n"
  "    hinf = alpha/sum\n"
  "    :\"p\" sodium persistent activation system\n"
  "    alpha = p_alpha_A * vtrap(p_alpha_v_offset-v2, p_alpha_k)\n"
  "    beta = p_beta_A * vtrap(v2-p_beta_v_offset, p_beta_k)\n"
  "    sum = alpha + beta\n"
  "    ptau = 1/sum\n"
  "    pinf = alpha/sum\n"
  "    :\"n\" fast potassium activation system\n"
  "    alpha = n_alpha_A*vtrap(n_alpha_v_offset-v2, n_alpha_k)\n"
  "    beta = n_beta_A*exp((n_beta_v_offset-v2)/n_beta_tau)\n"
  "    sum = alpha + beta\n"
  "    ntau = 1/sum\n"
  "    ninf = alpha/sum\n"
  "    :\"r\" slow potassium activation system\n"
  "    alpha = (r_alpha_A)/(exp((r_alpha_v_offset-v2)/r_alpha_k) + 1)\n"
  "    beta = rinact\n"
  "    sum = alpha + beta\n"
  "    rtau = 1/sum\n"
  "    rinf = alpha/sum\n"
  "}\n"
  "\n"
  "FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.\n"
  "    if (fabs(x/y) < 1e-6) {\n"
  "        vtrap = y*(1 - x/y/2)\n"
  "    } else {\n"
  "        vtrap = x/(exp(x/y) - 1)\n"
  "    }\n"
  "}\n"
  "\n"
  "UNITSON\n"
  ;
#endif
