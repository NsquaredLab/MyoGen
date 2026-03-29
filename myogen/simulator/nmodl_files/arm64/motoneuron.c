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
 
#define nrn_init _nrn_init__motoneuron
#define _nrn_initial _nrn_initial__motoneuron
#define nrn_cur _nrn_cur__motoneuron
#define _nrn_current _nrn_current__motoneuron
#define nrn_jacob _nrn_jacob__motoneuron
#define nrn_state _nrn_state__motoneuron
#define _net_receive _net_receive__motoneuron 
#define rates rates__motoneuron 
#define states states__motoneuron 
 
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
#define gl _p[0]
#define gl_columnindex 0
#define gna _p[1]
#define gna_columnindex 1
#define gk_fast _p[2]
#define gk_fast_columnindex 2
#define gk_slow _p[3]
#define gk_slow_columnindex 3
#define el _p[4]
#define el_columnindex 4
#define vt _p[5]
#define vt_columnindex 5
#define il _p[6]
#define il_columnindex 6
#define m _p[7]
#define m_columnindex 7
#define h _p[8]
#define h_columnindex 8
#define n _p[9]
#define n_columnindex 9
#define p _p[10]
#define p_columnindex 10
#define ena _p[11]
#define ena_columnindex 11
#define eks _p[12]
#define eks_columnindex 12
#define ekf _p[13]
#define ekf_columnindex 13
#define ina _p[14]
#define ina_columnindex 14
#define iks _p[15]
#define iks_columnindex 15
#define ikf _p[16]
#define ikf_columnindex 16
#define Dm _p[17]
#define Dm_columnindex 17
#define Dh _p[18]
#define Dh_columnindex 18
#define Dn _p[19]
#define Dn_columnindex 19
#define Dp _p[20]
#define Dp_columnindex 20
#define v _p[21]
#define v_columnindex 21
#define _g _p[22]
#define _g_columnindex 22
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
#define _ion_eks	*_ppvar[3]._pval
#define _ion_iks	*_ppvar[4]._pval
#define _ion_diksdv	*_ppvar[5]._pval
#define _ion_ekf	*_ppvar[6]._pval
#define _ion_ikf	*_ppvar[7]._pval
#define _ion_dikfdv	*_ppvar[8]._pval
 
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
 /* declaration of user functions */
 static void _hoc_rates(void);
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
 "setdata_motoneuron", _hoc_setdata,
 "rates_motoneuron", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
 static int _thread1data_inuse = 0;
static double _thread1data[8];
#define _gth 0
#define alpha_n_motoneuron _thread1data[0]
#define alpha_n _thread[_gth]._pval[0]
#define alpha_h_motoneuron _thread1data[1]
#define alpha_h _thread[_gth]._pval[1]
#define alpha_m_motoneuron _thread1data[2]
#define alpha_m _thread[_gth]._pval[2]
#define beta_n_motoneuron _thread1data[3]
#define beta_n _thread[_gth]._pval[3]
#define beta_h_motoneuron _thread1data[4]
#define beta_h _thread[_gth]._pval[4]
#define beta_m_motoneuron _thread1data[5]
#define beta_m _thread[_gth]._pval[5]
#define ptau_motoneuron _thread1data[6]
#define ptau _thread[_gth]._pval[6]
#define pinf_motoneuron _thread1data[7]
#define pinf _thread[_gth]._pval[7]
#define tau_max_p tau_max_p_motoneuron
 double tau_max_p = 4;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gl_motoneuron", 0, 1e+09,
 "gk_slow_motoneuron", 0, 1e+09,
 "gk_fast_motoneuron", 0, 1e+09,
 "gna_motoneuron", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gl_motoneuron", "nS/um2",
 "gna_motoneuron", "nS/um2",
 "gk_fast_motoneuron", "nS/um2",
 "gk_slow_motoneuron", "nS/um2",
 "el_motoneuron", "mV",
 "il_motoneuron", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double n0 = 0;
 static double p0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tau_max_p_motoneuron", &tau_max_p_motoneuron,
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
 
#define _cvode_ieq _ppvar[9]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"motoneuron",
 "gl_motoneuron",
 "gna_motoneuron",
 "gk_fast_motoneuron",
 "gk_slow_motoneuron",
 "el_motoneuron",
 "vt_motoneuron",
 0,
 "il_motoneuron",
 0,
 "m_motoneuron",
 "h_motoneuron",
 "n_motoneuron",
 "p_motoneuron",
 0,
 0};
 static Symbol* _na_sym;
 static Symbol* _ks_sym;
 static Symbol* _kf_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 23, _prop);
 	/*initialize range parameters*/
 	gl = 0.0003;
 	gna = 0.0003;
 	gk_fast = 0.0003;
 	gk_slow = 0.0003;
 	el = -70;
 	vt = -58;
 	_prop->param = _p;
 	_prop->param_size = 23;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 10, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 prop_ion = need_memb(_ks_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[3]._pval = &prop_ion->param[0]; /* eks */
 	_ppvar[4]._pval = &prop_ion->param[3]; /* iks */
 	_ppvar[5]._pval = &prop_ion->param[4]; /* _ion_diksdv */
 prop_ion = need_memb(_kf_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[6]._pval = &prop_ion->param[0]; /* ekf */
 	_ppvar[7]._pval = &prop_ion->param[3]; /* ikf */
 	_ppvar[8]._pval = &prop_ion->param[4]; /* _ion_dikfdv */
 
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

 void _motoneuron_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	ion_reg("ks", -10000.);
 	ion_reg("kf", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	_ks_sym = hoc_lookup("ks_ion");
 	_kf_sym = hoc_lookup("kf_ion");
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
  hoc_register_prop_size(_mechtype, 23, 10);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ks_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "ks_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "ks_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "kf_ion");
  hoc_register_dparam_semantics(_mechtype, 7, "kf_ion");
  hoc_register_dparam_semantics(_mechtype, 8, "kf_ion");
  hoc_register_dparam_semantics(_mechtype, 9, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 motoneuron /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/motoneuron.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[4], _dlist1[4];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm = alpha_m * ( 1.0 - m ) - beta_m * m ;
   Dh = 0.1 * alpha_h * ( 1.0 - h ) - 0.1 * beta_h * h ;
   Dn = 0.1 * alpha_n * ( 1.0 - n ) - 0.1 * beta_n * n ;
   Dp = ( pinf - p ) / ptau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( alpha_m )*( ( ( - 1.0 ) ) ) - ( beta_m )*( 1.0 ) )) ;
 Dh = Dh  / (1. - dt*( ( 0.1 * alpha_h )*( ( ( - 1.0 ) ) ) - ( 0.1 * beta_h )*( 1.0 ) )) ;
 Dn = Dn  / (1. - dt*( ( 0.1 * alpha_n )*( ( ( - 1.0 ) ) ) - ( 0.1 * beta_n )*( 1.0 ) )) ;
 Dp = Dp  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ptau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( alpha_m )*( ( ( - 1.0 ) ) ) - ( beta_m )*( 1.0 ))))*(- ( ( alpha_m )*( ( 1.0 ) ) ) / ( ( alpha_m )*( ( ( - 1.0 ) ) ) - ( beta_m )*( 1.0 ) ) - m) ;
    h = h + (1. - exp(dt*(( 0.1 * alpha_h )*( ( ( - 1.0 ) ) ) - ( 0.1 * beta_h )*( 1.0 ))))*(- ( ( ( 0.1 )*( alpha_h ) )*( ( 1.0 ) ) ) / ( ( ( 0.1 )*( alpha_h ) )*( ( ( - 1.0 ) ) ) - ( ( 0.1 )*( beta_h ) )*( 1.0 ) ) - h) ;
    n = n + (1. - exp(dt*(( 0.1 * alpha_n )*( ( ( - 1.0 ) ) ) - ( 0.1 * beta_n )*( 1.0 ))))*(- ( ( ( 0.1 )*( alpha_n ) )*( ( 1.0 ) ) ) / ( ( ( 0.1 )*( alpha_n ) )*( ( ( - 1.0 ) ) ) - ( ( 0.1 )*( beta_n ) )*( 1.0 ) ) - n) ;
    p = p + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ptau)))*(- ( ( ( pinf ) ) / ptau ) / ( ( ( ( - 1.0 ) ) ) / ptau ) - p) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   alpha_m = ( - 0.32 * ( _lv - vt - 13.0 ) ) / ( exp ( - ( _lv - vt - 13.0 ) / 4.0 ) - 1.0 ) ;
   beta_m = 0.28 * ( _lv - vt - 40.0 ) / ( exp ( ( _lv - vt - 40.0 ) / 5.0 ) - 1.0 ) ;
   alpha_h = 0.128 * exp ( - ( _lv - vt - 17.0 ) / 18.0 ) ;
   beta_h = 4.0 / ( 1.0 + exp ( - ( _lv - vt - 40.0 ) / 5.0 ) ) ;
   alpha_n = ( - 0.032 * ( _lv - vt - 15.0 ) ) / ( exp ( - ( _lv - vt - 15.0 ) / 5.0 ) - 1.0 ) ;
   beta_n = 0.5 * exp ( - ( _lv - vt - 10.0 ) / 40.0 ) ;
   pinf = 1.0 / ( 1.0 + exp ( - ( _lv + 35.0 ) / 10.0 ) ) ;
   ptau = tau_max_p / ( 3.3 * exp ( ( _lv + 35.0 ) / 20.0 ) + exp ( - ( _lv + 35.0 ) / 20.0 ) ) ;
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
 
static int _ode_count(int _type){ return 4;}
 
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
  eks = _ion_eks;
  ekf = _ion_ekf;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
    }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 4; ++_i) {
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
  eks = _ion_eks;
  ekf = _ion_ekf;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
  if (_thread1data_inuse) {_thread[_gth]._pval = (double*)ecalloc(8, sizeof(double));
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
   nrn_update_ion_pointer(_ks_sym, _ppvar, 3, 0);
   nrn_update_ion_pointer(_ks_sym, _ppvar, 4, 3);
   nrn_update_ion_pointer(_ks_sym, _ppvar, 5, 4);
   nrn_update_ion_pointer(_kf_sym, _ppvar, 6, 0);
   nrn_update_ion_pointer(_kf_sym, _ppvar, 7, 3);
   nrn_update_ion_pointer(_kf_sym, _ppvar, 8, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
  n = n0;
  p = p0;
 {
   rates ( _threadargscomma_ v ) ;
   m = 0.0 ;
   h = 1.0 ;
   n = 0.0 ;
   p = pinf ;
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
  eks = _ion_eks;
  ekf = _ion_ekf;
 initmodel(_p, _ppvar, _thread, _nt);
   }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ina = gna * m * m * m * h * ( v - ena ) ;
   ikf = gk_fast * n * n * n * n * ( v - ekf ) ;
   iks = gk_slow * p * p * ( v - eks ) ;
   il = gl * ( v - el ) ;
   }
 _current += ina;
 _current += iks;
 _current += ikf;
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
  eks = _ion_eks;
  ekf = _ion_ekf;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dikf;
 double _diks;
 double _dina;
  _dina = ina;
  _diks = iks;
  _dikf = ikf;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
  _ion_diksdv += (_diks - iks)/.001 ;
  _ion_dikfdv += (_dikf - ikf)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
  _ion_iks += iks ;
  _ion_ikf += ikf ;
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
  eks = _ion_eks;
  ekf = _ion_ekf;
 {   states(_p, _ppvar, _thread, _nt);
  }   }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
 _slist1[2] = n_columnindex;  _dlist1[2] = Dn_columnindex;
 _slist1[3] = p_columnindex;  _dlist1[3] = Dp_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/motoneuron.mod";
static const char* nmodl_file_text = 
  ": Motoneuron model implementing Hodgkin-Huxley type dynamics\n"
  ": This model includes:\n"
  ": - Sodium (Na+) channels with activation (m) and inactivation (h) gates\n"
  ": - Fast potassium (K+) channels with activation gate (n)\n"
  ": - Slow potassium (K+) channels with activation gate (p)\n"
  ": - Leak current\n"
  ": The model uses standard Hodgkin-Huxley formulations for ion channels\n"
  ": with voltage-dependent rate constants and conductances.\n"
  ":\n"
  ": Documentation:\n"
  ": - USEION syntax: https://nrn.readthedocs.io/en/8.2.6/guide/faq.html#what-units-does-neuron-use-for-current-concentration-etc\n"
  ": - Units: https://nrn.readthedocs.io/en/8.2.6/guide/units.html\n"
  ": - State variables: https://nrn.readthedocs.io/en/8.2.6/guide/faq.html#how-do-i-create-a-neuron-model\n"
  ": - Rate constants: https://nrn.readthedocs.io/en/8.2.6/guide/faq.html#is-there-a-list-of-functions-that-are-built-into-nmodl\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX motoneuron\n"
  "    : Sodium ion channel\n"
  "    : READ ena: reads the sodium reversal potential from the extracellular space\n"
  "    : WRITE ina: writes the computed sodium current back to the extracellular space\n"
  "    USEION na READ ena WRITE ina\n"
  "    : Slow potassium ion channel\n"
  "    : READ eks: reads the slow potassium reversal potential from the extracellular space\n"
  "    : WRITE iks: writes the computed slow potassium current back to the extracellular space\n"
  "    USEION ks READ eks WRITE iks\n"
  "    : Fast potassium ion channel\n"
  "    : READ ekf: reads the fast potassium reversal potential from the extracellular space\n"
  "    : WRITE ikf: writes the computed fast potassium current back to the extracellular space\n"
  "    USEION kf READ ekf WRITE ikf\n"
  "    : Leak current\n"
  "    NONSPECIFIC_CURRENT il\n"
  "    : Channel conductances and parameters\n"
  "    RANGE gna, gk_fast, gk_slow, gl, vt, el\n"
  "    : Rate constants\n"
  "    GLOBAL alpha_m, alpha_h, alpha_n, pinf, beta_m, beta_h, beta_n, ptau\n"
  "    : Allows parallel execution\n"
  "    THREADSAFE\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    : Current unit\n"
  "    (mA) = (milliamp)\n"
  "    : Voltage unit\n"
  "    (mV) = (millivolt)\n"
  "    : Conductance unit\n"
  "    (S) = (siemens)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    : Leak conductance\n"
  "    gl = 0.0003 (nS/um2) <0,1e9>\n"
  "    : Sodium channel conductance\n"
  "    gna = 0.0003 (nS/um2) <0,1e9>\n"
  "    : Fast potassium channel conductance\n"
  "    gk_fast = 0.0003 (nS/um2) <0,1e9>\n"
  "    : Slow potassium channel conductance\n"
  "    gk_slow = 0.0003 (nS/um2) <0,1e9>\n"
  "    : Leak reversal potential\n"
  "    el = -70 (mV)\n"
  "    : Maximum time constant for slow K+ channel\n"
  "    tau_max_p = 4\n"
  "    : Voltage threshold for activation\n"
  "    vt = -58\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    : Membrane potential\n"
  "    v (mV)\n"
  "    : Sodium reversal potential\n"
  "    ena (mV)\n"
  "    : Slow potassium reversal potential\n"
  "    eks (mV)\n"
  "    : Fast potassium reversal potential\n"
  "    ekf (mV)\n"
  "    : Sodium current\n"
  "    ina (mA/cm2)\n"
  "    : Slow potassium current\n"
  "    iks (mA/cm2)\n"
  "    : Fast potassium current\n"
  "    ikf (mA/cm2)\n"
  "    : Leak current\n"
  "    il (mA/cm2)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    : m,h: Na+ channel gates, n: fast K+ gate, p: slow K+ gate\n"
  "    m h n p\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    rates(v)\n"
  "    : Na+ activation gate starts closed\n"
  "    m = 0\n"
  "    : Na+ inactivation gate starts open\n"
  "    h = 1\n"
  "    : Fast K+ gate starts closed\n"
  "    n = 0\n"
  "    : Slow K+ gate starts at steady state\n"
  "    p = pinf\n"
  "}\n"
  "\n"
  "? currents\n"
  "BREAKPOINT {\n"
  "    : Solve differential equations\n"
  "    SOLVE states METHOD cnexp\n"
  "    : Sodium current (m^3*h formulation)\n"
  "    ina = gna*m*m*m*h*(v - ena)\n"
  "    : Fast potassium current (n^4 formulation)\n"
  "    ikf = gk_fast*n*n*n*n*(v - ekf)\n"
  "    : Slow potassium current (p^2 formulation)\n"
  "    iks = gk_slow*p*p*(v - eks)\n"
  "    : Leak current\n"
  "    il = gl*(v - el)\n"
  "    :printf(\"ina = %g\", ina)\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "    rates(v)\n"
  "    : Na+ activation gate\n"
  "    m' = alpha_m*(1-m) - beta_m*m\n"
  "    : Na+ inactivation gate\n"
  "    h' = 0.1*alpha_h*(1-h) - 0.1*beta_h*h\n"
  "    : Fast K+ gate\n"
  "    n' = 0.1*alpha_n*(1-n) - 0.1*beta_n*n\n"
  "    : Slow K+ gate\n"
  "    p' = (pinf - p) / ptau\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {\n"
  "    : Na+ activation forward rate\n"
  "    alpha_m = (-0.32*(v-vt-13))/(exp(-(v-vt-13)/4)-1)\n"
  "    : Na+ activation backward rate\n"
  "    beta_m = 0.28*(v-vt-40)/(exp((v-vt-40)/5)-1)\n"
  "    : Na+ inactivation forward rate\n"
  "    alpha_h = 0.128*exp(-(v-vt-17)/18)\n"
  "    : Na+ inactivation backward rate\n"
  "    beta_h = 4/(1+exp(-(v-vt-40)/5))\n"
  "\n"
  "    : Fast K+ activation forward rate\n"
  "    alpha_n = (-0.032*(v-vt-15))/(exp(-(v-vt-15)/5)-1)\n"
  "    : Fast K+ activation backward rate\n"
  "    beta_n = 0.5*exp(-(v-vt-10)/40)\n"
  "\n"
  "    : Slow K+ steady-state activation\n"
  "    pinf = 1/(1+exp(-(v+35)/10))\n"
  "    : Slow K+ time constant\n"
  "    ptau = tau_max_p/(3.3*exp((v+35)/20)+exp(-(v+35)/20))\n"
  "}\n"
  ;
#endif
