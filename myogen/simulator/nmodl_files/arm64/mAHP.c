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
 
#define nrn_init _nrn_init__mAHP
#define _nrn_initial _nrn_initial__mAHP
#define nrn_cur _nrn_cur__mAHP
#define _nrn_current _nrn_current__mAHP
#define nrn_jacob _nrn_jacob__mAHP
#define nrn_state _nrn_state__mAHP
#define _net_receive _net_receive__mAHP 
#define mcarate mcarate__mAHP 
#define rates rates__mAHP 
#define states states__mAHP 
 
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
#define gkcamax _p[0]
#define gkcamax_columnindex 0
#define gcamax _p[1]
#define gcamax_columnindex 1
#define depth _p[2]
#define depth_columnindex 2
#define tau _p[3]
#define tau_columnindex 3
#define ik _p[4]
#define ik_columnindex 4
#define ica _p[5]
#define ica_columnindex 5
#define mca _p[6]
#define mca_columnindex 6
#define n _p[7]
#define n_columnindex 7
#define cai _p[8]
#define cai_columnindex 8
#define ek _p[9]
#define ek_columnindex 9
#define eca _p[10]
#define eca_columnindex 10
#define ninf _p[11]
#define ninf_columnindex 11
#define ntau _p[12]
#define ntau_columnindex 12
#define minfca _p[13]
#define minfca_columnindex 13
#define drive_channel _p[14]
#define drive_channel_columnindex 14
#define Dmca _p[15]
#define Dmca_columnindex 15
#define Dn _p[16]
#define Dn_columnindex 16
#define Dcai _p[17]
#define Dcai_columnindex 17
#define v _p[18]
#define v_columnindex 18
#define _g _p[19]
#define _g_columnindex 19
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
#define _ion_eca	*_ppvar[3]._pval
#define _ion_ica	*_ppvar[4]._pval
#define _ion_dicadv	*_ppvar[5]._pval
 
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
 static void _hoc_mcarate(void);
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
 "setdata_mAHP", _hoc_setdata,
 "mcarate_mAHP", _hoc_mcarate,
 "rates_mAHP", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define bKCa bKCa_mAHP
 double bKCa = 0.1;
#define cainf cainf_mAHP
 double cainf = 0.0001;
#define caix caix_mAHP
 double caix = 2;
#define fKCa fKCa_mAHP
 double fKCa = 0.1;
#define mtauca mtauca_mAHP
 double mtauca = 1;
#define mslpca mslpca_mAHP
 double mslpca = 4;
#define mvhalfca mvhalfca_mAHP
 double mvhalfca = -30;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "mvhalfca_mAHP", "mV",
 "mslpca_mAHP", "mV",
 "mtauca_mAHP", "ms",
 "cainf_mAHP", "mM",
 "gkcamax_mAHP", "S/cm2",
 "gcamax_mAHP", "S/cm2",
 "depth_mAHP", "um",
 "tau_mAHP", "ms",
 "cai_mAHP", "mM",
 "ik_mAHP", "mA/cm2",
 "ica_mAHP", "mA/cm2",
 0,0
};
 static double cai0 = 0;
 static double delta_t = 0.01;
 static double mca0 = 0;
 static double n0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "mvhalfca_mAHP", &mvhalfca_mAHP,
 "mslpca_mAHP", &mslpca_mAHP,
 "mtauca_mAHP", &mtauca_mAHP,
 "caix_mAHP", &caix_mAHP,
 "cainf_mAHP", &cainf_mAHP,
 "fKCa_mAHP", &fKCa_mAHP,
 "bKCa_mAHP", &bKCa_mAHP,
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
"mAHP",
 "gkcamax_mAHP",
 "gcamax_mAHP",
 "depth_mAHP",
 "tau_mAHP",
 0,
 "ik_mAHP",
 "ica_mAHP",
 0,
 "mca_mAHP",
 "n_mAHP",
 "cai_mAHP",
 0,
 0};
 static Symbol* _k_sym;
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 20, _prop);
 	/*initialize range parameters*/
 	gkcamax = 0.03;
 	gcamax = 3e-05;
 	depth = 0.1;
 	tau = 20;
 	_prop->param = _p;
 	_prop->param_size = 20;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[3]._pval = &prop_ion->param[0]; /* eca */
 	_ppvar[4]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[5]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _mAHP_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	ion_reg("ca", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 20, 7);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 mAHP /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/mAHP.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.78e555060882cp+16, 96485.3}; /* 96485.3321233100141 */
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int mcarate(_threadargsprotocomma_ double);
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   drive_channel = - ( 10000.0 ) * ica / ( 2.0 * FARADAY * depth ) ;
   if ( drive_channel <= 0. ) {
     drive_channel = 0. ;
     }
   Dcai = drive_channel + ( cainf - cai ) / tau ;
   rates ( _threadargscomma_ cai ) ;
   Dn = ( ninf - n ) / ntau ;
   mcarate ( _threadargscomma_ v ) ;
   Dmca = ( minfca - mca ) / mtauca ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 drive_channel = - ( 10000.0 ) * ica / ( 2.0 * FARADAY * depth ) ;
 if ( drive_channel <= 0. ) {
   drive_channel = 0. ;
   }
 Dcai = Dcai  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau )) ;
 rates ( _threadargscomma_ cai ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ntau )) ;
 mcarate ( _threadargscomma_ v ) ;
 Dmca = Dmca  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtauca )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   drive_channel = - ( 10000.0 ) * ica / ( 2.0 * FARADAY * depth ) ;
   if ( drive_channel <= 0. ) {
     drive_channel = 0. ;
     }
    cai = cai + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau)))*(- ( drive_channel + ( ( cainf ) ) / tau ) / ( ( ( ( - 1.0 ) ) ) / tau ) - cai) ;
   rates ( _threadargscomma_ cai ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ntau)))*(- ( ( ( ninf ) ) / ntau ) / ( ( ( ( - 1.0 ) ) ) / ntau ) - n) ;
   mcarate ( _threadargscomma_ v ) ;
    mca = mca + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtauca)))*(- ( ( ( minfca ) ) / mtauca ) / ( ( ( ( - 1.0 ) ) ) / mtauca ) - mca) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lcai ) {
   double _la , _lb ;
  _la = fKCa * pow( ( 1e3 * ( _lcai - cainf ) ) , caix ) ;
   _lb = bKCa ;
   ntau = 1.0 / ( _la + _lb ) ;
   ninf = _la * ntau ;
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
 
static int  mcarate ( _threadargsprotocomma_ double _lv ) {
   minfca = 1.0 / ( 1.0 + exp ( - ( _lv - mvhalfca ) / mslpca ) ) ;
    return 0; }
 
static void _hoc_mcarate(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 mcarate ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
  eca = _ion_eca;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
   }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
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
  ek = _ion_ek;
  eca = _ion_eca;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 0);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 4, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 5, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  cai = cai0;
  mca = mca0;
  n = n0;
 {
   cai = cainf ;
   rates ( _threadargscomma_ cai ) ;
   mcarate ( _threadargscomma_ v ) ;
   n = ninf ;
   mca = minfca ;
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
  ek = _ion_ek;
  eca = _ion_eca;
 initmodel(_p, _ppvar, _thread, _nt);
  }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ica = gcamax * mca * ( v - eca ) ;
   ik = gkcamax * n * ( v - ek ) ;
   }
 _current += ik;
 _current += ica;

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
  ek = _ion_ek;
  eca = _ion_eca;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dica;
 double _dik;
  _dik = ik;
  _dica = ica;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dikdv += (_dik - ik)/.001 ;
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
  _ion_ica += ica ;
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
  ek = _ion_ek;
  eca = _ion_eca;
 {   states(_p, _ppvar, _thread, _nt);
  }  }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = cai_columnindex;  _dlist1[0] = Dcai_columnindex;
 _slist1[1] = n_columnindex;  _dlist1[1] = Dn_columnindex;
 _slist1[2] = mca_columnindex;  _dlist1[2] = Dmca_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/mAHP.mod";
static const char* nmodl_file_text = 
  "\n"
  " COMMENT\n"
  " \n"
  " mAHP.mod\n"
  " \n"
  " Calcium-dependent potassium channel responsible for mAHP in motoneurons\n"
  " Simplified calcium channel that provides Ca for the KCa conductance is included\n"
  " 	\n"
  " ENDCOMMENT\n"
  "\n"
  " NEURON {\n"
  " 	SUFFIX mAHP\n"
  " 	USEION k READ ek WRITE ik\n"
  " 	USEION ca READ eca WRITE ica\n"
  " 	RANGE n, gkcamax,gcamax,ik,cai,ica,depth,tau\n"
  " 	GLOBAL fKCa, bKCa, caix\n"
  " }\n"
  "\n"
  "\n"
  " UNITS {\n"
  " 	(mA) = (milliamp)\n"
  " 	(mV) = (millivolt)\n"
  " 	(S) = (siemens)\n"
  " 	(um) = (micron)\n"
  " 	(molar) = (1/liter)			: moles do not appear in units\n"
  " 	(mM)	= (millimolar)\n"
  " 	(msM)	= (ms mM)\n"
  " 	FARADAY = (faraday) (coulomb)\n"
  " } \n"
  " \n"
  " PARAMETER {\n"
  " 	gkcamax = 		0.03 (S/cm2)	\n"
  "	gcamax = 		3e-5 (S/cm2)\n"
  "	mvhalfca = 		-30	(mV)\n"
  "	mslpca = 		4 (mV)\n"
  "	mtauca =		1 (ms)	\n"
  " 	caix = 			2	\n"
  "  	cainf =			0.0001 (mM)\n"
  " 	depth = 		.1 (um)		: depth of shell\n"
  " 	tau	= 			20(ms)		: rate of calcium removal\n"
  "								\n"
  "  	fKCa = 			0.1			: max act rate  \n"
  " 	bKCa = 			0.1			: max deact rate \n"
  " \n"
  " 	celsius			(degC)\n"
  " } \n"
  " \n"
  " \n"
  " ASSIGNED {\n"
  " 	ik 			(mA/cm2)\n"
  " 	v 			(mV)\n"
  "	ica 		(mA/cm2)\n"
  " 	ek			(mV)\n"
  "	eca			(mV)\n"
  " 	ninf\n"
  " 	ntau 		(ms)\n"
  "	minfca	\n"
  "	drive_channel\n"
  " }\n"
  "  \n"
  " \n"
  " STATE {\n"
  " mca \n"
  " n \n"
  " cai (mM)\n"
  "}\n"
  " \n"
  " INITIAL { \n"
  "	cai = cainf\n"
  " 	rates(cai)\n"
  "	mcarate(v)\n"
  " 	n = ninf\n"
  "	mca = minfca\n"
  " }\n"
  " \n"
  " BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	ica = gcamax*mca*(v - eca)\n"
  " 	ik = gkcamax *n* (v - ek)\n"
  " } \n"
  " \n"
  "\n"
  "DERIVATIVE states { \n"
  " 	drive_channel = - (10000) * ica/ (2 * FARADAY * depth)\n"
  " 	if (drive_channel <= 0.) { drive_channel = 0. }	: cannot pump inward\n"
  " 	cai' = drive_channel + (cainf-cai)/tau\n"
  "\n"
  "	rates(cai)    \n"
  "	n' = (ninf-n)/ntau\n"
  "	mcarate(v)    \n"
  "	mca' = (minfca-mca)/mtauca\n"
  "}\n"
  "\n"
  "PROCEDURE rates(cai(mM)) {  \n"
  "	LOCAL a,b\n"
  "	UNITSOFF\n"
  "		a = fKCa * (1e3*(cai  -cainf))^caix		: rate constant depends on cai in uM\n"
  "		b = bKCa\n"
  "		ntau = 1/(a+b)\n"
  "		ninf = a*ntau\n"
  "	UNITSON\n"
  " }\n"
  "\n"
  "PROCEDURE mcarate(v (mV)) {\n"
  "	minfca = 1/(1+exp(-(v-mvhalfca)/mslpca))\n"
  "}\n"
  ;
#endif
