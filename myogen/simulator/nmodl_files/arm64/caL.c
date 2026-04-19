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
 
#define nrn_init _nrn_init__caL
#define _nrn_initial _nrn_initial__caL
#define nrn_cur _nrn_cur__caL
#define _nrn_current _nrn_current__caL
#define nrn_jacob _nrn_jacob__caL
#define nrn_state _nrn_state__caL
#define _net_receive _net_receive__caL 
#define rates rates__caL 
#define states states__caL 
 
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
#define gcaLbar _p[0]
#define gcaLbar_columnindex 0
#define gl _p[1]
#define gl_columnindex 1
#define el _p[2]
#define el_columnindex 2
#define vtraub _p[3]
#define vtraub_columnindex 3
#define gama _p[4]
#define gama_columnindex 4
#define Ltau _p[5]
#define Ltau_columnindex 5
#define gcaL _p[6]
#define gcaL_columnindex 6
#define icaL _p[7]
#define icaL_columnindex 7
#define il _p[8]
#define il_columnindex 8
#define L _p[9]
#define L_columnindex 9
#define DL _p[10]
#define DL_columnindex 10
#define ecaL _p[11]
#define ecaL_columnindex 11
#define v _p[12]
#define v_columnindex 12
#define _g _p[13]
#define _g_columnindex 13
#define _ion_ecaL	*_ppvar[0]._pval
#define _ion_icaL	*_ppvar[1]._pval
#define _ion_dicaLdv	*_ppvar[2]._pval
 
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
 "setdata_caL", _hoc_setdata,
 "rates_caL", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
 static int _thread1data_inuse = 0;
static double _thread1data[1];
#define _gth 0
#define Linf_caL _thread1data[0]
#define Linf _thread[_gth]._pval[0]
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gcaLbar_caL", 0, 1e+09,
 "gl_caL", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gcaLbar_caL", "S/cm2",
 "gl_caL", "S/cm2",
 "el_caL", "mV",
 "vtraub_caL", "mV",
 "Ltau_caL", "ms",
 "gcaL_caL", "S/cm2",
 "icaL_caL", "mA/cm2",
 "il_caL", "mA/cm2",
 0,0
};
 static double L0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Linf_caL", &Linf_caL,
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
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"caL",
 "gcaLbar_caL",
 "gl_caL",
 "el_caL",
 "vtraub_caL",
 "gama_caL",
 "Ltau_caL",
 0,
 "gcaL_caL",
 "icaL_caL",
 "il_caL",
 0,
 "L_caL",
 0,
 0};
 static Symbol* _caL_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 14, _prop);
 	/*initialize range parameters*/
 	gcaLbar = 0.03;
 	gl = 0.0003;
 	el = -54.3;
 	vtraub = 70;
 	gama = 1;
 	Ltau = 20;
 	_prop->param = _p;
 	_prop->param_size = 14;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_caL_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ecaL */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* icaL */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dicaLdv */
 
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

 void _caL_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("caL", 2.0);
 	_caL_sym = hoc_lookup("caL_ion");
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
  hoc_register_prop_size(_mechtype, 14, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "caL_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "caL_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "caL_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 caL /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/caL.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "caL.mod   ";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   DL = ( Linf - L ) / Ltau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 DL = DL  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Ltau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    L = L + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Ltau)))*(- ( ( ( Linf ) ) / Ltau ) / ( ( ( ( - 1.0 ) ) ) / Ltau ) - L) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   double _lv2 ;
  _lv2 = _lv - vtraub ;
   Linf = 1.0 / ( exp ( ( _lv2 + 30.0 ) / ( - 1.0 ) ) + 1.0 ) ;
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
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ecaL = _ion_ecaL;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
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
  ecaL = _ion_ecaL;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
  if (_thread1data_inuse) {_thread[_gth]._pval = (double*)ecalloc(1, sizeof(double));
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
   nrn_update_ion_pointer(_caL_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_caL_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_caL_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  L = L0;
 {
   rates ( _threadargscomma_ v ) ;
   L = Linf ;
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
  ecaL = _ion_ecaL;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gcaL = gcaLbar * L ;
   icaL = gcaL * gama * ( v - ecaL ) ;
   il = gl * ( v - el ) ;
   }
 _current += icaL;
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
  ecaL = _ion_ecaL;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dicaL;
  _dicaL = icaL;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicaLdv += (_dicaL - icaL)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_icaL += icaL ;
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
  ecaL = _ion_ecaL;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = L_columnindex;  _dlist1[0] = DL_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/caL.mod";
static const char* nmodl_file_text = 
  "TITLE caL.mod   \n"
  "\n"
  "COMMENT\n"
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
  "        SUFFIX caL\n"
  "	USEION caL READ ecaL WRITE icaL VALENCE 2\n"
  "	NONSPECIFIC_CURRENT il\n"
  "        RANGE  gl, el, gcaL, gcaLbar, vtraub, icaL, gama, Ltau, ecaL\n"
  "        GLOBAL Linf\n"
  "	THREADSAFE : assigned GLOBALs will be per thread\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "        gcaLbar = .030 (S/cm2)	<0,1e9>\n"
  "	gl = .0003 (S/cm2)	<0,1e9>\n"
  "        el = -54.3 (mV)\n"
  "\n"
  "	vtraub = 70.0 (mV)\n"
  "	gama = 1\n"
  "	Ltau = 20 (ms)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "        L \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "        v (mV)\n"
  "        celsius (degC)\n"
  "        ecaL (mV)\n"
  "	\n"
  "    	gcaL (S/cm2)\n"
  "        icaL (mA/cm2)\n"
  "        il (mA/cm2)\n"
  "        Linf\n"
  "	\n"
  "}\n"
  "\n"
  "? currents\n"
  "BREAKPOINT {\n"
  "        SOLVE states METHOD cnexp\n"
  "        gcaL = gcaLbar*L\n"
  "	icaL = gcaL*gama*(v-ecaL)\n"
  "        il = gl*(v - el)\n"
  "}\n"
  "\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	L = Linf\n"
  "	\n"
  "}\n"
  "\n"
  "? states\n"
  "DERIVATIVE states {\n"
  "        rates(v)\n"
  "        L' =  (Linf-L)/Ltau\n"
  "}\n"
  "\n"
  "? rates\n"
  "PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.\n"
  "                      :Call once from HOC to initialize inf at resting v.\n"
  "        LOCAL v2\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "	v2 = v - vtraub\n"
  "\n"
  "                :\"L\" calcium activation system\n"
  "   \n"
  "\n"
  "        Linf = 1/(exp((v2+30)/(-1))+1)\n"
  "}\n"
  "\n"
  "\n"
  "UNITSON\n"
  ;
#endif
