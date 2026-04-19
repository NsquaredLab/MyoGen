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
 
#define nrn_init _nrn_init__na3rp
#define _nrn_initial _nrn_initial__na3rp
#define nrn_cur _nrn_cur__na3rp
#define _nrn_current _nrn_current__na3rp
#define nrn_jacob _nrn_jacob__na3rp
#define nrn_state _nrn_state__na3rp
#define _net_receive _net_receive__na3rp 
#define states states__na3rp 
#define trates trates__na3rp 
 
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
#define sh _p[0]
#define sh_columnindex 0
#define gbar _p[1]
#define gbar_columnindex 1
#define thinf _p[2]
#define thinf_columnindex 2
#define qinf _p[3]
#define qinf_columnindex 3
#define ar _p[4]
#define ar_columnindex 4
#define ina _p[5]
#define ina_columnindex 5
#define minf _p[6]
#define minf_columnindex 6
#define hinf _p[7]
#define hinf_columnindex 7
#define mtau _p[8]
#define mtau_columnindex 8
#define htau _p[9]
#define htau_columnindex 9
#define sinf _p[10]
#define sinf_columnindex 10
#define taus _p[11]
#define taus_columnindex 11
#define m _p[12]
#define m_columnindex 12
#define h _p[13]
#define h_columnindex 13
#define s _p[14]
#define s_columnindex 14
#define ena _p[15]
#define ena_columnindex 15
#define thegna _p[16]
#define thegna_columnindex 16
#define Dm _p[17]
#define Dm_columnindex 17
#define Dh _p[18]
#define Dh_columnindex 18
#define Ds _p[19]
#define Ds_columnindex 19
#define v _p[20]
#define v_columnindex 20
#define _g _p[21]
#define _g_columnindex 21
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
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
 static void _hoc_alps(void);
 static void _hoc_bets(void);
 static void _hoc_trap0(void);
 static void _hoc_trates(void);
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
 "setdata_na3rp", _hoc_setdata,
 "alps_na3rp", _hoc_alps,
 "bets_na3rp", _hoc_bets,
 "trap0_na3rp", _hoc_trap0,
 "trates_na3rp", _hoc_trates,
 0, 0
};
#define alps alps_na3rp
#define bets bets_na3rp
#define trap0 trap0_na3rp
 extern double alps( _threadargsprotocomma_ double );
 extern double bets( _threadargsprotocomma_ double );
 extern double trap0( _threadargsprotocomma_ double , double , double , double );
 #define _zmexp _thread[0]._pval[0]
 #define _zhexp _thread[0]._pval[1]
 #define _zsexp _thread[0]._pval[2]
 /* declare global and static user variables */
#define Rd Rd_na3rp
 double Rd = 0.03;
#define Rg Rg_na3rp
 double Rg = 0.01;
#define Rb Rb_na3rp
 double Rb = 0.124;
#define Ra Ra_na3rp
 double Ra = 0.4;
#define avs avs_na3rp
 double avs = 30;
#define asvh asvh_na3rp
 double asvh = -85;
#define a0s a0s_na3rp
 double a0s = 0.001;
#define bvs bvs_na3rp
 double bvs = 10;
#define bsvh bsvh_na3rp
 double bsvh = -17;
#define b0s b0s_na3rp
 double b0s = 0.0034;
#define hmin hmin_na3rp
 double hmin = 0.5;
#define mmin mmin_na3rp
 double mmin = 0.02;
#define qq qq_na3rp
 double qq = 10;
#define q10 q10_na3rp
 double q10 = 2;
#define qg qg_na3rp
 double qg = 1.5;
#define qd qd_na3rp
 double qd = 1.5;
#define qa qa_na3rp
 double qa = 7.2;
#define tq tq_na3rp
 double tq = -55;
#define thi2 thi2_na3rp
 double thi2 = -45;
#define thi1 thi1_na3rp
 double thi1 = -45;
#define tha tha_na3rp
 double tha = -30;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tha_na3rp", "mV",
 "qa_na3rp", "mV",
 "Ra_na3rp", "/ms",
 "Rb_na3rp", "/ms",
 "thi1_na3rp", "mV",
 "thi2_na3rp", "mV",
 "qd_na3rp", "mV",
 "qg_na3rp", "mV",
 "Rg_na3rp", "/ms",
 "Rd_na3rp", "/ms",
 "qq_na3rp", "mV",
 "tq_na3rp", "mV",
 "a0s_na3rp", "/ms",
 "b0s_na3rp", "/ms",
 "asvh_na3rp", "mV",
 "bsvh_na3rp", "mV",
 "avs_na3rp", "mV",
 "bvs_na3rp", "mV",
 "sh_na3rp", "mV",
 "gbar_na3rp", "mho/cm2",
 "thinf_na3rp", "mV",
 "qinf_na3rp", "mV",
 "ar_na3rp", "1",
 "ina_na3rp", "mA/cm2",
 "mtau_na3rp", "ms",
 "htau_na3rp", "ms",
 "sinf_na3rp", "ms",
 "taus_na3rp", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double s0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tha_na3rp", &tha_na3rp,
 "qa_na3rp", &qa_na3rp,
 "Ra_na3rp", &Ra_na3rp,
 "Rb_na3rp", &Rb_na3rp,
 "thi1_na3rp", &thi1_na3rp,
 "thi2_na3rp", &thi2_na3rp,
 "qd_na3rp", &qd_na3rp,
 "qg_na3rp", &qg_na3rp,
 "mmin_na3rp", &mmin_na3rp,
 "hmin_na3rp", &hmin_na3rp,
 "q10_na3rp", &q10_na3rp,
 "Rg_na3rp", &Rg_na3rp,
 "Rd_na3rp", &Rd_na3rp,
 "qq_na3rp", &qq_na3rp,
 "tq_na3rp", &tq_na3rp,
 "a0s_na3rp", &a0s_na3rp,
 "b0s_na3rp", &b0s_na3rp,
 "asvh_na3rp", &asvh_na3rp,
 "bsvh_na3rp", &bsvh_na3rp,
 "avs_na3rp", &avs_na3rp,
 "bvs_na3rp", &bvs_na3rp,
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
"na3rp",
 "sh_na3rp",
 "gbar_na3rp",
 "thinf_na3rp",
 "qinf_na3rp",
 "ar_na3rp",
 0,
 "ina_na3rp",
 "minf_na3rp",
 "hinf_na3rp",
 "mtau_na3rp",
 "htau_na3rp",
 "sinf_na3rp",
 "taus_na3rp",
 0,
 "m_na3rp",
 "h_na3rp",
 "s_na3rp",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 22, _prop);
 	/*initialize range parameters*/
 	sh = 8;
 	gbar = 0.01;
 	thinf = -50;
 	qinf = 4;
 	ar = 1;
 	_prop->param = _p;
 	_prop->param_size = 22;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
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

 void _na3rp_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 2);
  _extcall_thread = (Datum*)ecalloc(1, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 22, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 na3rp /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/na3rp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 /*Top LOCAL _zmexp , _zhexp , _zsexp */
static int _reset;
static char *modelname = "na3rp";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int trates(_threadargsprotocomma_ double, double, double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
double alps ( _threadargsprotocomma_ double _lv ) {
   double _lalps;
 _lalps = a0s * exp ( ( asvh - _lv ) / avs ) ;
   
return _lalps;
 }
 
static void _hoc_alps(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alps ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double bets ( _threadargsprotocomma_ double _lv ) {
   double _lbets;
 _lbets = b0s / ( exp ( ( bsvh - _lv ) / bvs ) + 1.0 ) ;
   
return _lbets;
 }
 
static void _hoc_bets(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  bets ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   trates ( _threadargscomma_ v , ar , sh ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   Ds = ( sinf - s ) / taus ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 trates ( _threadargscomma_ v , ar , sh ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
 Ds = Ds  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taus )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   trates ( _threadargscomma_ v , ar , sh ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
    s = s + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taus)))*(- ( ( ( sinf ) ) / taus ) / ( ( ( ( - 1.0 ) ) ) / taus ) - s) ;
   }
  return 0;
}
 
static int  trates ( _threadargsprotocomma_ double _lvm , double _la2 , double _lsh2 ) {
   double _la , _lb , _lc , _lqt ;
 _lqt = pow( q10 , ( ( celsius - 24.0 ) / 10.0 ) ) ;
   _la = trap0 ( _threadargscomma_ _lvm , tha + _lsh2 , Ra , qa ) ;
   _lb = trap0 ( _threadargscomma_ - _lvm , - tha - _lsh2 , Rb , qa ) ;
   mtau = 1.0 / ( _la + _lb ) / _lqt ;
   if ( mtau < mmin ) {
     mtau = mmin ;
     }
   minf = _la / ( _la + _lb ) ;
   _la = trap0 ( _threadargscomma_ _lvm , thi1 + _lsh2 , Rd , qd ) ;
   _lb = trap0 ( _threadargscomma_ - _lvm , - thi2 - _lsh2 , Rg , qg ) ;
   htau = 1.0 / ( _la + _lb ) / _lqt ;
   if ( htau < hmin ) {
     htau = hmin ;
     }
   hinf = 1.0 / ( 1.0 + exp ( ( _lvm - thinf - _lsh2 ) / qinf ) ) ;
   taus = 1.0 / ( alps ( _threadargscomma_ _lvm ) + bets ( _threadargscomma_ _lvm ) ) ;
   _lc = alps ( _threadargscomma_ _lvm ) * taus ;
   sinf = _lc + _la2 * ( 1.0 - _lc ) ;
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 trates ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) );
 hoc_retpushx(_r);
}
 
double trap0 ( _threadargsprotocomma_ double _lv , double _lth , double _la , double _lq ) {
   double _ltrap0;
 if ( fabs ( _lv - _lth ) > 1e-6 ) {
     _ltrap0 = _la * ( _lv - _lth ) / ( 1.0 - exp ( - ( _lv - _lth ) / _lq ) ) ;
     }
   else {
     _ltrap0 = _la * _lq ;
     }
   
return _ltrap0;
 }
 
static void _hoc_trap0(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  trap0 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
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
  ena = _ion_ena;
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
  ena = _ion_ena;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
   _thread[0]._pval = (double*)ecalloc(3, sizeof(double));
 }
 
static void _thread_cleanup(Datum* _thread) {
   free((void*)(_thread[0]._pval));
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
  s = s0;
 {
   trates ( _threadargscomma_ v , ar , sh ) ;
   m = minf ;
   h = hinf ;
   s = sinf ;
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
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   thegna = gbar * m * m * m * h * s ;
   ina = thegna * ( v - ena ) ;
   }
 _current += ina;

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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
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
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
 _slist1[2] = s_columnindex;  _dlist1[2] = Ds_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/na3rp.mod";
static const char* nmodl_file_text = 
  "TITLE na3rp\n"
  ": Na current \n"
  ": modified from Jeff Magee. M.Migliore may97\n"
  ": added sh to account for higher threshold M.Migliore, Apr.2002\n"
  ": modified by RP to have slow inactivation given in Fleiderivsh et al.\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX na3rp\n"
  "	USEION na READ ena WRITE ina\n"
  "	RANGE  gbar, ar, sh,ina\n"
  "	RANGE minf, hinf, mtau, htau, sinf, taus, qinf, thinf\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	sh = 		8 (mV)\n"
  "	gbar = 		0.010 (mho/cm2)	\n"
  "								\n"
  "	tha =  		-30	(mV)		: v 1/2 for act	\n"
  "	qa = 		7.2	(mV)		: act slope (4.5)		\n"
  "	Ra = 		0.4	(/ms)		: open (v)		\n"
  "	Rb = 		0.124 (/ms)		: close (v)		\n"
  "\n"
  "	thi1 = 		-45	(mV)		: v 1/2 for inact 	\n"
  "	thi2 = 		-45 (mV)		: v 1/2 for inact 	\n"
  "	qd = 		1.5	(mV)		: inact tau slope\n"
  "	qg = 		1.5 (mV)\n"
  "	mmin =		0.02	\n"
  "	hmin =		0.5			\n"
  "	q10 =		2\n"
  "	Rg = 		0.01 (/ms)		: inact recov (v) 	\n"
  "	Rd = 		.03 (/ms)		: inact (v)	\n"
  "	qq = 		10 (mV)\n"
  "	tq = 		-55 (mV)\n"
  "\n"
  "	thinf = 	-50 (mV)		: inact inf slope	\n"
  "	qinf = 		4 (mV)			: inact inf slope \n"
  "\n"
  "	a0s =		0.001 (/ms)	\n"
  "	b0s =		0.0034 (/ms)\n"
  "	asvh =		-85 (mV) \n"
  "	bsvh =		-17 (mV) \n"
  "	avs =		30 (mV)\n"
  "	bvs =		10 (mV)\n"
  "	ar =		1 (1)			: 1=no inact., 0=max inact.\n"
  "	ena			(mV)            : must be explicitly def. in hoc\n"
  "	celsius\n"
  "	v 			(mV)\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(pS) = (picosiemens)\n"
  "	(um) = (micron)\n"
  "} \n"
  "\n"
  "ASSIGNED {\n"
  "	ina 		(mA/cm2)\n"
  "	thegna		(mho/cm2)\n"
  "	minf 		hinf 		\n"
  "	mtau (ms)	htau (ms) 	\n"
  "	sinf (ms)	taus (ms)\n"
  "}\n"
  "\n"
  "STATE { m h s}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	thegna = gbar*m*m*m*h*s\n"
  "	ina = thegna * (v - ena)\n"
  "} \n"
  "\n"
  "INITIAL {\n"
  "	trates(v,ar,sh)\n"
  "	m=minf  \n"
  "	h=hinf\n"
  "	s=sinf\n"
  "}\n"
  " \n"
  "FUNCTION alps(v(mV)) {  \n"
  "   alps = a0s*exp((asvh-v)/avs)\n"
  "}\n"
  "\n"
  "FUNCTION bets(v(mV)) {\n"
  "  bets = b0s/(exp((bsvh-v)/bvs)+1)\n"
  "}\n"
  "\n"
  "LOCAL mexp, hexp, sexp\n"
  "\n"
  "DERIVATIVE states {   \n"
  "	trates(v,ar,sh)      \n"
  "	m' = (minf-m)/mtau\n"
  "	h' = (hinf-h)/htau\n"
  "	s' = (sinf - s)/taus\n"
  "}\n"
  "\n"
  "PROCEDURE trates(vm,a2,sh2) {  \n"
  "	LOCAL  a, b, c, qt\n"
  "	qt=q10^((celsius-24)/10)\n"
  "	a = trap0(vm,tha+sh2,Ra,qa)\n"
  "	b = trap0(-vm,-tha-sh2,Rb,qa)\n"
  "	mtau = 1/(a+b)/qt\n"
  "	if (mtau<mmin) {mtau=mmin}\n"
  "	minf = a/(a+b)\n"
  "\n"
  "	a = trap0(vm,thi1+sh2,Rd,qd)\n"
  "	b = trap0(-vm,-thi2-sh2,Rg,qg)\n"
  "	htau =  1/(a+b)/qt\n"
  "	if (htau<hmin) {htau=hmin}\n"
  "	hinf = 1/(1+exp((vm-thinf-sh2)/qinf))\n"
  "	taus = 1/(alps(vm)+bets(vm))\n"
  "	c = alps(vm)*taus\n"
  "	sinf = c+a2*(1-c)\n"
  "}\n"
  "\n"
  "FUNCTION trap0(v,th,a,q) {\n"
  "	if (fabs(v-th) > 1e-6) {\n"
  "		trap0 = a * (v - th) / (1 - exp(-(v - th)/q))\n"
  "	} else {\n"
  "		trap0 = a * q\n"
  " 	}\n"
  "}	\n"
  ;
#endif
