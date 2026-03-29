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
 
#define nrn_init _nrn_init__GammaProcess
#define _nrn_initial _nrn_initial__GammaProcess
#define nrn_cur _nrn_cur__GammaProcess
#define _nrn_current _nrn_current__GammaProcess
#define nrn_jacob _nrn_jacob__GammaProcess
#define nrn_state _nrn_state__GammaProcess
#define _net_receive _net_receive__GammaProcess 
#define event_time event_time__GammaProcess 
#define seed seed__GammaProcess 
 
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
#define alpha _p[0]
#define alpha_columnindex 0
#define beta _p[1]
#define beta_columnindex 1
#define start _p[2]
#define start_columnindex 2
#define duration _p[3]
#define duration_columnindex 3
#define event _p[4]
#define event_columnindex 4
#define on _p[5]
#define on_columnindex 5
#define end _p[6]
#define end_columnindex 6
#define _tsav _p[7]
#define _tsav_columnindex 7
#define _nd_area  *_ppvar[0]._pval
 
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
 static double _hoc_event_time(void*);
 static double _hoc_rand_gamma(void*);
 static double _hoc_seed(void*);
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

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "event_time", _hoc_event_time,
 "rand_gamma", _hoc_rand_gamma,
 "seed", _hoc_seed,
 0, 0
};
#define rand_gamma rand_gamma_GammaProcess
 extern double rand_gamma( double , double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "beta", 1e-09, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "beta", "/ms",
 "start", "ms",
 "duration", "ms",
 0,0
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"GammaProcess",
 "alpha",
 "beta",
 "start",
 "duration",
 0,
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 8, _prop);
 	/*initialize range parameters*/
 	alpha = 1;
 	beta = 0.1;
 	start = 1;
 	duration = 1000;
  }
 	_prop->param = _p;
 	_prop->param_size = 8;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _gammapointprocess_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,(void*)0, (void*)0, (void*)0, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 8, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
 add_nrn_artcell(_mechtype, 2);
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GammaProcess /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/gammapointprocess.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int event_time();
static int seed(double);
 
static int  seed (  double _lx ) {
   set_seed ( _lx ) ;
    return 0; }
 
static double _hoc_seed(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 seed (  *getarg(1) );
 return(_r);
}
 
static int  event_time (  ) {
   event = event + rand_gamma ( _threadargscomma_ alpha , beta ) ;
   if ( event > end ) {
     on = 0.0 ;
     }
    return 0; }
 
static double _hoc_event_time(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 event_time (  );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{    _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(Object*); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 2.0 ) {
     if ( on  == 0.0 ) {
       on = 1.0 ;
       event = t ;
       end = t + 1e-6 + duration ;
       artcell_net_send ( _tqitem, _args, _pnt, t +  0.0 , 1.0 ) ;
       }
     }
   if ( _lflag  == 1.0  && on  == 1.0 ) {
     net_event ( _pnt, t ) ;
     event = event + rand_gamma ( _threadargscomma_ alpha , beta ) ;
     if ( event > end ) {
       on = 0.0 ;
       }
     if ( on  == 1.0 ) {
       artcell_net_send ( _tqitem, _args, _pnt, t +  event - t , 1.0 ) ;
       }
     }
   } }
 
double rand_gamma (  double _lalpha , double _lbeta ) {
   double _lrand_gamma;
 double _li , _lZ , _lU , _lT ;
 _li = 1.0 ;
   _lZ = 1.0 ;
   while ( _li <= _lalpha ) {
     _lU = scop_random ( ) ;
     _lZ = _lZ * _lU ;
     _li = _li + 1.0 ;
     }
   _lT = - log ( _lZ ) / _lalpha ;
   _li = 1.0 ;
   while ( _lT > 0.0 ) {
     _lT = _lT - _lbeta * dt ;
     _li = _li + 1.0 ;
     }
   _lrand_gamma = _li * dt ;
   
return _lrand_gamma;
 }
 
static double _hoc_rand_gamma(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  rand_gamma (  *getarg(1) , *getarg(2) );
 return(_r);
}

static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   on = 0.0 ;
   event = rand_gamma ( _threadargscomma_ alpha , beta ) ;
   if ( start >= 0.0 ) {
     artcell_net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  event , 2.0 ) ;
     }
   }

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
 _tsav = -1e20;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
 v=_v;
{
}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/gammapointprocess.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "\n"
  "Spike generator following a gamma process.\n"
  "\n"
  "Gamma distributed random variables are generated using the Marsaglia-Tang method:\n"
  "\n"
  "  G. Marsaglia and W. Tsang (2000) A simple method for generating gamma variables.\n"
  "  ACM Transactions on Mathematical Software, 26(3):363-372. doi:10.1145/358407.358414\n"
  "\n"
  "Parameters:\n"
  "    alpha:     shape parameter of the gamma distribution. 1 = Poisson process.\n"
  "    beta:      rate parameter of gamma distribution (/ms). Note that the mean interval is alpha/beta\n"
  "    start:     start of gamma process (ms)\n"
  "    duration:  length in ms of the spike train.\n"
  "\n"
  "Author: Andrew P. Davison, UNIC, CNRS\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON  {\n"
  "    ARTIFICIAL_CELL GammaProcess\n"
  "    RANGE alpha, beta, start, duration\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    alpha =     1                           : shape parameter of gamma distribution. 1 = Poisson process.\n"
  "    beta =      0.1     (/ms) <1e-9,1e9>    : rate parameter of gamma distribution\n"
  "                                            : mean interval is alpha/beta\n"
  "    start =     1       (ms)                : start of first spike\n"
  "    duration =  1000    (ms)                : input duration\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    event               (ms)\n"
  "    on\n"
  "    end                 (ms)\n"
  "}\n"
  "\n"
  "PROCEDURE seed(x) {\n"
  "    set_seed(x)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    on = 0\n"
  "    event = rand_gamma(alpha, beta)\n"
  "    if (start >= 0) {\n"
  "        net_send(event, 2)\n"
  "    }\n"
  "}\n"
  "\n"
  "PROCEDURE event_time() {\n"
  "    event = event + rand_gamma(alpha, beta)\n"
  "    if (event > end) {\n"
  "        on = 0\n"
  "    }\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "    if (flag == 2) { : from INITIAL\n"
  "        if (on == 0) {\n"
  "            on = 1\n"
  "            event = t\n"
  "            end = t + 1e-6 + duration\n"
  "            net_send(0, 1)\n"
  "        }\n"
  "    }\n"
  "    if (flag == 1 && on == 1) {\n"
  "        net_event(t)\n"
  "        event = event + rand_gamma(alpha, beta)\n"
  "        if (event > end) {\n"
  "            on = 0\n"
  "        }\n"
  "        if (on == 1) {\n"
  "            net_send(event - t, 1)\n"
  "        }\n"
  "    }\n"
  "}\n"
  "\n"
  "FUNCTION rand_gamma(alpha(1), beta(/ms)) (1) {\n"
  "    LOCAL i, Z, U, T\n"
  "    i = 1\n"
  "    Z = 1.0\n"
  "    while (i <= alpha) {\n"
  "        U = scop_random()\n"
  "        Z = Z*U\n"
  "        i = i + 1\n"
  "    }\n"
  "    T = -log(Z)/alpha\n"
  "    i = 1\n"
  "    while (T>0) { \n"
  "        T = T - beta*dt\n"
  "        i = i + 1\n"
  "    }\n"
  "\n"
  "    rand_gamma = i*dt\n"
  "}\n"
  ;
#endif
