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
 
#define nrn_init _nrn_init__GammaStim
#define _nrn_initial _nrn_initial__GammaStim
#define nrn_cur _nrn_cur__GammaStim
#define _nrn_current _nrn_current__GammaStim
#define nrn_jacob _nrn_jacob__GammaStim
#define nrn_state _nrn_state__GammaStim
#define _net_receive _net_receive__GammaStim 
#define event_time event_time__GammaStim 
#define init_sequence init_sequence__GammaStim 
#define seed seed__GammaStim 
 
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
#define interval _p[0]
#define interval_columnindex 0
#define start _p[1]
#define start_columnindex 1
#define noise _p[2]
#define noise_columnindex 2
#define duration _p[3]
#define duration_columnindex 3
#define order _p[4]
#define order_columnindex 4
#define refractoryPeriod _p[5]
#define refractoryPeriod_columnindex 5
#define event _p[6]
#define event_columnindex 6
#define on _p[7]
#define on_columnindex 7
#define end _p[8]
#define end_columnindex 8
#define _tsav _p[9]
#define _tsav_columnindex 9
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
 static double _hoc_init_sequence(void*);
 static double _hoc_invl(void*);
 static double _hoc_meanRndGamma(void*);
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
 "init_sequence", _hoc_init_sequence,
 "invl", _hoc_invl,
 "meanRndGamma", _hoc_meanRndGamma,
 "seed", _hoc_seed,
 0, 0
};
#define invl invl_GammaStim
#define meanRndGamma meanRndGamma_GammaStim
 extern double invl( double );
 extern double meanRndGamma( double , double , double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "interval", 1e-09, 1e+09,
 "noise", 0, 1,
 "order", 1, 6,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "interval", "ms",
 "start", "ms",
 "duration", "ms",
 "order", "1.0",
 "refractoryPeriod", "ms",
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
"GammaStim",
 "interval",
 "start",
 "noise",
 "duration",
 "order",
 "refractoryPeriod",
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
 	_p = nrn_prop_data_alloc(_mechtype, 10, _prop);
 	/*initialize range parameters*/
 	interval = 10;
 	start = 1;
 	noise = 0;
 	duration = 1000;
 	order = 1;
 	refractoryPeriod = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 10;
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

 void _GammaStim_reg() {
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
  hoc_register_prop_size(_mechtype, 10, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
 add_nrn_artcell(_mechtype, 2);
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GammaStim /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/GammaStim.mod\n");
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
static int init_sequence(double);
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
 
static int  init_sequence (  double _lt ) {
   on = 1.0 ;
   event = _lt ;
   end = _lt + 1e-6 + duration ;
    return 0; }
 
static double _hoc_init_sequence(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 init_sequence (  *getarg(1) );
 return(_r);
}
 
double invl (  double _lmean ) {
   double _linvl;
 if ( _lmean <= 0. ) {
     _lmean = .01 ;
     }
   if ( noise  == 0.0 ) {
     _linvl = _lmean ;
     }
   else {
     _linvl = ( 1. - noise ) * _lmean + noise * meanRndGamma ( _threadargscomma_ order , refractoryPeriod , _lmean ) ;
     }
   
return _linvl;
 }
 
static double _hoc_invl(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  invl (  *getarg(1) );
 return(_r);
}
 
static int  event_time (  ) {
   event = event + invl ( _threadargscomma_ interval ) ;
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
   if ( _lflag  == 0.0 ) {
     if ( _args[0] > 0.0  && on  == 0.0 ) {
       init_sequence ( _threadargscomma_ t ) ;
       artcell_net_send ( _tqitem, _args, _pnt, t +  0.0 , 1.0 ) ;
       }
     else if ( _args[0] < 0.0  && on  == 1.0 ) {
       on = 0.0 ;
       }
     }
   if ( _lflag  == 3.0 ) {
     if ( on  == 0.0 ) {
       init_sequence ( _threadargscomma_ t ) ;
       artcell_net_send ( _tqitem, _args, _pnt, t +  0.0 , 1.0 ) ;
       }
     }
   if ( _lflag  == 1.0  && on  == 1.0 ) {
     net_event ( _pnt, t ) ;
     event_time ( _threadargs_ ) ;
     if ( on  == 1.0 ) {
       artcell_net_send ( _tqitem, _args, _pnt, t +  event - t , 1.0 ) ;
       }
     artcell_net_send ( _tqitem, _args, _pnt, t +  .1 , 2.0 ) ;
     }
   } }
 
double meanRndGamma (  double _lgammaOrder , double _lrefractoryPeriod , double _lmean ) {
   double _lmeanRndGamma;
 double _lx ;
 _lx = 1.0 ;
   {int  _li ;for ( _li = 0 ; _li <= ((int) _lgammaOrder ) - 1 ; _li ++ ) {
     _lx = _lx * scop_random ( ) ;
     } }
   _lx = - log ( _lx ) * ( interval - _lrefractoryPeriod ) / _lgammaOrder ;
   _lmeanRndGamma = _lx + _lrefractoryPeriod ;
   
return _lmeanRndGamma;
 }
 
static double _hoc_meanRndGamma(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  meanRndGamma (  *getarg(1) , *getarg(2) , *getarg(3) );
 return(_r);
}

static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   on = 0.0 ;
   if ( order < 1.0  || order > 6.0 ) {
     order = 1.0 ;
     }
   if ( noise < 0.0 ) {
     noise = 0.0 ;
     }
   if ( noise > 1.0 ) {
     noise = 1.0 ;
     }
   if ( start >= 0.0 ) {
     event = start + invl ( _threadargscomma_ interval ) - interval * ( 1. - noise ) ;
     if ( event < 0.0 ) {
       event = 0.0 ;
       }
     artcell_net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  event , 3.0 ) ;
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
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/GammaStim.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "Modification by Johannes Luthman of the built-in NetStim.mod of NEURON 6.1.\n"
  "NB, this code has not been used with CVode.\n"
  "\n"
  "Changes from NetStim:\n"
  "    The output events can be set to follow gamma distributions of order 1-6,  \n"
  "    where 1 corresponds to the original Poisson process generated by NetStim.mod.\n"
  "    The gamma process is generated in the same way as that given by timetable.c\n"
  "    in GENESIS 2.3.\n"
  "    A refractory period has been added.\n"
  "    The output length is determined by duration in ms instead of number of events.\n"
  "\n"
  "Parameters:\n"
  "    interval: 	mean time between spikes (ms)\n"
  "    start:      start of first spike (ms)\n"
  "    noise:      amount of randomness in the spike train [0-1], where 0 generates\n"
  "                fully regular spiking with isi given by parameter interval.\n"
  "    duration:   length in ms of the spike train.\n"
  "    order:      Integers [1-6] giving the order of gamma distribution.\n"
  "    refractoryPeriod (ms)\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON  {\n"
  "    ARTIFICIAL_CELL GammaStim\n"
  "    RANGE interval, start, duration, order, noise, refractoryPeriod\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    interval =          10      (ms) <1e-9,1e9>	: time between spikes (msec)\n"
  "    start =             1       (ms)            : start of first spike\n"
  "    noise =             0 <0,1>                 : amount of randomness (0.0 - 1.0) in spike timing.\n"
  "    duration =          1000    (ms)		    : input duration\n"
  "    order =             1 <1,6>                 : order of gamma distribution. 1=pure poisson process.\n"
  "    refractoryPeriod =  0       (ms)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    event                       (ms)\n"
  "    on\n"
  "    end                         (ms)\n"
  "}\n"
  "\n"
  "PROCEDURE seed(x) {\n"
  "    set_seed(x) \n"
  "    : Calling .seed() from hoc affects the event streams \n"
  "    : generated by all NetStims, see http://www.neuron.yale.edu/phpBB2/viewtopic.php?p=3285&sid=511cb3101cc8f4c12d47299198ed40c2\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    on = 0 : off\n"
  "    if (order < 1 || order > 6) {\n"
  "        order = 1\n"
  "    }\n"
  "    if (noise < 0) {\n"
  "        noise = 0\n"
  "    }\n"
  "    if (noise > 1) {\n"
  "        noise = 1\n"
  "    }\n"
  "    if (start >= 0) {\n"
  "        : randomize the first spike so on average it occurs at\n"
  "        : start + noise*interval\n"
  "        event = start + invl(interval) - interval*(1. - noise)\n"
  "        : but not earlier than 0\n"
  "        if (event < 0) {\n"
  "            event = 0\n"
  "        }\n"
  "        net_send(event, 3): (Cdur,nspike), see The NEURON book ch 10 p343\n"
  "    }\n"
  "}\n"
  "\n"
  "PROCEDURE init_sequence(t(ms)) {\n"
  "    on = 1\n"
  "    event = t\n"
  "    end = t + 1e-6 + duration\n"
  "}\n"
  "\n"
  "FUNCTION invl(mean (ms)) (ms) {\n"
  "    : This function returns spiking interval\n"
  "    if (mean <= 0.) {\n"
  "        mean = .01 (ms)\n"
  "    }\n"
  "    if (noise == 0) {\n"
  "        invl = mean\n"
  "    }else{\n"
  "        invl = (1. - noise) * mean + noise*meanRndGamma(order, refractoryPeriod, mean)\n"
  "    }\n"
  "}\n"
  "\n"
  "PROCEDURE event_time() {\n"
  "    event = event + invl(interval)\n"
  "    if (event > end) {\n"
  "        on = 0\n"
  "    }\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "    if (flag == 0) { : external event\n"
  "        if (w > 0 && on == 0) { : turn on spike sequence\n"
  "            init_sequence(t)\n"
  "            net_send(0, 1)\n"
  "        } else if (w < 0 && on == 1) { : turn off spiking\n"
  "            on = 0\n"
  "        }\n"
  "    }\n"
  "    if (flag == 3) { : from INITIAL\n"
  "        if (on == 0) {\n"
  "            init_sequence(t)\n"
  "            net_send(0, 1)\n"
  "        }\n"
  "    }\n"
  "    if (flag == 1 && on == 1) {\n"
  "        net_event(t) : See NEURON book p. 345. Sum: net_event tells NetCon something has happened.\n"
  "        event_time()\n"
  "        if (on == 1) {\n"
  "            net_send(event - t, 1)\n"
  "        }\n"
  "        net_send(.1, 2)\n"
  "    }\n"
  "}\n"
  "\n"
  "FUNCTION meanRndGamma(gammaOrder(1), refractoryPeriod(ms), mean(ms)) (1) {\n"
  "    : Code translated from the timetable object of GENESIS 2.3.\n"
  "	LOCAL x\n"
  "\n"
  "	x = 1.0\n"
  "	FROM i = 0 TO gammaOrder-1 {\n"
  "	    x = x * scop_random()\n"
  "    }\n"
  "	x = -log(x) * (interval - refractoryPeriod) / gammaOrder\n"
  "	meanRndGamma = x + refractoryPeriod\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
