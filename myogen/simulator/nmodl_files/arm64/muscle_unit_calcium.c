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
 
#define nrn_init _nrn_init__muscle_unit_calcium
#define _nrn_initial _nrn_initial__muscle_unit_calcium
#define nrn_cur _nrn_cur__muscle_unit_calcium
#define _nrn_current _nrn_current__muscle_unit_calcium
#define nrn_jacob _nrn_jacob__muscle_unit_calcium
#define nrn_state _nrn_state__muscle_unit_calcium
#define _net_receive _net_receive__muscle_unit_calcium 
#define rate rate__muscle_unit_calcium 
#define states_force states_force__muscle_unit_calcium 
 
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
#define Tc _p[0]
#define Tc_columnindex 0
#define Fmax _p[1]
#define Fmax_columnindex 1
#define k1 _p[2]
#define k1_columnindex 2
#define k2 _p[3]
#define k2_columnindex 3
#define k3 _p[4]
#define k3_columnindex 4
#define k4 _p[5]
#define k4_columnindex 5
#define k5i _p[6]
#define k5i_columnindex 6
#define k6i _p[7]
#define k6i_columnindex 7
#define k _p[8]
#define k_columnindex 8
#define Rmax _p[9]
#define Rmax_columnindex 9
#define Umax _p[10]
#define Umax_columnindex 10
#define tau1 _p[11]
#define tau1_columnindex 11
#define tau2 _p[12]
#define tau2_columnindex 12
#define phi1 _p[13]
#define phi1_columnindex 13
#define phi2 _p[14]
#define phi2_columnindex 14
#define phi3 _p[15]
#define phi3_columnindex 15
#define phi4 _p[16]
#define phi4_columnindex 16
#define SF_AM _p[17]
#define SF_AM_columnindex 17
#define T0 _p[18]
#define T0_columnindex 18
#define R _p[19]
#define R_columnindex 19
#define R1 _p[20]
#define R1_columnindex 20
#define R2 _p[21]
#define R2_columnindex 21
#define c1i _p[22]
#define c1i_columnindex 22
#define c1n1 _p[23]
#define c1n1_columnindex 23
#define c1n2 _p[24]
#define c1n2_columnindex 24
#define c1n3 _p[25]
#define c1n3_columnindex 25
#define tauc1 _p[26]
#define tauc1_columnindex 26
#define c2i _p[27]
#define c2i_columnindex 27
#define c2n1 _p[28]
#define c2n1_columnindex 28
#define c2n2 _p[29]
#define c2n2_columnindex 29
#define c2n3 _p[30]
#define c2n3_columnindex 30
#define tauc2 _p[31]
#define tauc2_columnindex 31
#define c3 _p[32]
#define c3_columnindex 32
#define c4 _p[33]
#define c4_columnindex 33
#define c5 _p[34]
#define c5_columnindex 34
#define alpha _p[35]
#define alpha_columnindex 35
#define temp _p[36]
#define temp_columnindex 36
#define spike _p[37]
#define spike_columnindex 37
#define F _p[38]
#define F_columnindex 38
#define A _p[39]
#define A_columnindex 39
#define k5 _p[40]
#define k5_columnindex 40
#define k6 _p[41]
#define k6_columnindex 41
#define AMinf _p[42]
#define AMinf_columnindex 42
#define AMtau _p[43]
#define AMtau_columnindex 43
#define c1inf _p[44]
#define c1inf_columnindex 44
#define c2inf _p[45]
#define c2inf_columnindex 45
#define CaSR _p[46]
#define CaSR_columnindex 46
#define CaT _p[47]
#define CaT_columnindex 47
#define AM _p[48]
#define AM_columnindex 48
#define x1 _p[49]
#define x1_columnindex 49
#define x2 _p[50]
#define x2_columnindex 50
#define xm _p[51]
#define xm_columnindex 51
#define CaSRCS _p[52]
#define CaSRCS_columnindex 52
#define Ca _p[53]
#define Ca_columnindex 53
#define CaB _p[54]
#define CaB_columnindex 54
#define c1 _p[55]
#define c1_columnindex 55
#define c2 _p[56]
#define c2_columnindex 56
#define DCaSR _p[57]
#define DCaSR_columnindex 57
#define DCaT _p[58]
#define DCaT_columnindex 58
#define DAM _p[59]
#define DAM_columnindex 59
#define Dx1 _p[60]
#define Dx1_columnindex 60
#define Dx2 _p[61]
#define Dx2_columnindex 61
#define Dxm _p[62]
#define Dxm_columnindex 62
#define DCaSRCS _p[63]
#define DCaSRCS_columnindex 63
#define DCa _p[64]
#define DCa_columnindex 64
#define DCaB _p[65]
#define DCaB_columnindex 65
#define Dc1 _p[66]
#define Dc1_columnindex 66
#define Dc2 _p[67]
#define Dc2_columnindex 67
#define v _p[68]
#define v_columnindex 68
#define _g _p[69]
#define _g_columnindex 69
#define _tsav _p[70]
#define _tsav_columnindex 70
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_U(void*);
 static double _hoc_phi(void*);
 static double _hoc_rate(void*);
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
 _extcall_prop = _prop;
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
 "U", _hoc_U,
 "phi", _hoc_phi,
 "rate", _hoc_rate,
 0, 0
};
#define U U_muscle_unit_calcium
#define phi phi_muscle_unit_calcium
 extern double U( _threadargsprotocomma_ double );
 extern double phi( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define B0 B0_muscle_unit_calcium
 double B0 = 0.00043;
#define CS0 CS0_muscle_unit_calcium
 double CS0 = 0.03;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 0,0
};
 static double AM0 = 0;
 static double CaB0 = 0;
 static double Ca0 = 0;
 static double CaSRCS0 = 0;
 static double CaT0 = 0;
 static double CaSR0 = 0;
 static double c20 = 0;
 static double c10 = 0;
 static double delta_t = 0.01;
 static double xm0 = 0;
 static double x20 = 0;
 static double x10 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "CS0_muscle_unit_calcium", &CS0_muscle_unit_calcium,
 "B0_muscle_unit_calcium", &B0_muscle_unit_calcium,
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
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[2]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"muscle_unit_calcium",
 "Tc",
 "Fmax",
 "k1",
 "k2",
 "k3",
 "k4",
 "k5i",
 "k6i",
 "k",
 "Rmax",
 "Umax",
 "tau1",
 "tau2",
 "phi1",
 "phi2",
 "phi3",
 "phi4",
 "SF_AM",
 "T0",
 "R",
 "R1",
 "R2",
 "c1i",
 "c1n1",
 "c1n2",
 "c1n3",
 "tauc1",
 "c2i",
 "c2n1",
 "c2n2",
 "c2n3",
 "tauc2",
 "c3",
 "c4",
 "c5",
 "alpha",
 "temp",
 0,
 "spike",
 "F",
 "A",
 "k5",
 "k6",
 "AMinf",
 "AMtau",
 "c1inf",
 "c2inf",
 0,
 "CaSR",
 "CaT",
 "AM",
 "x1",
 "x2",
 "xm",
 "CaSRCS",
 "Ca",
 "CaB",
 "c1",
 "c2",
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
 	_p = nrn_prop_data_alloc(_mechtype, 71, _prop);
 	/*initialize range parameters*/
 	Tc = 100;
 	Fmax = 1;
 	k1 = 3000;
 	k2 = 3;
 	k3 = 400;
 	k4 = 1;
 	k5i = 400000;
 	k6i = 150;
 	k = 850;
 	Rmax = 10;
 	Umax = 2000;
 	tau1 = 1;
 	tau2 = 13;
 	phi1 = 0.004;
 	phi2 = 0.98;
 	phi3 = 0.0002;
 	phi4 = 0.999;
 	SF_AM = 5;
 	T0 = 7e-05;
 	R = 0;
 	R1 = 0;
 	R2 = 0;
 	c1i = 0.154;
 	c1n1 = 0.01;
 	c1n2 = 0.15;
 	c1n3 = 0.01;
 	tauc1 = 85;
 	c2i = 0.11;
 	c2n1 = -0.0315;
 	c2n2 = 0.27;
 	c2n3 = 0.015;
 	tauc2 = 70;
 	c3 = 54.717;
 	c4 = -18.847;
 	c5 = 3.905;
 	alpha = 2;
 	temp = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 71;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _net_receive(Point_process*, double*, double);
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _muscle_unit_calcium_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 5,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
  _extcall_thread = (Datum*)ecalloc(4, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 71, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 muscle_unit_calcium /Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/muscle_unit_calcium.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rate(_threadargsprotocomma_ double, double, double);
 
#define _deriv1_advance _thread[0]._i
#define _dith1 1
#define _recurse _thread[2]._i
#define _newtonspace1 _thread[3]._pvoid
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist2[10];
  static int _slist1[10], _dlist1[10];
 static int states_force(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   Dx1 = x2 ;
   Dx2 = - 2.0 / Tc * x2 - 1.0 / ( Tc * Tc ) * x1 + CaT / 0.0001 / Tc ;
   DCaSR = - R + U ( _threadargscomma_ Ca ) - k1 * CS0 * CaSR + ( k1 * CaSR + k2 ) * CaSRCS ;
   DCaSRCS = k1 * CS0 * CaSR - ( k1 * CaSR + k2 ) * CaSRCS ;
   DCa = - k5 * T0 * Ca + ( k5 * Ca + k6 ) * CaT + R - U ( _threadargscomma_ Ca ) - k3 * B0 * Ca + ( k3 * Ca + k4 ) * CaB ;
   DCaB = k3 * B0 * Ca - ( k3 * Ca + k4 ) * CaB ;
   DCaT = k5 * T0 * Ca - ( k5 * Ca + k6 ) * CaT ;
   DAM = 0.0 ;
   Dc1 = 0.0 ;
   Dc2 = 0.0 ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 Dx1 = Dx1  / (1. - dt*( 0.0 )) ;
 Dx2 = Dx2  / (1. - dt*( ( - 2.0 / Tc )*( 1.0 ) )) ;
 DCaSR = DCaSR  / (1. - dt*( ( - ( k1 * CS0 )*( 1.0 ) ) + ( ( ( k1 )*( 1.0 ) ) )*( CaSRCS ) )) ;
 DCaSRCS = DCaSRCS  / (1. - dt*( ( - ( ( k1 * CaSR + k2 ) )*( 1.0 ) ) )) ;
 DCa = DCa  / (1. - dt*( (( - k5 * T0 * ( Ca  + .001) + ( k5 * ( Ca  + .001) + k6 ) * CaT + R - U ( _threadargscomma_ ( Ca  + .001) ) - k3 * B0 * ( Ca  + .001) + ( k3 * ( Ca  + .001) + k4 ) * CaB ) - ( - k5 * T0 * Ca + ( k5 * Ca + k6 ) * CaT + R - U ( _threadargscomma_ Ca ) - k3 * B0 * Ca + ( k3 * Ca + k4 ) * CaB  )) / .001 )) ;
 DCaB = DCaB  / (1. - dt*( ( - ( ( k3 * Ca + k4 ) )*( 1.0 ) ) )) ;
 DCaT = DCaT  / (1. - dt*( ( - ( ( k5 * Ca + k6 ) )*( 1.0 ) ) )) ;
 DAM = DAM  / (1. - dt*( 0.0 )) ;
 Dc1 = Dc1  / (1. - dt*( 0.0 )) ;
 Dc2 = Dc2  / (1. - dt*( 0.0 )) ;
  return 0;
}
 /*END CVODE*/
 
static int states_force (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0; int error = 0;
 { double* _savstate1 = _thread[_dith1]._pval;
 double* _dlist2 = _thread[_dith1]._pval + 10;
 int _counte = -1;
 if (!_recurse) {
 _recurse = 1;
 {int _id; for(_id=0; _id < 10; _id++) { _savstate1[_id] = _p[_slist1[_id]];}}
 error = nrn_newton_thread(_newtonspace1, 10,_slist2, _p, states_force, _dlist2, _ppvar, _thread, _nt);
 _recurse = 0; if(error) {abort_run(error);}}
 {
   Dx1 = x2 ;
   Dx2 = - 2.0 / Tc * x2 - 1.0 / ( Tc * Tc ) * x1 + CaT / 0.0001 / Tc ;
   DCaSR = - R + U ( _threadargscomma_ Ca ) - k1 * CS0 * CaSR + ( k1 * CaSR + k2 ) * CaSRCS ;
   DCaSRCS = k1 * CS0 * CaSR - ( k1 * CaSR + k2 ) * CaSRCS ;
   DCa = - k5 * T0 * Ca + ( k5 * Ca + k6 ) * CaT + R - U ( _threadargscomma_ Ca ) - k3 * B0 * Ca + ( k3 * Ca + k4 ) * CaB ;
   DCaB = k3 * B0 * Ca - ( k3 * Ca + k4 ) * CaB ;
   DCaT = k5 * T0 * Ca - ( k5 * Ca + k6 ) * CaT ;
   DAM = 0.0 ;
   Dc1 = 0.0 ;
   Dc2 = 0.0 ;
   {int _id; for(_id=0; _id < 10; _id++) {
if (_deriv1_advance) {
 _dlist2[++_counte] = _p[_dlist1[_id]] - (_p[_slist1[_id]] - _savstate1[_id])/dt;
 }else{
_dlist2[++_counte] = _p[_slist1[_id]] - _savstate1[_id];}}}
 } }
 return _reset;}
 
static int  rate ( _threadargsprotocomma_ double _lCaT , double _lAM , double _lt ) {
   k5 = phi ( _threadargscomma_ 5.0 ) * k5i ;
   k6 = k6i / ( 1.0 + SF_AM * _lAM ) ;
   AMinf = 0.5 * ( 1.0 + tanh ( ( _lCaT / T0 - c1 ) / c2 ) ) ;
   AMtau = c3 / ( cosh ( ( _lCaT / T0 - c4 ) / ( 2.0 * c5 ) ) ) ;
   c1inf = c1n1 * ( 1.0 + tanh ( ( _lCaT / T0 - c1n2 ) / c1n3 ) ) + c1i ;
   c2inf = c2n1 * ( 1.0 + tanh ( ( _lCaT / T0 - c2n2 ) / c2n3 ) ) + c2i ;
    return 0; }
 
static double _hoc_rate(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 rate ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) );
 return(_r);
}
 
double U ( _threadargsprotocomma_ double _lx ) {
   double _lU;
 if ( _lx >= 0.0 ) {
     _lU = Umax * pow( ( pow( _lx , 2.0 ) * pow( k , 2.0 ) / ( 1.0 + _lx * k + pow( _lx , 2.0 ) * pow( k , 2.0 ) ) ) , 2.0 ) ;
     }
   else {
     _lU = 0.0 ;
     }
   
return _lU;
 }
 
static double _hoc_U(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  U ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
double phi ( _threadargsprotocomma_ double _lx ) {
   double _lphi;
 if ( _lx <= 5.0 ) {
     _lphi = phi1 * _lx + phi2 ;
     }
   else {
     _lphi = phi3 * _lx + phi4 ;
     }
   
return _lphi;
 }
 
static double _hoc_phi(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  phi ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(Object*); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   spike = 2.7182818 / dt ;
   R1 = R1 + 1.0 ;
   R2 = R2 + 1.0 ;
   } }
 
static int _ode_count(int _type){ return 10;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 10; ++_i) {
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
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
   _thread[_dith1]._pval = (double*)ecalloc(20, sizeof(double));
   _newtonspace1 = nrn_cons_newtonspace(10);
 }
 
static void _thread_cleanup(Datum* _thread) {
   free((void*)(_thread[_dith1]._pval));
   nrn_destroy_newtonspace(_newtonspace1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  AM = AM0;
  CaB = CaB0;
  Ca = Ca0;
  CaSRCS = CaSRCS0;
  CaT = CaT0;
  CaSR = CaSR0;
  c2 = c20;
  c1 = c10;
  xm = xm0;
  x2 = x20;
  x1 = x10;
 {
   x1 = 0.0 ;
   x2 = 0.0 ;
   spike = 0.0 ;
   CaSR = 0.0025 ;
   CaSRCS = 0.0 ;
   Ca = 1e-10 ;
   CaT = 0.0 ;
   AM = 0.0 ;
   CaB = 0.0 ;
   c1 = 0.154 ;
   c2 = 0.11 ;
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
 _tsav = -1e20;
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{
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
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
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
 {  _deriv1_advance = 1;
 derivimplicit_thread(10, _slist1, _dlist1, _p, states_force, _ppvar, _thread, _nt);
_deriv1_advance = 0;
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 10; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 } {
   R1 = R1 * exp ( - dt / tau2 ) ;
   R2 = R2 * exp ( - dt / tau2 ) * exp ( - dt / tau1 ) ;
   R = CaSR * Rmax * ( R1 - R2 ) ;
   rate ( _threadargscomma_ CaT , AM , t ) ;
   F = Fmax * x1 ;
   spike = 0.0 ;
   A = pow( AM , alpha ) ;
   }
}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = x1_columnindex;  _dlist1[0] = Dx1_columnindex;
 _slist1[1] = x2_columnindex;  _dlist1[1] = Dx2_columnindex;
 _slist1[2] = CaSR_columnindex;  _dlist1[2] = DCaSR_columnindex;
 _slist1[3] = CaSRCS_columnindex;  _dlist1[3] = DCaSRCS_columnindex;
 _slist1[4] = Ca_columnindex;  _dlist1[4] = DCa_columnindex;
 _slist1[5] = CaB_columnindex;  _dlist1[5] = DCaB_columnindex;
 _slist1[6] = CaT_columnindex;  _dlist1[6] = DCaT_columnindex;
 _slist1[7] = AM_columnindex;  _dlist1[7] = DAM_columnindex;
 _slist1[8] = c1_columnindex;  _dlist1[8] = Dc1_columnindex;
 _slist1[9] = c2_columnindex;  _dlist1[9] = Dc2_columnindex;
 _slist2[0] = AM_columnindex;
 _slist2[1] = CaB_columnindex;
 _slist2[2] = Ca_columnindex;
 _slist2[3] = CaSRCS_columnindex;
 _slist2[4] = CaT_columnindex;
 _slist2[5] = CaSR_columnindex;
 _slist2[6] = c2_columnindex;
 _slist2[7] = c1_columnindex;
 _slist2[8] = x2_columnindex;
 _slist2[9] = x1_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/oj98yqyk/code/MyoGen/myogen/simulator/nmodl_files/muscle_unit_calcium.mod";
static const char* nmodl_file_text = 
  "NEURON {\n"
  "    POINT_PROCESS muscle_unit_calcium\n"
  "    RANGE Tc, Fmax, spike, F, A, temp\n"
  "	RANGE k1, k2, k3, k4, k5, k6, k, k5i, k6i\n"
  "	RANGE Umax, Rmax, tau1, tau2, R, R1, R2\n"
  "	RANGE phi0, phi1, phi2, phi3, phi4\n"
  "	RANGE AMinf, AMtau, SF_AM, T0\n"
  "	RANGE c1i, c1n1, c1n2, c1n3, tauc1, c2i, c2n1, c2n2, c2n3, tauc2, c3, c4, c5, c1inf, c2inf\n"
  "	RANGE acm, alpha :alpha1, alpha2, alpha3, beta, gamma\n"
  "}\n"
  "\n"
  "PARAMETER{\n"
  "    Tc = 100\n"
  "    Fmax = 1\n"
  "    :: Calcium dynamics ::\n"
  "	k1 = 3000		: M-1*ms-1\n"
  "	k2 = 3			: ms-1\n"
  "	k3 = 400		: M-1*ms-1\n"
  "	k4 = 1			: ms-1\n"
  "	k5i = 4e5		: M-1*ms-1\n"
  "	k6i = 150		: ms-1\n"
  "	k = 850			: M-1	\n"
  "	Rmax = 10		: ms-1\n"
  "	Umax = 2000		: M-1*ms-1\n"
  "	tau1 = 1			: ms\n"
  "	tau2 = 13			: ms\n"
  "	phi1 = 0.004\n"
  "	phi2 = 0.98\n"
  "	phi3 = 0.0002\n"
  "	phi4 = 0.999\n"
  "	SF_AM = 5\n"
  "	CS0 = 0.03     	:[M]\n"
  "	B0 = 0.00043	:[M]\n"
  "	T0 = 0.00007 	:[M]\n"
  "    R = 0\n"
  "	R1 = 0\n"
  "	R2 = 0 \n"
  "\n"
  "	:: Muscle activation::\n"
  "	c1i = 0.154\n"
  "	c1n1 = 0.01\n"
  "	c1n2 = 0.15\n"
  "	c1n3 = 0.01\n"
  "	tauc1 = 85\n"
  "	c2i = 0.11\n"
  "	c2n1 = -0.0315\n"
  "	c2n2 = 0.27\n"
  "	c2n3 = 0.015\n"
  "	tauc2 = 70\n"
  "	c3 = 54.717\n"
  "	c4 = -18.847\n"
  "	c5 = 3.905\n"
  "	alpha = 2\n"
  "	:alpha1 = 4.77\n"
  "	:alpha2 = 400\n"
  "	:alpha3 = 160\n"
  "	:beta = 0.47\n"
  "	:gamma = 0.001\n"
  "	temp = 0\n"
  "}\n"
  "\n"
  "ASSIGNED{\n"
  "    spike\n"
  "    F\n"
  "	A\n"
  "	k5\n"
  "	k6\n"
  "	AMinf\n"
  "	AMtau\n"
  "	c1inf\n"
  "	c2inf\n"
  "	:xm_temp1\n"
  "	:xm_temp2\n"
  "	:vm\n"
  "	:acm\n"
  "}\n"
  "\n"
  "STATE{\n"
  "    CaSR CaT AM x1 x2 xm CaSRCS Ca CaB c1 c2\n"
  "}\n"
  "\n"
  "INITIAL{\n"
  "    x1 = 0.0\n"
  "    x2 = 0.0\n"
  "    spike = 0\n"
  "	CaSR = 0.0025  		:[M]\n"
  "	CaSRCS = 0.0		    :[M]\n"
  "	Ca = 1e-10		    :[M]\n"
  "	CaT = 0.0				:[M]\n"
  "	AM = 0.0				:[M]\n"
  "	CaB = 0.0				:[M]\n"
  "	c1 = 0.154\n"
  "	c2 = 0.11\n"
  "}\n"
  "\n"
  "\n"
  "BREAKPOINT{\n"
  "	R1 = R1*exp(-dt/tau2)\n"
  "	R2 = R2*exp(-dt/tau2)*exp(-dt/tau1)\n"
  "	R = CaSR*Rmax*(R1 - R2)\n"
  "	:printf(\"R = %g\", R)\n"
  "	rate (CaT, AM, t)\n"
  "    SOLVE states_force METHOD derivimplicit\n"
  "    F = Fmax*x1\n"
  "	spike = 0\n"
  "	A = AM^alpha\n"
  "	:printf(\"CaSR = %g\", CaSR)\n"
  "}\n"
  "\n"
  "DERIVATIVE states_force{\n"
  "    x1' = x2\n"
  "    x2' = -2/Tc*x2 - 1/(Tc*Tc)*x1 + CaT/0.0001/Tc    	\n"
  "	CaSR' = - R + U(Ca) -k1*CS0*CaSR + (k1*CaSR+k2)*CaSRCS \n"
  "	CaSRCS' = k1*CS0*CaSR - (k1*CaSR+k2)*CaSRCS\n"
  "	Ca' = -k5*T0*Ca + (k5*Ca+k6)*CaT + R - U(Ca) - k3*B0*Ca +(k3*Ca+k4)*CaB \n"
  "	CaB' = k3*B0*Ca -(k3*Ca+k4)*CaB\n"
  "	CaT' = k5*T0*Ca - (k5*Ca+k6)*CaT\n"
  "	AM' = 0 :(AMinf -AM)/AMtau\n"
  "	c1' = 0 :(c1inf - c1)/tauc1\n"
  "	c2' = 0 :(c2inf - c2)/tauc2\n"
  "	\n"
  "}\n"
  "\n"
  "PROCEDURE rate(CaT (M), AM (M), t(ms)) {\n"
  "	k5 = phi(5)*k5i\n"
  "	k6 = k6i/(1 + SF_AM*AM)\n"
  "	AMinf = 0.5*(1+tanh((CaT/T0-c1)/c2))\n"
  "	AMtau = c3/(cosh((CaT/T0-c4)/(2*c5)))\n"
  "	c1inf = c1n1*(1+tanh((CaT/T0-c1n2)/c1n3))+c1i\n"
  "	c2inf = c2n1*(1+tanh((CaT/T0-c2n2)/c2n3))+c2i\n"
  "}\n"
  "\n"
  "FUNCTION U(x) {\n"
  "	if (x >= 0) {U = Umax*(x^2*k^2/(1+x*k+x^2*k^2))^2}\n"
  "	else {U = 0}\n"
  "}\n"
  "\n"
  "FUNCTION phi(x) {\n"
  "	if (x <= 5) {phi = phi1*x + phi2}\n"
  "	else {phi = phi3*x + phi4}\n"
  "}\n"
  "\n"
  "NET_RECEIVE (weight) {\n"
  "	spike = 2.7182818/dt\n"
  "    R1 = R1 + 1\n"
  " 	R2 = R2 + 1\n"
  "	:temp = (AMinf -AM)/AMtau\n"
  "	:printf(\"%g \", temp)\n"
  "	:printf(\"R1 = %g, R2 = %g, R = %g, CaSR = %g\", R1, R2, R, CaSR)\n"
  "}\n"
  "\n"
  ;
#endif
