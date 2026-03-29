#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _GammaStim_reg(void);
extern void _Gfluctdv_reg(void);
extern void _L_Ca_inact_reg(void);
extern void _caL_reg(void);
extern void _constant_reg(void);
extern void _dummy_reg(void);
extern void _gammapointprocess_reg(void);
extern void _gh_reg(void);
extern void _izap_reg(void);
extern void _kdrRL_reg(void);
extern void _mAHP_reg(void);
extern void _motoneuron_reg(void);
extern void _muscle_unit_reg(void);
extern void _muscle_unit_calcium_reg(void);
extern void _na3rp_reg(void);
extern void _napp_reg(void);
extern void _naps_reg(void);
extern void _nsloc_reg(void);
extern void _vecevent_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"./GammaStim.mod\"");
    fprintf(stderr, " \"./Gfluctdv.mod\"");
    fprintf(stderr, " \"./L_Ca_inact.mod\"");
    fprintf(stderr, " \"./caL.mod\"");
    fprintf(stderr, " \"./constant.mod\"");
    fprintf(stderr, " \"./dummy.mod\"");
    fprintf(stderr, " \"./gammapointprocess.mod\"");
    fprintf(stderr, " \"./gh.mod\"");
    fprintf(stderr, " \"./izap.mod\"");
    fprintf(stderr, " \"./kdrRL.mod\"");
    fprintf(stderr, " \"./mAHP.mod\"");
    fprintf(stderr, " \"./motoneuron.mod\"");
    fprintf(stderr, " \"./muscle_unit.mod\"");
    fprintf(stderr, " \"./muscle_unit_calcium.mod\"");
    fprintf(stderr, " \"./na3rp.mod\"");
    fprintf(stderr, " \"./napp.mod\"");
    fprintf(stderr, " \"./naps.mod\"");
    fprintf(stderr, " \"./nsloc.mod\"");
    fprintf(stderr, " \"./vecevent.mod\"");
    fprintf(stderr, "\n");
  }
  _GammaStim_reg();
  _Gfluctdv_reg();
  _L_Ca_inact_reg();
  _caL_reg();
  _constant_reg();
  _dummy_reg();
  _gammapointprocess_reg();
  _gh_reg();
  _izap_reg();
  _kdrRL_reg();
  _mAHP_reg();
  _motoneuron_reg();
  _muscle_unit_reg();
  _muscle_unit_calcium_reg();
  _na3rp_reg();
  _napp_reg();
  _naps_reg();
  _nsloc_reg();
  _vecevent_reg();
}

#if defined(__cplusplus)
}
#endif
