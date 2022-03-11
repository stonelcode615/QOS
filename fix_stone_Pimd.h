
#ifdef FIX_CLASS
// clang-format off
FixStyle(stonePimd,FixStonePimd);
// clang-format on
#else

#ifndef FIX_STONE_PIMD_H
#define FIX_STONE_PIMD_H

#include "fix.h"

namespace LAMMPS_NS {

class FixStonePimd : public Fix {
 public:
  FixStonePimd(class LAMMPS *, int, char **);
  virtual ~FixStonePimd();

  int setmask();

  void init();
  void setup(int);
  void post_force(int);
  void initial_integrate(int);
  void final_integrate();

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  int pack_restart(int, double *);
  void unpack_restart(int, int);
  int maxsize_restart();
  int size_restart(int);
  double compute_vector(int);

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);

  int method;
  int np;
  double inverse_np;

  /* ring-polymer model */

  double omega_np, fbond, spring_energy, sp;
  int x_last, x_next;

  void spring_force();

  /* fictitious mass */

  double fmass, *mass;

  /* inter-partition communication */

  int max_nsend;
  tagint *tag_send;
  double *buf_send;

  int max_nlocal;
  double *buf_recv, **buf_beads;

  int size_plan;
  int *plan_send, *plan_recv;
  double **comm_ptr;

  void comm_init();
  void comm_exec(double **);

  /* normal-mode operations */

  double *lam, **M_x2xp, **M_xp2x, **M_f2fp, **M_fp2f;
  int *mode_index;

  void nmpimd_init();
  void nmpimd_fill(double **);
  void nmpimd_transform(double **, double **, double *);

  /* Nose-hoover chain integration */

  int nhc_offset_one_1, nhc_offset_one_2;
  int nhc_size_one_1, nhc_size_one_2;
  int nhc_nchain;
  bool nhc_ready;
  double nhc_temp, dtv, dtf, t_sys;

  double **nhc_eta;        /* coordinates of NH chains for ring-polymer beads */
  double **nhc_eta_dot;    /* velocities of NH chains                         */
  double **nhc_eta_dotdot; /* acceleration of NH chains                       */
  double **nhc_eta_mass;   /* mass of NH chains                               */

  void nhc_init();
  void nhc_update_v();
  void nhc_update_x();

  /*  Stone Feb 22 2022 */
  double stone_step_size;
  int    stone_step_numb;
};

}    // namespace LAMMPS_NS

#endif
#endif
