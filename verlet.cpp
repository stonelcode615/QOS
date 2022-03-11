// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "verlet.h"

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "pair.h"
#include "timer.h"
#include "update.h"

#include <cstring>

#include "universe.h" //stone Mar. 1 2022
#include "group.h"    //stone Mar. 2 2022

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

Verlet::Verlet(LAMMPS *lmp, int narg, char **arg) : Integrate(lmp, narg, arg) 
{
  printf("stone, in Verlet constructor\n");
}



/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

void Verlet::init()
{
  Integrate::init();

  // warn if no fixes doing time integration

  bool do_time_integrate = false;
  for (const auto &fix : modify->get_fix_list())
    if (fix->time_integrate) do_time_integrate = true;

  if (!do_time_integrate && (comm->me == 0))
    error->warning(FLERR,"No fixes with time integration, atoms won't move");

  // virial_style:
  // VIRIAL_PAIR if computed explicitly in pair via sum over pair interactions
  // VIRIAL_FDOTR if computed implicitly in pair by
  //   virial_fdotr_compute() via sum over ghosts

  if (force->newton_pair) virial_style = VIRIAL_FDOTR;
  else virial_style = VIRIAL_PAIR;

  // setup lists of computes for global and per-atom PE and pressure

  ev_setup();

  // detect if fix omp is present for clearing force arrays

  if (modify->get_fix_by_id("package_omp")) external_force_clear = 1;

  // set flags for arrays to clear in force_clear()

  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;

  // orthogonal vs triclinic simulation box

  triclinic = domain->triclinic;
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void Verlet::setup(int flag)
{
  if (comm->me == 0 && screen) {
    fputs("Setting up Verlet run ...\n",screen);
    if (flag) {
      fmt::print(screen,"  Unit style    : {}\n"
                        "  Current step  : {}\n"
                        "  Time step     : {}\n",
                 update->unit_style,update->ntimestep,update->dt);
      timer->print_timeout(screen);
    }
  }

  if (lmp->kokkos)
    error->all(FLERR,"KOKKOS package requires run_style verlet/kk");

  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  atom->setup();
  modify->setup_pre_exchange();
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) atom->sort();
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  domain->image_check();
  domain->box_too_small_check();
  modify->setup_pre_neighbor();
  neighbor->build(1);
  modify->setup_post_neighbor();
  neighbor->ncalls = 0;

  // compute all forces

  force->setup();
  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  output->setup(flag);
  update->setupflag = 0;
}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void Verlet::setup_minimal(int flag)
{
  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  if (flag) {
    modify->setup_pre_exchange();
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;
  }

  // compute all forces

  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  update->setupflag = 0;
}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */
// head of run()
void Verlet::run(int n)
{
  bigint ntimestep;
  int nflag,sortflag;

  int n_post_integrate = modify->n_post_integrate;
  int n_pre_exchange = modify->n_pre_exchange;
  int n_pre_neighbor = modify->n_pre_neighbor;
  int n_post_neighbor = modify->n_post_neighbor;
  int n_pre_force = modify->n_pre_force;
  int n_pre_reverse = modify->n_pre_reverse;
  int n_post_force = modify->n_post_force;
  int n_end_of_step = modify->n_end_of_step;

  if (atom->sortfreq > 0) sortflag = 1;
  else sortflag = 0;

  int me;
  MPI_Comm_rank(world,&me); // stone
 
  for (int i = 0; i < n; i++) {
    if (timer->check_timeout(i)) {
      update->nsteps = i;
      break;
    }


    ntimestep = ++update->ntimestep;
    ev_set(ntimestep);

    // initial time integration

    // run_stone to compute the average position fix_stone_Pimd
    
    double *stone_x1 = new double [atom->natoms*3];
    double *stone_ave_x1 = new double [atom->natoms*3];

    printf("stone, run step %d, process = %d, partition = %d nlocal = %d\n",
            i, universe->me,universe->iworld,atom->nlocal);
    if (modify->fix_stone != 0) {
      run_stone(1); // call run_stone()
      modify->fix_run_stone = 0;
    }
    // convert 2d array atom-> into 1d array stone_x1 for MPI_Allreduce sum
    for (int ndx0 = 0; ndx0 < atom->natoms; ndx0++) {
      for (int ndx1 = 0; ndx1 < 3; ndx1++) {
        stone_x1[3*ndx0+ndx1] = atom->x[ndx0][ndx1];
      }
    }
    //double stone_a = (atom->nlocal==4) ? 10 : 15;
    //double stone_global = 0.00;
    //MPI_Allreduce(&stone_a, &stone_global, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
    //printf("stone, run       step %d, process = %d, partition = %d,stone_a = %f stone_global = %f\n",
    //       i, universe->me,universe->iworld,stone_a, stone_global);
    MPI_Allreduce(stone_x1, stone_ave_x1, atom->natoms*3,MPI_DOUBLE,MPI_SUM,universe->uworld);
    for (int ndx0 = 0; ndx0 < atom->natoms; ndx0++) {
      for (int ndx1 = 0; ndx1 < 3; ndx1++) {
        atom->x[ndx0][ndx1] = stone_ave_x1[3*ndx0+ndx1];
      }
    }
    for (int ndx = 0; ndx < atom->natoms; ndx++) {
      printf("stone, run           step %d, process = %d, partition = %d,x->atom = %d, x: %f %f %f\n",
             i, universe->me,universe->iworld,ndx,atom->x[ndx][0],atom->x[ndx][1],atom->x[ndx][2]);
    }

    timer->stamp();
    modify->initial_integrate(vflag);
    if (n_post_integrate) modify->post_integrate();
    timer->stamp(Timer::MODIFY);

    // regular communication vs neighbor list rebuild

    nflag = neighbor->decide();

    if (nflag == 0) {
      timer->stamp();
      comm->forward_comm();
      timer->stamp(Timer::COMM);
    } else {
      if (n_pre_exchange) {
        timer->stamp();
        modify->pre_exchange();
        timer->stamp(Timer::MODIFY);
      }
      if (triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();
      if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
      }
      timer->stamp();
      comm->exchange();
      if (sortflag && ntimestep >= atom->nextsort) atom->sort();
      comm->borders();
      if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
      timer->stamp(Timer::COMM);
      if (n_pre_neighbor) {
        modify->pre_neighbor();
        timer->stamp(Timer::MODIFY);
      }
      neighbor->build(1);
      timer->stamp(Timer::NEIGH);
      if (n_post_neighbor) {
        modify->post_neighbor();
        timer->stamp(Timer::MODIFY);
      }
    }

    // force computations
    // important for pair to come before bonded contributions
    // since some bonded potentials tally pairwise energy/virial
    // and Pair:ev_tally() needs to be called before any tallying


    force_clear();

    timer->stamp();

    if (n_pre_force) {
      modify->pre_force(vflag);
      timer->stamp(Timer::MODIFY);
    }

    if (pair_compute_flag) {
      force->pair->compute(eflag,vflag);
      timer->stamp(Timer::PAIR);
    }

    if (atom->molecular != Atom::ATOMIC) {
      if (force->bond) force->bond->compute(eflag,vflag);
      if (force->angle) force->angle->compute(eflag,vflag);
      if (force->dihedral) force->dihedral->compute(eflag,vflag);
      if (force->improper) force->improper->compute(eflag,vflag);
      timer->stamp(Timer::BOND);
    }

    if (kspace_compute_flag) {
      force->kspace->compute(eflag,vflag);
      timer->stamp(Timer::KSPACE);
    }

    if (n_pre_reverse) {
      modify->pre_reverse(eflag,vflag);
      timer->stamp(Timer::MODIFY);
    }

    // reverse communication of forces

    if (force->newton) {
      comm->reverse_comm();
      timer->stamp(Timer::COMM);
    }

    // force modifications, final time integration, diagnostics

    //printf("stone, it is in Verlet.cpp n_post_force=%d\n",n_post_force);
    //printf("stone, it is in Verlet.cpp n_end_of_step=%d\n",n_end_of_step);

    if (n_post_force) modify->post_force(vflag);
    modify->final_integrate();

    if (n_end_of_step) {
      modify->end_of_step();
    }
    timer->stamp(Timer::MODIFY);

    // all output

    if (ntimestep == output->next) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }

    delete stone_x1;
    delete stone_ave_x1;

  } // end of run step loop

} //tail of run()

// head of run_stone()
void Verlet::run_stone(int ntimestep_run_stone=2) 
{
  bigint ntimestep;
  int nflag,sortflag;

  int n_post_integrate = modify->n_post_integrate;
  int n_pre_exchange = modify->n_pre_exchange;
  int n_pre_neighbor = modify->n_pre_neighbor;
  int n_post_neighbor = modify->n_post_neighbor;
  int n_pre_force = modify->n_pre_force;
  int n_pre_reverse = modify->n_pre_reverse;
  int n_post_force = modify->n_post_force;
  int n_end_of_step = modify->n_end_of_step;
  
  double **run_stone_ave_x = new double*[atom->natoms];
  for (int ndx0=0; ndx0 < atom->natoms; ndx0++)
    run_stone_ave_x[ndx0] = new double [3];
  for (int ndx0=0; ndx0 < atom->natoms; ndx0++) {
    run_stone_ave_x[ndx0][0] = 0.000;
    run_stone_ave_x[ndx0][1] = 0.000;
    run_stone_ave_x[ndx0][2] = 0.000;
  }
  
  //double run_stone_ave_x[10000][3] = {0.000};

  if (atom->sortfreq > 0) sortflag = 1;
  else sortflag = 0;

  // find oscillator group
  //int igroup = group->find("drude");
  //int groupbit = group->bitmask[igroup];
  //printf("stone run, drude group : igroup = %d groupbit = %d\n",igroup,groupbit);

  for (int i = 0; i < ntimestep_run_stone; i++) {
    //if (timer->check_timeout(i)) {
    //  update->nsteps = i;
    //  break;
    //}

    ntimestep = ++update->ntimestep;
    modify->fix_run_stone = 1;
    ev_set(ntimestep);
    --update->ntimestep; // stone


    // initial time integration in run_stone

    timer->stamp();
    modify->initial_integrate(vflag);

    for (int ndx = 0; ndx < atom->natoms; ndx++) {
      double stonex0 = atom->x[ndx][0];
      double stonex1 = atom->x[ndx][1];
      double stonex2 = atom->x[ndx][2];
      printf("stone00,  run_stone, step %d, process = %d, partition = %d,atom    = %d, x: %f %f %f\n",
              i, universe->me,universe->iworld,ndx,stonex0,stonex1,stonex2);
    }

    if (n_post_integrate) modify->post_integrate();
    timer->stamp(Timer::MODIFY);

    // regular communication vs neighbor list rebuild

    nflag = neighbor->decide();

    if (nflag == 0) {
      timer->stamp();
      comm->forward_comm();
      timer->stamp(Timer::COMM);
    } else {
      if (n_pre_exchange) {
        timer->stamp();
        modify->pre_exchange();
        timer->stamp(Timer::MODIFY);
      }
      if (triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();
      if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
      }
      timer->stamp();
      comm->exchange();
      if (sortflag && ntimestep >= atom->nextsort) atom->sort();
      comm->borders();
      if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
      timer->stamp(Timer::COMM);
      if (n_pre_neighbor) {
        modify->pre_neighbor();
        timer->stamp(Timer::MODIFY);
      }
      neighbor->build(1);
      timer->stamp(Timer::NEIGH);
      if (n_post_neighbor) {
        modify->post_neighbor();
        timer->stamp(Timer::MODIFY);
      }
    }

    // force computations

    force_clear();

    timer->stamp();

    if (n_pre_force) {
      modify->pre_force(vflag);
      timer->stamp(Timer::MODIFY);
    }

    if (pair_compute_flag) {
      force->pair->compute(eflag,vflag);
      timer->stamp(Timer::PAIR);
    }

    if (kspace_compute_flag) {
      force->kspace->compute(eflag,vflag);
      timer->stamp(Timer::KSPACE);
    }

    if (n_pre_reverse) {
      modify->pre_reverse(eflag,vflag);
      timer->stamp(Timer::MODIFY);
    }

    // reverse communication of forces

    if (force->newton) {
      comm->reverse_comm();
      timer->stamp(Timer::COMM);
    }

    // force modifications, final time integration, diagnostics

    if (n_post_force) modify->post_force(vflag);
    modify->final_integrate();
    
    if (n_end_of_step) {
      modify->end_of_step();
    }

    timer->stamp(Timer::MODIFY);

    // all output

    if (ntimestep == output->next) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
    
    // stone store and average x position
    for (int ndx = 0; ndx < atom->natoms; ndx++) {
      double stonex0 = atom->x[ndx][0];
      double stonex1 = atom->x[ndx][1];
      double stonex2 = atom->x[ndx][2];
      //printf("stone,    run_stone, step %d, process = %d, partition = %d,nprocs    = %d\n",
      //        i, universe->me,universe->iworld,universe->nprocs);
      //printf("stone,    run_stone, step %d, process = %d, partition = %d,atom    = %d, x: %f %f %f\n",
      //        i, universe->me,universe->iworld,ndx,stonex0,stonex1,stonex2);
      //run_stone_ave_x[ndx][0] += atom->x[ndx][0]/ntimestep_run_stone/universe->nprocs;
      //run_stone_ave_x[ndx][1] += atom->x[ndx][1]/ntimestep_run_stone/universe->nprocs;
      //run_stone_ave_x[ndx][2] += atom->x[ndx][2]/ntimestep_run_stone/universe->nprocs;
      if ( atom->nlocal != 0 ) {
        run_stone_ave_x[ndx][0] += atom->x[ndx][0]/ntimestep_run_stone/universe->nworlds;
        run_stone_ave_x[ndx][1] += atom->x[ndx][1]/ntimestep_run_stone/universe->nworlds;
        run_stone_ave_x[ndx][2] += atom->x[ndx][2]/ntimestep_run_stone/universe->nworlds;
      } else {
        run_stone_ave_x[ndx][0]  = 0.000;
        run_stone_ave_x[ndx][1]  = 0.000;
        run_stone_ave_x[ndx][2]  = 0.000;
      }
    }
  } // end of run_stone step loop

  for (int ndx = 0; ndx < atom->natoms; ndx++) {
    atom->x[ndx][0] = run_stone_ave_x[ndx][0];
    atom->x[ndx][1] = run_stone_ave_x[ndx][1];
    atom->x[ndx][2] = run_stone_ave_x[ndx][2];
    /*if ( atom->nlocal != 0 ) {
      atom->x[ndx][0] = run_stone_ave_x[ndx][0];
      atom->x[ndx][1] = run_stone_ave_x[ndx][1];
      atom->x[ndx][2] = run_stone_ave_x[ndx][2];
    } else {
      atom->x[ndx][0] = 0.000;
      atom->x[ndx][1] = 0.000;
      atom->x[ndx][2] = 0.000;
    } */
    delete[] run_stone_ave_x[ndx];
  }
} // tail of run_stone()


/* ---------------------------------------------------------------------- */

void Verlet::cleanup()
{
  modify->post_run();
  domain->box_too_small_check();
  update->update_time();
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void Verlet::force_clear()
{
  size_t nbytes;

  if (external_force_clear) return;

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  int nlocal = atom->nlocal;
  //printf("stone, in Verlet::force_clear, nlocal = atom->nlocal = %d\n",nlocal);

  if (neighbor->includegroup == 0) {
    nbytes = sizeof(double) * nlocal;
    if (force->newton) nbytes += sizeof(double) * atom->nghost;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    nbytes = sizeof(double) * atom->nfirst;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

    if (force->newton) {
      nbytes = sizeof(double) * atom->nghost;

      if (nbytes) {
        memset(&atom->f[nlocal][0],0,3*nbytes);
        if (torqueflag) memset(&atom->torque[nlocal][0],0,3*nbytes);
        if (extraflag) atom->avec->force_clear(nlocal,nbytes);
      }
    }
  }
}
