
#include "fix_stone_Pout.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "respa.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

FixStonePout::FixStonePout(LAMMPS *lmp, int narg, char **arg): 
    Fix(lmp, narg, arg)
{
    if (narg < 4) 
        error->all(FLERR, "Illegal fix stone print command");
    nevery = atoi(arg[3]);
    if (nevery <= 0)
        error->all(FLERR,"Illegal fix print nevery ");
    Printed = arg[4];
}

int FixStonePout::setmask()
{
    int mask = 0;
    mask |= FixConst::END_OF_STEP;
    return mask;
}

void FixStonePout::end_of_step()
{
    //printf("Hello, Stone dive into Lammps\n");
    printf(" %s\n", Printed.c_str());
}


















