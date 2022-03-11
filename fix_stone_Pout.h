#ifdef FIX_CLASS
FixStyle(stonePout,FixStonePout);
#else

#include "fix.h"
#include <string>

using namespace std;

namespace LAMMPS_NS{
    class FixStonePout: public Fix {
    public:
        string  Printed;
        FixStonePout(class LAMMPS *, int, char **);
        int setmask() override;
        void end_of_step() override;
    }; // end claim class FixStonePout


} // end of namespace LAMMPS_NS

#endif
