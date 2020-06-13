"""
cython declarations for the Mersenne Twister RNG
http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
"""

cdef extern from "mt19937ar/mt19937ar.h":

    # initialize mt[N] by setting a seed
    void init_genrand(unsigned long s) nogil

    # generate a random uint32 number
    unsigned long genrand_int32() nogil

    # generate a random [0.0, 1.0) real number
    double genrand_real2() nogil

