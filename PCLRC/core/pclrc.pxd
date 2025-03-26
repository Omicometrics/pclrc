cdef:
    void clr(float[:, ::1] x, float q, float[:, ::1] corr_x,
             float[:, ::1] b_adjx, float[::1] corr_counts, float[:, ::1] xt)
