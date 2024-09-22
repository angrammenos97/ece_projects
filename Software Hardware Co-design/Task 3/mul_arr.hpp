#pragma once

// WARNING: ln\lm\lp must be greater or equal to 4
#define ln 4
#define lm 6
#define lp 4

#define n (1 << ln)
#define m (1 << lm)
#define p (1 << lp)

const unsigned int n_len = n;
const unsigned int m_len = m;
const unsigned int p_len = p;
