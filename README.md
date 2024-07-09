# SIMD--Black-Scholes-Monte-Carlo
Black-Scholes model with Monte-Carlo simulations. Optimized through SIMD
Still a work in progress. Will try to optimize process anyway I can. 

To run, paste the following line into the terminal

g++ -std=c++23 -mavx2 -mfma -O3 -o pricer main.cpp && ./pricer
