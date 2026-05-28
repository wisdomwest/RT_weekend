Check <https://github.com/RayTracing/raytracing.github.io/wiki/Further-Readings>

g++ -O3 -std=c++17 -fopenmp -march=native -DNDEBUG main.cc -o main

g++ -O3 -std=c++17 -fopenmp -mavx2 -mfma -ffast-math -DNDEBUG main.cc -o main
   OMP_NUM_THREADS=8 ./main > 8_threads.ppm
   OMP_NUM_THREADS=4 ./main > 4_cores.ppm
