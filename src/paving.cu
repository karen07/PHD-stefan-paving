#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

using namespace std;

#define Sq(x) ((x) * (x))

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ double delt_d;
__device__ double freeze_temp_d;

__device__ double briq_temp_d;
__device__ double goo_temp_d;
__device__ double bound_temp_d;

__device__ int sphere_count_x_d;
__device__ int sphere_count_y_d;
__device__ int sphere_count_z_d;

__device__ int problem_size_x_d;
__device__ int problem_size_y_d;
__device__ int problem_size_z_d;

__device__ double dx_d;
__device__ double dt_d;

__device__ int get_pos(int i, int j, int k)
{
    return i + j * problem_size_x_d + k * problem_size_x_d * problem_size_y_d;
}

__device__ double c_ro(double t)
{
    double c_frosen = 1600;
    double ro_frosen = 1200;

    double c_melt = 1900;
    double ro_melt = 1200;

    double L = 330000 * 0.1;

    if (t < freeze_temp_d - delt_d)
        return c_frosen * ro_frosen;

    if (t >= freeze_temp_d - delt_d && t < freeze_temp_d)
        return (c_frosen + L / delt_d / 2) * ro_frosen;

    if (t >= freeze_temp_d && t < freeze_temp_d + delt_d)
        return (c_melt + L / delt_d / 2) * ro_melt;

    if (t >= freeze_temp_d + delt_d)
        return c_melt * ro_melt;

    return 0;
}

__device__ double k(double t)
{
    double k_frosen = 0.92;

    double k_melt = 0.72;

    double scale = 3600;

    if (t < freeze_temp_d)
        return k_frosen * scale;

    if (t >= freeze_temp_d)
        return k_melt * scale;

    return 0;
}

__global__ void init(double *in, double *out)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    double R = problem_size_x_d / sphere_count_x_d / 2.0;

    if (idx < problem_size_x_d && idy < problem_size_y_d && idz < problem_size_z_d) {
        in[get_pos(idx, idy, idz)] = briq_temp_d;
        out[get_pos(idx, idy, idz)] = briq_temp_d;
    }

    if (idx > 0 && idx < problem_size_x_d - 1 && idy > 0 && idy < problem_size_y_d - 1 && idz > 0 &&
        idz < problem_size_z_d - 1) {
        in[get_pos(idx, idy, idz)] = goo_temp_d;
        out[get_pos(idx, idy, idz)] = goo_temp_d;
        for (int i = 0; i < sphere_count_x_d; i++)
            for (int j = 0; j < sphere_count_y_d; j++)
                for (int k = 0; k < sphere_count_z_d; k++)
                    if (Sq(idx - R * (1 + 2 * i)) + Sq(idy - R * (1 + 2 * j)) +
                            Sq(idz - R * (1 + 2 * k)) <=
                        Sq(R + 1)) {
                        in[get_pos(idx, idy, idz)] = briq_temp_d;
                        out[get_pos(idx, idy, idz)] = briq_temp_d;
                    }
    }
}

__global__ void solve(double *out, double *in)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    if (idx > 0 && idx < problem_size_x_d - 1 && idy > 0 && idy < problem_size_y_d - 1 && idz > 0 &&
        idz < problem_size_z_d - 1) {
        if (idx == 1) {
            in[get_pos(idx - 1, idy, idz)] = bound_temp_d; //in[get_pos(idx, idy, idz)];
        } else if (idx == problem_size_x_d - 2) {
            in[get_pos(idx + 1, idy, idz)] = bound_temp_d; //in[get_pos(idx, idy, idz)];
        } else if (idy == 1) {
            in[get_pos(idx, idy - 1, idz)] = in[get_pos(idx, idy, idz)];
        } else if (idy == problem_size_y_d - 2) {
            in[get_pos(idx, idy + 1, idz)] = in[get_pos(idx, idy, idz)];
        } else if (idz == 1) {
            in[get_pos(idx, idy, idz - 1)] = in[get_pos(idx, idy, idz)];
        } else if (idz == problem_size_z_d - 2) {
            in[get_pos(idx, idy, idz + 1)] = in[get_pos(idx, idy, idz)];
        }
        out[get_pos(idx, idy, idz)] =
            in[get_pos(idx, idy, idz)] +
            dt_d / ((dx_d * dx_d) * c_ro(in[get_pos(idx, idy, idz)])) *
                ((((k(in[get_pos(idx + 1, idy, idz)]) + k(in[get_pos(idx, idy, idz)])) / 2.0) *
                      (in[get_pos(idx + 1, idy, idz)] - in[get_pos(idx, idy, idz)]) -
                  ((k(in[get_pos(idx, idy, idz)]) + k(in[get_pos(idx - 1, idy, idz)])) / 2.0) *
                      (in[get_pos(idx, idy, idz)] - in[get_pos(idx - 1, idy, idz)])) +
                 (((k(in[get_pos(idx, idy + 1, idz)]) + k(in[get_pos(idx, idy, idz)])) / 2.0) *
                      (in[get_pos(idx, idy + 1, idz)] - in[get_pos(idx, idy, idz)]) -
                  ((k(in[get_pos(idx, idy, idz)]) + k(in[get_pos(idx, idy - 1, idz)])) / 2.0) *
                      (in[get_pos(idx, idy, idz)] - in[get_pos(idx, idy - 1, idz)])) +
                 (((k(in[get_pos(idx, idy, idz + 1)]) + k(in[get_pos(idx, idy, idz)])) / 2.0) *
                      (in[get_pos(idx, idy, idz + 1)] - in[get_pos(idx, idy, idz)]) -
                  ((k(in[get_pos(idx, idy, idz)]) + k(in[get_pos(idx, idy, idz - 1)])) / 2.0) *
                      (in[get_pos(idx, idy, idz)] - in[get_pos(idx, idy, idz - 1)])));
    }
}

static void swap4(float *v)
{
    char in[4], out[4];
    memcpy(in, v, 4);
    out[0] = in[3];
    out[1] = in[2];
    out[2] = in[1];
    out[3] = in[0];
    memcpy(v, out, 4);
}

int main()
{
    int max_size_x = 500;
    int max_size_y = 500;
    int max_size_z = 500;

    double dx = 0.01;
    double dt = 0.001;
    double delt = 0.1;

    double freeze_temp = 0;

    int sphere_count_x = 8;
    int sphere_count_y = 8;
    int sphere_count_z = 8;

    double centre_line[500];

    gpuErrchk(cudaMemcpyToSymbol(sphere_count_x_d, &sphere_count_x, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(sphere_count_y_d, &sphere_count_y, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(sphere_count_z_d, &sphere_count_z, sizeof(int)));

    gpuErrchk(cudaMemcpyToSymbol(dt_d, &dt, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(dx_d, &dx, sizeof(double)));

    gpuErrchk(cudaMemcpyToSymbol(freeze_temp_d, &freeze_temp, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(delt_d, &delt, sizeof(double)));

    double *heat_array_old;
    gpuErrchk(cudaMalloc((void **)&heat_array_old,
                         max_size_x * max_size_y * max_size_z * sizeof(double)));

    double *heat_array_now;
    gpuErrchk(cudaMalloc((void **)&heat_array_now,
                         max_size_x * max_size_y * max_size_z * sizeof(double)));

    double *heat_array_cpu =
        (double *)malloc(max_size_x * max_size_y * max_size_z * sizeof(double));

    ofstream file_out("out.txt");

    for (int i = -30; i <= -30; i += 5) { //Temperature degrees
        for (int j = 30; j <= 30; j += 5) { //Size cm
            for (int k = 10; k <= 10; k += 5) { //Pulp temperature
                for (int d = -6; d <= -6; d += 3) { //Wall temperature
                    sphere_count_x = ceil(200.0 / j);
                    gpuErrchk(cudaMemcpyToSymbol(sphere_count_x_d, &sphere_count_x, sizeof(int)));

                    int problem_size_x = sphere_count_x * j * 0.01 / dx + 1;
                    gpuErrchk(cudaMemcpyToSymbol(problem_size_x_d, &problem_size_x, sizeof(int)));

                    int problem_size_y = sphere_count_y * j * 0.01 / dx + 1;
                    gpuErrchk(cudaMemcpyToSymbol(problem_size_y_d, &problem_size_y, sizeof(int)));

                    int problem_size_z = sphere_count_z * j * 0.01 / dx + 1;
                    gpuErrchk(cudaMemcpyToSymbol(problem_size_z_d, &problem_size_z, sizeof(int)));

                    double briq_temp = i;
                    gpuErrchk(cudaMemcpyToSymbol(briq_temp_d, &briq_temp, sizeof(double)));

                    double goo_temp = k;
                    gpuErrchk(cudaMemcpyToSymbol(goo_temp_d, &goo_temp, sizeof(double)));

                    double bound_temp = d;
                    gpuErrchk(cudaMemcpyToSymbol(bound_temp_d, &bound_temp, sizeof(double)));

                    dim3 threadsPerBlock(8, 8, 8);
                    dim3 numBlocks(problem_size_x / threadsPerBlock.x + 1,
                                   problem_size_y / threadsPerBlock.y + 1,
                                   problem_size_z / threadsPerBlock.z + 1);

                    file_out << "Briquette temperature:" << briq_temp << "C "
                             << "Briquette side size:" << j * 0.01 << "m "
                             << "Pulp temperature:" << goo_temp << "C "
                             << "Wall temperature:" << bound_temp << "C" << endl
                             << "Time(clock)    Maximum pulp temperature(C)" << endl;

                    init<<<numBlocks, threadsPerBlock>>>(heat_array_old, heat_array_now);

                    for (int time = 0; time <= 168 / dt; time++) {
                        solve<<<numBlocks, threadsPerBlock>>>(heat_array_now, heat_array_old);

                        double *tmp = heat_array_now;
                        heat_array_now = heat_array_old;
                        heat_array_old = tmp;

                        if (time % ((int)(0.1 / dt)) == 0) {
                            gpuErrchk(cudaMemcpy(
                                &centre_line,
                                &heat_array_old[0 + problem_size_y / 2 * problem_size_x +
                                                problem_size_z / 2 * problem_size_x *
                                                    problem_size_y],
                                problem_size_x * sizeof(double), cudaMemcpyDeviceToHost));
                            double max_value = centre_line[0];
                            for (int i = 0; i < problem_size_x; i++) {
                                if (max_value < centre_line[i])
                                    max_value = centre_line[i];
                            }
                            file_out << time * dt << "    " << max_value << endl;

                            gpuErrchk(cudaMemcpy(heat_array_cpu, heat_array_old,
                                                 problem_size_x * problem_size_y * problem_size_z *
                                                     sizeof(double),
                                                 cudaMemcpyDeviceToHost));
                            char out_string[100];
                            sprintf(out_string, "plot/result_%d.vtk", time);
                            ofstream out(out_string, ios::out | ios::binary);
                            out << "# vtk DataFile Version 2.0" << endl;
                            out << "Heat" << endl;
                            out << "BINARY" << endl;
                            out << "DATASET STRUCTURED_POINTS" << endl;
                            out << "DIMENSIONS " << problem_size_x << " " << problem_size_y << " "
                                << problem_size_z << endl;
                            out << "ASPECT_RATIO 1 1 1" << endl;
                            out << "ORIGIN 0 0 0" << endl;
                            out << "POINT_DATA " << problem_size_x * problem_size_y * problem_size_z
                                << endl;
                            out << "SCALARS heat float 1" << endl;
                            out << "LOOKUP_TABLE default" << endl;
                            for (int k = 0; k < problem_size_z; k++) {
                                for (int j = 0; j < problem_size_y; j++) {
                                    for (int i = 0; i < problem_size_x; i++) {
                                        float tmp =
                                            heat_array_cpu[i + j * problem_size_x +
                                                           k * problem_size_x * problem_size_y];
                                        swap4(&tmp);
                                        out.write((char *)(&tmp), sizeof(float));
                                    }
                                }
                            }
                            out << endl;
                            out.close();

                            if (max_value < -0.1)
                                break;
                        }
                    }

                    file_out << endl;
                }
            }
        }
    }

    return 0;
}
