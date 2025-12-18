/**
 * HPSparseLU_Corrected.cpp
 * 修复了数学逻辑错误的并行LU分解
 * * 核心修复:
 * 1. 使用 Right-Looking 算法更新整个右下角子矩阵
 * 2. 修正了 L 和 U 的更新逻辑
 * 3. 移除了不必要的锁 (Col-wise 并行天然无竞争)
 * * 编译: g++ -O3 -fopenmp -march=native -I/usr/include/eigen3 HPSparseLU_Corrected.cpp -o hplu_fixed
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

typedef SparseMatrix<double, ColMajor> SpMat;
typedef Triplet<double> T;

class HPSparseLU {
private:
    int n;
    PermutationMatrix<Dynamic, Dynamic, int> perm; 
    SpMat L, U;
    bool is_factorized = false;

public:
    HPSparseLU() = default;
    HPSparseLU(int size) : n(size) {
        perm.resize(size);
        L.resize(size, size);
        U.resize(size, size);
    }

    void factorize(const SpMat& A_input) {
        // 1. Ordering (AMD) 减少 Fill-in
        AMDOrdering<int> ordering;
        ordering(A_input, perm); 
        
        // 排列矩阵 A_perm = P * A * P^T (这里简化为 P*A 既然我们要解 P*A = LU)
        // 通常 SparseLU 处理 A_perm = P_r * A * P_c，这里简化处理只重排列/行以聚集非零元
        SpMat A_perm;
        A_perm = A_input.twistedBy(perm);
        A_perm.makeCompressed();

        // 2. 转换为密集列向量组 (Dense Column Buffer)
        // 这一步虽然消耗内存，但对于 N=1000 这种规模，能极大利用 CPU 缓存和向量化
        vector<VectorXd> work_mat(n, VectorXd::Zero(n));
        
        for (int k = 0; k < n; ++k) {
            for (SpMat::InnerIterator it(A_perm, k); it; ++it) {
                work_mat[k](it.row()) = it.value();
            }
        }

        cout << "Start Parallel Numerical Factorization on " << omp_get_max_threads() << " threads..." << endl;
        auto start = std::chrono::high_resolution_clock::now();

        // 3. 数值分解 (Right-Looking LU)
        // 遍历每一列 k 作为主元列
        for (int k = 0; k < n; ++k) {
            
            // --- A. 处理主元列 (单线程) ---
            double diag = work_mat[k](k);
            
            // 简单的静态 Pivot 修正 (防止除以0)
            if (std::abs(diag) < 1e-15) {
                diag = (diag < 0) ? -1e-15 : 1e-15;
                work_mat[k](k) = diag; // 回写修正值
            }

            // 归一化当前列的下三角部分 (计算 L 的第 k 列)
            // L_ik = A_ik / U_kk
            double inv_diag = 1.0 / diag;
            for (int i = k + 1; i < n; ++i) {
                work_mat[k](i) *= inv_diag;
            }

            // --- B. 并行更新右下角子矩阵 (Schur Complement) ---
            // 更新后续所有列 j > k
            // A_ij = A_ij - L_ik * U_kj
            #pragma omp parallel for schedule(dynamic, 16)
            for (int j = k + 1; j < n; ++j) {
                // U_kj 就是当前 work_mat[j](k) 的值 (上三角部分不动)
                double u_val = work_mat[j](k);

                // 稀疏优化: 如果 U_kj 为 0，则该列无需减去 (L_k * 0)
                if (std::abs(u_val) > 1e-15) {
                    // 使用 SIMD 友好的密集向量更新
                    // 只需要更新行号 i > k 的部分 (即 Schur Complement)
                    // 虽然 i <= k 的部分理论上不变，但为了代码简单只循环 i > k
                    
                    // 手动循环或者利用 Eigen 的 segment 操作
                    // work_mat[j].segment(k + 1, n - k - 1) -= work_mat[k].segment(k + 1, n - k - 1) * u_val;
                    
                    // 手写循环通常对编译器优化更友好
                    double* col_j_ptr = work_mat[j].data();
                    const double* col_k_ptr = work_mat[k].data();
                    
                    for (int i = k + 1; i < n; ++i) {
                        col_j_ptr[i] -= col_k_ptr[i] * u_val;
                    }
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "Factorization (Parallel) finished in: " << elapsed.count() << " seconds." << endl;

        // 4. 转换回稀疏格式 (提取 L 和 U)
        cout << "Converting to sparse format..." << endl;
        vector<T> L_triplets, U_triplets;
        L_triplets.reserve(n * 10); // 预估
        U_triplets.reserve(n * 10);

        for (int j = 0; j < n; ++j) {
            // 对角线显式设为 1.0 (L)
            L_triplets.push_back(T(j, j, 1.0));
            
            for (int i = 0; i < n; ++i) {
                double val = work_mat[j](i);
                if (std::abs(val) > 1e-15) {
                    if (i > j) {
                        // 下三角 (严格) -> L
                        L_triplets.push_back(T(i, j, val));
                    } else {
                        // 上三角 (含对角线) -> U
                        U_triplets.push_back(T(i, j, val));
                    }
                }
            }
        }

        L.setFromTriplets(L_triplets.begin(), L_triplets.end());
        U.setFromTriplets(U_triplets.begin(), U_triplets.end());
        
        L.makeCompressed();
        U.makeCompressed();
        is_factorized = true;
    }

    VectorXd solve(const VectorXd& b) {
        if(!is_factorized) {
            cerr << "Error: Matrix not factorized!" << endl;
            return VectorXd::Zero(n);
        }

        // 1. Permute b (Apply P)
        VectorXd b_perm = perm * b;

        // 2. Forward solve: L * y = P * b
        // 使用 Eigen 自带的高效三角求解器
        VectorXd y = L.triangularView<Lower>().solve(b_perm);

        // 3. Backward solve: U * x_perm = y
        VectorXd x_perm = U.triangularView<Upper>().solve(y);

        // 4. Inverse Permute x (Apply P^T)
        VectorXd x = perm.inverse() * x_perm;

        return x;
    }
};