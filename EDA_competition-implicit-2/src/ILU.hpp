#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace Eigen;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef Eigen::VectorXd Vec;

// ==========================================
// 1. AVX-512 基础内核
// ==========================================

void* aligned_malloc(size_t size) {
    return _mm_malloc(size, 64);
}

void aligned_free(void* ptr) {
    _mm_free(ptr);
}

inline double dot_product_avx512(const double* __restrict__ a, 
                                 const double* __restrict__ b, 
                                 int n) {
    double sum = 0.0;
    #pragma omp parallel reduction(+:sum)
    {
        __m512d vsum = _mm512_setzero_pd();
        #pragma omp for
        for (int i = 0; i < n; i += 8) {
            int remain = n - i;
            if (remain >= 8) {
                __m512d va = _mm512_load_pd(&a[i]);
                __m512d vb = _mm512_load_pd(&b[i]);
                vsum = _mm512_fmadd_pd(va, vb, vsum);
            } else {
                __mmask8 mask = (1 << remain) - 1;
                __m512d va = _mm512_maskz_load_pd(mask, &a[i]);
                __m512d vb = _mm512_maskz_load_pd(mask, &b[i]);
                vsum = _mm512_fmadd_pd(va, vb, vsum);
            }
        }
        sum += _mm512_reduce_add_pd(vsum);
    }
    return sum;
}

inline double norm_avx512(const double* x, int n) {
    return std::sqrt(dot_product_avx512(x, x, n));
}

// z = x + alpha * y (支持原地操作: z 可以等于 x)
inline void axpy_avx512(double* z, 
                        const double* x,
                        double alpha,
                        const double* y, 
                        int n) {
    __m512d valpha = _mm512_set1_pd(alpha);
    #pragma omp parallel for
    for (int i = 0; i < n; i += 8) {
        int remain = n - i;
        if (remain >= 8) {
            __m512d vx = _mm512_load_pd(&x[i]);
            __m512d vy = _mm512_load_pd(&y[i]);
            __m512d vz = _mm512_fmadd_pd(valpha, vy, vx);
            _mm512_store_pd(&z[i], vz);
        } else {
            __mmask8 mask = (1 << remain) - 1;
            __m512d vx = _mm512_maskz_load_pd(mask, &x[i]);
            __m512d vy = _mm512_maskz_load_pd(mask, &y[i]);
            __m512d vz = _mm512_fmadd_pd(valpha, vy, vx);
            _mm512_mask_store_pd(&z[i], mask, vz);
        }
    }
}

inline void scale_avx512(double* __restrict__ x, double alpha, int n) {
    __m512d valpha = _mm512_set1_pd(alpha);
    #pragma omp parallel for
    for (int i = 0; i < n; i += 8) {
        int remain = n - i;
        if (remain >= 8) {
            __m512d vx = _mm512_load_pd(&x[i]);
            _mm512_store_pd(&x[i], _mm512_mul_pd(vx, valpha));
        } else {
            __mmask8 mask = (1 << remain) - 1;
            __m512d vx = _mm512_maskz_load_pd(mask, &x[i]);
            _mm512_mask_store_pd(&x[i], mask, _mm512_mul_pd(vx, valpha));
        }
    }
}

// ==========================================
// 2. 优化的 SpMV
// ==========================================
void spmv_csr_parallel(const SpMat& A, const double* x, double* y) {
    const int outerSize = A.outerSize();
    const int* outerIndexPtr = A.outerIndexPtr();
    const int* innerIndexPtr = A.innerIndexPtr();
    const double* valuePtr = A.valuePtr();

    #pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < outerSize; ++i) {
        double sum = 0.0;
        int start = outerIndexPtr[i];
        int end = outerIndexPtr[i+1];
        
        for (int k = start; k < end; ++k) {
            sum += valuePtr[k] * x[innerIndexPtr[k]];
        }
        y[i] = sum;
    }
}

// ==========================================
// 3. 异步 ILU 预条件子（统一接口）
// ==========================================
class AsyncILU {
private:
    SpMat L_;  // 下三角矩阵
    SpMat U_;  // 上三角矩阵
    Vec D_;    // 对角元素
    int n_;    // 矩阵大小
    int maxIter_;  // 最大迭代次数
    double tol_;   // 收敛容差
    
    // 前向替换求解 L*y = b（原始指针版本）
    void forwardSolve_ptr(const double* b, double* y) const {
        std::memset(y, 0, n_ * sizeof(double));
        for (int i = 0; i < n_; ++i) {
            double sum = 0.0;
            for (SpMat::InnerIterator it(L_, i); it; ++it) {
                sum += it.value() * y[it.col()];
            }
            y[i] = b[i] - sum;
        }
    }
    
    // 对角缩放 D*z = y（原始指针版本）
    void diagonalSolve_ptr(const double* y, double* z) const {
        #pragma omp parallel for
        for (int i = 0; i < n_; ++i) {
            z[i] = y[i] / D_(i);
        }
    }
    
    // 后向替换求解 U*x = z（原始指针版本）
    void backwardSolve_ptr(const double* z, double* x) const {
        std::memset(x, 0, n_ * sizeof(double));
        for (int i = n_ - 1; i >= 0; --i) {
            double sum = 0.0;
            for (SpMat::InnerIterator it(U_, i); it; ++it) {
                if (it.col() > i) {
                    sum += it.value() * x[it.col()];
                }
            }
            x[i] = z[i] - sum;
        }
    }
    
public:
    AsyncILU(int maxIter = 10, double tol = 1e-6) 
        : maxIter_(maxIter), tol_(tol) {}
    
    void compute(const SpMat& A, int numThreads = 4) {
        n_ = A.rows();
        
        // 初始化L, U, D
        L_ = SpMat(n_, n_);
        U_ = SpMat(n_, n_);
        D_ = Vec::Zero(n_);
        
        // 预分配内存，按照A的非零模式
        std::vector<Triplet<double>> L_triplets, U_triplets;
        
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SpMat::InnerIterator it(A, k); it; ++it) {
                int i = it.row();
                int j = it.col();
                if (i > j) {
                    L_triplets.push_back(Triplet<double>(i, j, 0.0));
                } else if (i == j) {
                    D_(i) = it.value();
                } else {
                    U_triplets.push_back(Triplet<double>(i, j, 0.0));
                }
            }
        }
        
        L_.setFromTriplets(L_triplets.begin(), L_triplets.end());
        U_.setFromTriplets(U_triplets.begin(), U_triplets.end());
        
        // 异步迭代ILU分解
        omp_set_num_threads(numThreads);
        
        for (int iter = 0; iter < maxIter_; ++iter) {
            double maxChange = 0.0;
            
            #pragma omp parallel
            {
                double localMaxChange = 0.0;
                
                // 更新下三角部分L
                #pragma omp for schedule(dynamic)
                for (int i = 0; i < n_; ++i) {
                    for (SpMat::InnerIterator it(A, i); it; ++it) {
                        if (it.row() > it.col()) {  // 下三角
                            int row = it.row();
                            int col = it.col();
                            
                            double sum = 0.0;
                            SpMat::InnerIterator itL(L_, row);
                            SpMat::InnerIterator itU(U_, col);
                            
                            while (itL && itU) {
                                if (itL.col() < itU.row()) {
                                    ++itL;
                                } else if (itL.col() > itU.row()) {
                                    ++itU;
                                } else {
                                    sum += itL.value() * itU.value();
                                    ++itL;
                                    ++itU;
                                }
                            }
                            
                            double oldVal = L_.coeff(row, col);
                            double newVal = (it.value() - sum) / D_(col);
                            L_.coeffRef(row, col) = newVal;
                            
                            localMaxChange = std::max(localMaxChange, 
                                                     std::abs(newVal - oldVal));
                        }
                    }
                }
                
                // 更新对角部分D
                #pragma omp for schedule(dynamic)
                for (int i = 0; i < n_; ++i) {
                    double sum = 0.0;
                    for (SpMat::InnerIterator itL(L_, i); itL; ++itL) {
                        int k = itL.col();
                        double u_ki = U_.coeff(k, i);
                        sum += itL.value() * D_(k) * u_ki;
                    }
                    
                    double oldVal = D_(i);
                    double newVal = A.coeff(i, i) - sum;
                    D_(i) = newVal;
                    
                    localMaxChange = std::max(localMaxChange, 
                                             std::abs(newVal - oldVal));
                }
                
                // 更新上三角部分U
                #pragma omp for schedule(dynamic)
                for (int j = 0; j < n_; ++j) {
                    for (SpMat::InnerIterator it(A, j); it; ++it) {
                        if (it.row() < it.col()) {  // 上三角
                            int row = it.row();
                            int col = it.col();
                            
                            double sum = 0.0;
                            SpMat::InnerIterator itL(L_, row);
                            SpMat::InnerIterator itU(U_, col);
                            
                            while (itL && itU) {
                                if (itL.col() < itU.row()) {
                                    ++itL;
                                } else if (itL.col() > itU.row()) {
                                    ++itU;
                                } else {
                                    sum += itL.value() * D_(itL.col()) * itU.value();
                                    ++itL;
                                    ++itU;
                                }
                            }
                            
                            double oldVal = U_.coeff(row, col);
                            double newVal = (it.value() - sum) / D_(row);
                            U_.coeffRef(row, col) = newVal;
                            
                            localMaxChange = std::max(localMaxChange, 
                                                     std::abs(newVal - oldVal));
                        }
                    }
                }
                
                #pragma omp critical
                {
                    maxChange = std::max(maxChange, localMaxChange);
                }
            }
            
            if (maxChange < tol_) {
                std::cout << "ILU收敛于第 " << iter + 1 << " 次迭代" << std::endl;
                break;
            }
        }
    }
    
    // ====== 统一的接口：支持 Eigen 向量 ======
    Vec solve(const Vec& b) const {
        Vec y = Vec::Zero(n_);
        Vec z = Vec::Zero(n_);
        Vec x = Vec::Zero(n_);
        
        forwardSolve_ptr(b.data(), y.data());
        diagonalSolve_ptr(y.data(), z.data());
        backwardSolve_ptr(z.data(), x.data());
        
        return x;
    }
    
    // ====== 统一的接口：支持原始指针（供 GMRES 使用）======
    void solve(const double* b, double* x, double* temp) const {
        forwardSolve_ptr(b, temp);      // temp = L^-1 * b
        diagonalSolve_ptr(temp, x);     // x = D^-1 * temp
        backwardSolve_ptr(x, temp);     // temp = U^-1 * x
        std::memcpy(x, temp, n_ * sizeof(double)); // x = temp
    }
};

// ==========================================
// 4. Givens 旋转辅助函数
// ==========================================

inline void generate_givens_rotation(double a, double b, double& c, double& s) {
    if (std::abs(b) < 1e-14) {
        c = 1.0;
        s = 0.0;
    } else if (std::abs(b) > std::abs(a)) {
        double tau = -a / b;
        s = 1.0 / std::sqrt(1.0 + tau * tau);
        c = s * tau;
    } else {
        double tau = -b / a;
        c = 1.0 / std::sqrt(1.0 + tau * tau);
        s = c * tau;
    }
}

inline void apply_givens_rotation(double& x, double& y, double c, double s) {
    double temp = c * x - s * y;
    y = s * x + c * y;
    x = temp;
}

// ==========================================
// 5. 基于 Givens 旋转的高性能 GMRES
// ==========================================
class FastGMRES {
    int n_;
    int restart_;
    
    double* data_V_flat;
    std::vector<double*> V;
    double* x_ptr;
    double* r_ptr;
    double* w_ptr;
    double* Av_ptr;
    double* temp_ptr;
    
    std::vector<double> cos_array;
    std::vector<double> sin_array;
    
public:
    FastGMRES(int n, int restart) : n_(n), restart_(restart) {
        size_t vec_size = n * sizeof(double);
        data_V_flat = (double*)aligned_malloc(vec_size * (restart + 1));
        
        V.resize(restart + 1);
        for(int i=0; i<=restart; ++i) V[i] = data_V_flat + i * n;
        
        x_ptr = (double*)aligned_malloc(vec_size);
        r_ptr = (double*)aligned_malloc(vec_size);
        w_ptr = (double*)aligned_malloc(vec_size);
        Av_ptr = (double*)aligned_malloc(vec_size);
        temp_ptr = (double*)aligned_malloc(vec_size);
        
        cos_array.resize(restart);
        sin_array.resize(restart);
    }
    
    ~FastGMRES() {
        aligned_free(data_V_flat);
        aligned_free(x_ptr);
        aligned_free(r_ptr);
        aligned_free(w_ptr);
        aligned_free(Av_ptr);
        aligned_free(temp_ptr);
    }
    
    VectorXd solve(const SpMat& A, const VectorXd& b, const AsyncILU& precond, 
                   int max_iter, double tol) {
        // 初始化 x = 0, r = b
        std::memset(x_ptr, 0, n_ * sizeof(double));
        std::memcpy(r_ptr, b.data(), n_ * sizeof(double));

        // 预条件处理初始残差
        precond.solve(r_ptr, w_ptr, temp_ptr); 
        std::memcpy(r_ptr, w_ptr, n_ * sizeof(double));

        double beta = norm_avx512(r_ptr, n_);
        if (beta < tol) return Map<VectorXd>(x_ptr, n_);

        MatrixXd H = MatrixXd::Zero(restart_ + 1, restart_);
        VectorXd g = VectorXd::Zero(restart_ + 1);

        int iter = 0;
        while (iter < max_iter) {
            std::memcpy(V[0], r_ptr, n_ * sizeof(double));
            scale_avx512(V[0], 1.0/beta, n_);
            
            g.setZero();
            g(0) = beta;
            
            int i = 0;
            for (; i < restart_ && iter < max_iter; ++i, ++iter) {
                // w = M^-1 * A * V[i]
                spmv_csr_parallel(A, V[i], Av_ptr);
                precond.solve(Av_ptr, w_ptr, temp_ptr);
                
                // Modified Gram-Schmidt
                for (int k = 0; k <= i; ++k) {
                    double h_ki = dot_product_avx512(w_ptr, V[k], n_);
                    H(k, i) = h_ki;
                    axpy_avx512(w_ptr, w_ptr, -h_ki, V[k], n_);
                }
                
                double h_next = norm_avx512(w_ptr, n_);
                H(i + 1, i) = h_next;
                
                if (h_next > 1e-14) {
                    scale_avx512(w_ptr, 1.0/h_next, n_);
                    std::memcpy(V[i + 1], w_ptr, n_ * sizeof(double));
                } else {
                    break;
                }

                // 应用之前的 Givens 旋转
                for (int k = 0; k < i; ++k) {
                    apply_givens_rotation(H(k, i), H(k + 1, i), 
                                         cos_array[k], sin_array[k]);
                }
                
                // 生成新的 Givens 旋转
                generate_givens_rotation(H(i, i), H(i + 1, i), 
                                        cos_array[i], sin_array[i]);
                
                // 应用新旋转
                apply_givens_rotation(H(i, i), H(i + 1, i), 
                                     cos_array[i], sin_array[i]);
                apply_givens_rotation(g(i), g(i + 1), 
                                     cos_array[i], sin_array[i]);
                
                double residual = std::abs(g(i + 1));
                
                if (residual < tol) {
                    // 回代求解上三角系统
                    VectorXd y(i + 1);
                    for (int k = i; k >= 0; --k) {
                        double sum = g(k);
                        for (int j = k + 1; j <= i; ++j) {
                            sum -= H(k, j) * y(j);
                        }
                        y(k) = sum / H(k, k);
                    }
                    
                    // 更新解
                    for (int k = 0; k <= i; ++k) {
                        axpy_avx512(x_ptr, x_ptr, y(k), V[k], n_);
                    }
                    
                    std::cout << "GMRES收敛于第 " << iter + 1 
                              << " 次迭代, 残差=" << residual << std::endl;
                    return Map<VectorXd>(x_ptr, n_);
                }
            }

            // Restart：更新解
            VectorXd y(i);
            for (int k = i - 1; k >= 0; --k) {
                double sum = g(k);
                for (int j = k + 1; j < i; ++j) {
                    sum -= H(k, j) * y(j);
                }
                y(k) = sum / H(k, k);
            }
            
            for (int k = 0; k < i; ++k) {
                axpy_avx512(x_ptr, x_ptr, y(k), V[k], n_);
            }

            // 重新计算残差
            spmv_csr_parallel(A, x_ptr, Av_ptr);
            
            #pragma omp parallel for
            for(int k=0; k<n_; ++k) r_ptr[k] = b[k] - Av_ptr[k];
            
            precond.solve(r_ptr, w_ptr, temp_ptr);
            std::memcpy(r_ptr, w_ptr, n_ * sizeof(double));
            
            beta = norm_avx512(r_ptr, n_);
            std::cout << "Restart后残差: " << beta << std::endl;
            if (beta < tol) return Map<VectorXd>(x_ptr, n_);
        }

        return Map<VectorXd>(x_ptr, n_);
    }
};