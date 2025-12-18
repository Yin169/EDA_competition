
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Eigen;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef VectorXd Vec;

/**
 * 异步迭代ILU(0)分解类
 * 使用OpenMP实现并行异步迭代
 */
class AsyncILU {
private:
    SpMat L_;  // 下三角矩阵
    SpMat U_;  // 上三角矩阵
    Vec D_;    // 对角元素
    int n_;    // 矩阵大小
    int maxIter_;  // 最大迭代次数
    double tol_;   // 收敛容差
    
public:
    AsyncILU(int maxIter = 10, double tol = 1e-6) 
        : maxIter_(maxIter), tol_(tol) {}
    
    void compute(const SpMat& A) {
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
        // omp_set_num_threads(numThreads);
        
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
                            // 计算 sum = sum(L(row,k) * U(k,col))
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
                    // 计算 sum = sum(L(i,k) * D(k) * U(k,i))
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
                            // 计算 sum = sum(L(row,k) * U(k,col))
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
            
            std::cout << "Iteration " << iter << ", max change: " 
                      << maxChange << std::endl;
            
            if (maxChange < tol_) {
                std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }
        }
    }
    
    /**
     * 前向替换求解 L*y = b
     */
    Vec forwardSolve(const Vec& b) const {
        Vec y = Vec::Zero(n_);
        for (int i = 0; i < n_; ++i) {
            double sum = 0.0;
            for (SpMat::InnerIterator it(L_, i); it; ++it) {
                sum += it.value() * y(it.col());
            }
            y(i) = (b(i) - sum);
        }
        return y;
    }
    
    /**
     * 对角缩放 D*z = y
     */
    Vec diagonalSolve(const Vec& y) const {
        Vec z = Vec::Zero(n_);
        for (int i = 0; i < n_; ++i) {
            z(i) = y(i) / D_(i);
        }
        return z;
    }
    
    /**
     * 后向替换求解 U*x = z
     */
    Vec backwardSolve(const Vec& z) const {
        Vec x = Vec::Zero(n_);
        for (int i = n_ - 1; i >= 0; --i) {
            double sum = 0.0;
            for (SpMat::InnerIterator it(U_, i); it; ++it) {
                if (it.col() > i) {
                    sum += it.value() * x(it.col());
                }
            }
            x(i) = z(i) - sum;
        }
        return x;
    }
    
    /**
     * ILU预处理求解 (LDU)*x = b
     */
    Vec solve(const Vec& b) const {
        Vec y = forwardSolve(b);
        Vec z = diagonalSolve(y);
        Vec x = backwardSolve(z);
        return x;
    }
};

/**
 * 使用ILU预处理的共轭梯度法
 */
Vec preconditionedCG(const SpMat& A, const Vec& b, const AsyncILU& precond,
                     int maxIter = 1000, double tol = 1e-10) {
    int n = b.size();
    Vec x = Vec::Zero(n);
    Vec r = b - A * x;
    Vec z = precond.solve(r);
    Vec p = z;
    double rsold = r.dot(z);
    
    for (int i = 0; i < maxIter; ++i) {
        Vec Ap = A * p;
        double alpha = rsold / p.dot(Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        
        double residual = r.norm();
        if (residual < tol) {
            std::cout << "PCG converged in " << i + 1 << " iterations, "
                      << "residual = " << residual << std::endl;
            return x;
        }
        
        z = precond.solve(r);
        double rsnew = r.dot(z);
        double beta = rsnew / rsold;
        p = z + beta * p;
        rsold = rsnew;
        
        if ((i + 1) % 50 == 0) {
            std::cout << "Iteration " << i + 1 << ", residual = " 
                      << residual << std::endl;
        }
    }
    
    std::cout << "PCG did not converge" << std::endl;
    return x;
}

// 测试函数：生成稀疏测试矩阵
SpMat generateTestMatrix(int n, int nnzPerRow = 5) {
    std::vector<Triplet<double>> triplets;
    
    for (int i = 0; i < n; ++i) {
        // 对角元素
        triplets.push_back(Triplet<double>(i, i, 4.0));
        
        // 非对角元素
        for (int j = 0; j < nnzPerRow - 1; ++j) {
            int col = (i + j + 1) % n;
            if (col != i) {
                triplets.push_back(Triplet<double>(i, col, -1.0 / nnzPerRow));
            }
        }
    }
    
    SpMat A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

/**
 * ==========================================
 * 新增: 使用左预处理的重启GMRES算法 (GMRES(m))
 * ==========================================
 * 求解 M^{-1}Ax = M^{-1}b
 */
Vec preconditionedGMRES(const SpMat& A, const Vec& b, const AsyncILU& precond,
                        int maxIter = 1000, int restart = 30, double tol = 1e-10) {
    int n = b.size();
    Vec x = Vec::Zero(n); // 初始猜测 x0 = 0
    
    // Hessberg矩阵 H (大小 m+1 x m)
    MatrixXd H = MatrixXd::Zero(restart + 1, restart);
    // Krylov子空间基向量 V
    std::vector<Vec> V(restart + 1);
    // 旋转变换的正弦和余弦值
    Vec sn = Vec::Zero(restart);
    Vec cs = Vec::Zero(restart);
    // 最小二乘问题的右端项
    Vec s = Vec::Zero(restart + 1);

    // 计算初始残差 r0 = b - Ax
    Vec r = b - A * x;
    // 应用左预处理: r = M^{-1} * r
    r = precond.solve(r);
    
    double beta = r.norm();
    
    if (beta < tol) {
        std::cout << "GMRES: Initial guess is good enough." << std::endl;
        return x;
    }

    int iter = 0;
    while (iter < maxIter) {
        // 1. 初始化 Arnoldi 过程
        V[0] = r / beta;
        s.setZero();
        s(0) = beta;
        
        // 将 i 声明在循环外，以便后续使用
        int i = 0;
        
        // 2. 内部循环 (Arnoldi Process)
        for (; i < restart && iter < maxIter; ++i, ++iter) {
            // w = M^{-1} * A * V[i] (左预处理)
            Vec w = A * V[i];
            w = precond.solve(w);
            
            // Modified Gram-Schmidt 正交化
            for (int k = 0; k <= i; ++k) {
                H(k, i) = w.dot(V[k]);
                w -= H(k, i) * V[k];
            }
            
            H(i + 1, i) = w.norm();
            
            if (H(i + 1, i) > 1e-14) {
                V[i + 1] = w / H(i + 1, i);
            } else {
                break;
            }
            
            // 3. 应用之前的 Givens 旋转
            for (int k = 0; k < i; ++k) {
                double temp = cs(k) * H(k, i) + sn(k) * H(k + 1, i);
                H(k + 1, i) = -sn(k) * H(k, i) + cs(k) * H(k + 1, i);
                H(k, i) = temp;
            }
            
            // 4. 计算新的 Givens 旋转
            double nu = std::sqrt(H(i, i)*H(i, i) + H(i+1, i)*H(i+1, i));
            cs(i) = H(i, i) / nu;
            sn(i) = H(i + 1, i) / nu;
            
            H(i, i) = cs(i) * H(i, i) + sn(i) * H(i + 1, i);
            H(i + 1, i) = 0.0;
            
            // 更新右端项 s
            double temp_s = cs(i) * s(i); 
            s(i + 1) = -sn(i) * s(i);
            s(i) = temp_s;
            
            double current_error = std::abs(s(i + 1));
            
            if ((iter + 1) % 3 == 0) {
                std::cout << "GMRES Iter " << iter + 1 << ", Residual: " << current_error << std::endl;
            }
            
            // 如果收敛，立即计算结果并返回
            if (current_error < tol) {
                // 更新 i (因为本次迭代已完成，实际上维度是 i+1，但在 0-based index 中直接用 i 即可对应逻辑)
                // 求解 H*y = s
                Vec y = Vec::Zero(i + 1);
                for (int k = i; k >= 0; --k) {
                    y(k) = s(k);
                    for (int j = k + 1; j <= i; ++j) {
                        y(k) -= H(k, j) * y(j);
                    }
                    y(k) /= H(k, k);
                }
                
                for (int j = 0; j <= i; ++j) {
                    x += V[j] * y(j);
                }
                
                std::cout << "GMRES converged in " << iter + 1 << " iterations." << std::endl;
                return x;
            }
        }
        
        // 5. 重启 (Restart): 求解当前子空间的解并更新 x
        // 此时 i 等于 restart (或者 maxIter 耗尽)
        // 子空间维度为 i (即 0 到 i-1)
        {
            Vec y = Vec::Zero(i);
            for (int k = i - 1; k >= 0; --k) {
                y(k) = s(k);
                for (int j = k + 1; j < i; ++j) {
                    y(k) -= H(k, j) * y(j);
                }
                y(k) /= H(k, k);
            }
            
            for (int j = 0; j < i; ++j) {
                x += V[j] * y(j);
            }
        }
        
        // 6. 重新计算残差，准备下一次 restart 循环
        r = b - A * x;
        r = precond.solve(r);
        beta = r.norm();
        
        if (beta < tol) {
            std::cout << "GMRES converged at restart check." << std::endl;
            return x;
        }
    }
    
    std::cout << "GMRES reached max iterations without full convergence." << std::endl;
    return x;
}
