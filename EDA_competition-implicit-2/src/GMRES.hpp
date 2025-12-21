/**
 * ==========================================
 * 高性能并行区域分解GMRES求解器 (优化版)
 * ==========================================
 */

#define EIGEN_DONT_PARALLELIZE
#include <omp.h>
#include <vector>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace Eigen;
typedef SparseMatrix<double, RowMajor> SpMat; 
typedef VectorXd Vec;

// ... [保留原有的 parallelSpMV, parallelDot, parallelNorm 等基础函数] ...
// 为节省篇幅，这里假设基础并行操作函数（parallelSpMV等）已定义，与原代码一致。
// 请确保保留 parallelSpMV, parallelSubtract, parallelNorm。
// 下面重写关键部分。

void parallelSpMV(const SpMat& A, const Vec& x, Vec& y) {
    int n = A.rows();
    #pragma omp parallel for schedule(dynamic, 128) // 稍微增大chunk size
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (SpMat::InnerIterator it(A, i); it; ++it) {
            sum += it.value() * x(it.col());
        }
        y(i) = sum;
    }
}

inline double parallelNorm(const Vec& a) {
    double result = 0.0;
    int n = a.size();
    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < n; ++i) result += a(i) * a(i);
    return std::sqrt(result);
}

inline void parallelSubtract(const Vec& a, const Vec& b, Vec& result) {
    int n = a.size();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) result(i) = a(i) - b(i);
}

/**
 * 优化点1: 工作空间预分配结构
 * 避免在并行循环内部进行 malloc/free
 */
struct GMRESWorkspace {
    MatrixXd H;
    std::vector<Vec> V;
    Vec sn;
    Vec cs;
    Vec s;
    Vec y; // 用于回代
    
    void resize(int restart, int max_local_size) {
        H.resize(restart + 1, restart);
        V.resize(restart + 1);
        for(auto& v : V) v.resize(max_local_size); // 预分配最大可能的大小
        sn.resize(restart);
        cs.resize(restart);
        s.resize(restart + 1);
        y.resize(restart + 1);
    }
};

struct SubdomainData {
    std::vector<int> localToGlobal;
    std::vector<int> globalToLocal;
    SpMat A_local;
    Vec delta_x_local;
    Vec residual_local; // 预分配残差向量
    GMRESWorkspace workspace; // 每个子域独有的工作空间
    int subdomain_id;
    
    void preallocate(int size, int restart) {
        delta_x_local = Vec::Zero(size);
        residual_local = Vec::Zero(size);
        workspace.resize(restart, size);
    }
};
class DomainDecomposition {
private:
    int num_subdomains;
    std::vector<SubdomainData> subdomains;
    // 移除 global_counts，RAS 不需要平均权重
    int global_size;
    int overlap_size; // 保存 overlap 大小

public:
    DomainDecomposition(int n_subdomains) 
        : num_subdomains(n_subdomains), global_size(0), overlap_size(0) {}
    
    void partitionMatrix(const SpMat& A, int overlap, int restart_limit) {
        global_size = A.rows();
        overlap_size = overlap;
        subdomains.resize(num_subdomains);
        
        int base_size = global_size / num_subdomains;
        
        #pragma omp parallel for schedule(dynamic)
        for (int sd = 0; sd < num_subdomains; ++sd) {
            SubdomainData& subdomain = subdomains[sd];
            subdomain.subdomain_id = sd;
            
            // === RAS 关键：区分 "求解区域" (含 overlap) 和 "更新区域" (不含 overlap) ===
            
            // 1. 扩展区域 (Extended): 用于提取矩阵和计算 update (包含 overlap)
            int ext_start = std::max(0, (sd * base_size) - overlap);
            int ext_end = std::min(global_size, ((sd == num_subdomains - 1) ? global_size : (sd + 1) * base_size) + overlap);
            
            subdomain.localToGlobal.clear();
            subdomain.globalToLocal.assign(global_size, -1);
            
            for (int i = ext_start; i < ext_end; ++i) {
                subdomain.globalToLocal[i] = subdomain.localToGlobal.size();
                subdomain.localToGlobal.push_back(i);
            }
            
            // 2. 提取子矩阵 (RowMajor 下提取逻辑相同，但存储更紧凑)
            int local_size = subdomain.localToGlobal.size();
            std::vector<Triplet<double>> triplets;
            // 预估非零元数量
            triplets.reserve(local_size * 10); 
            
            // 注意：因为 A 变成了 RowMajor，为了性能，外层循环建议遍历行
            // 或者直接利用 Eigen 的 block 提取功能（此处为了兼容性保留手动提取）
            for (int k = ext_start; k < ext_end; ++k) { // 遍历全局行
               int local_row = subdomain.globalToLocal[k];
               for (SpMat::InnerIterator it(A, k); it; ++it) {
                   int local_col = subdomain.globalToLocal[it.col()];
                   if (local_col >= 0) {
                       triplets.emplace_back(local_row, local_col, it.value());
                   }
               }
            }
            
            subdomain.A_local.resize(local_size, local_size);
            subdomain.A_local.setFromTriplets(triplets.begin(), triplets.end());
            subdomain.preallocate(local_size, restart_limit);
        }
    }
    
    // 快速限制残差，无内存分配
    void restrictResidual(int sd_id, const Vec& global_residual) {
        SubdomainData& sd = subdomains[sd_id];
        int n = sd.localToGlobal.size();
        // 直接写入预分配的向量，不resize，不alloc
        for (int i = 0; i < n; ++i) {
            sd.residual_local(i) = global_residual(sd.localToGlobal[i]);
        }
    }
    
 /**
     * 修复版子域求解器
     * 修复了 Givens Rotation 的计算顺序错误
     */
    void solveSubdomainCorrection(int sd_id, int maxIter, int restart, double tol) {
        SubdomainData& sd = subdomains[sd_id];
        GMRESWorkspace& ws = sd.workspace;
        
        int n = sd.residual_local.size();
        
        // 1. 初始化
        sd.delta_x_local.setZero(); 
        Vec r = sd.residual_local; // 初始残差
        double beta = r.norm();
        double initial_res = beta;
        
        if (beta < 1e-14) return; // 残差极小，无需计算
        
        int iter = 0;
        while (iter < maxIter) { // 外层循环处理 Restart
            // 初始化 V[0]
            ws.V[0].head(n) = r / beta;
            ws.s.setZero();
            ws.s(0) = beta;
            
            int i = 0;
            // Arnoldi 过程
            for (; i < restart && iter < maxIter; ++i, ++iter) {
                // w = A * V[i]
                ws.V[i+1].head(n) = sd.A_local * ws.V[i].head(n);
                Vec& w = ws.V[i+1];
                
                // Modified Gram-Schmidt 正交化
                for (int k = 0; k <= i; ++k) {
                    double h_val = w.head(n).dot(ws.V[k].head(n));
                    ws.H(k, i) = h_val;
                    w.head(n) -= h_val * ws.V[k].head(n);
                }
                
                ws.H(i + 1, i) = w.head(n).norm();
                
                // Lucky breakdown check (防止除零)
                if (ws.H(i + 1, i) < 1e-14) {
                    // 此时已经精确收敛或发生数值崩溃，停止扩展子空间
                    // 标记为需要直接求解
                } else {
                    w.head(n) /= ws.H(i + 1, i);
                }
                
                // === [关键修复] 应用之前的 Givens 旋转 ===
                for (int k = 0; k < i; ++k) {
                    double temp_h = ws.H(k, i);
                    double temp_h_next = ws.H(k + 1, i);
                    ws.H(k, i)     = ws.cs(k) * temp_h + ws.sn(k) * temp_h_next;
                    ws.H(k + 1, i) = -ws.sn(k) * temp_h + ws.cs(k) * temp_h_next;
                }
                
                // === 计算新的 Givens 旋转 ===
                double h_ii = ws.H(i, i);
                double h_ip1 = ws.H(i + 1, i); // H(i+1, i)
                
                double nu = std::sqrt(h_ii * h_ii + h_ip1 * h_ip1);
                
                if (nu > 1e-14) {
                    ws.cs(i) = h_ii / nu;
                    ws.sn(i) = h_ip1 / nu;
                    
                    // 更新对角线元素
                    ws.H(i, i) = ws.cs(i) * h_ii + ws.sn(i) * h_ip1;
                    ws.H(i + 1, i) = 0.0;
                } else {
                    // 极小值处理，防止 NaN
                    ws.cs(i) = 1.0;
                    ws.sn(i) = 0.0;
                }
                
                // 更新 s 向量
                double temp_s = ws.cs(i) * ws.s(i);
                ws.s(i + 1) = -ws.sn(i) * ws.s(i); // 残差传递到下一行
                ws.s(i) = temp_s;
                
                // 检查收敛性 (|s(i+1)| 即为当前残差范数)
                if (std::abs(ws.s(i + 1)) < tol * initial_res || std::abs(ws.s(i+1)) < 1e-14) {
                    // === 收敛：回代求解 ===
                    // 此时有效子空间维度为 i+1 (列下标 0 到 i)
                    for (int k = i; k >= 0; --k) {
                        ws.y(k) = ws.s(k);
                        for (int j = k + 1; j <= i; ++j) {
                            ws.y(k) -= ws.H(k, j) * ws.y(j);
                        }
                        // 避免除零
                        if (std::abs(ws.H(k, k)) > 1e-16)
                            ws.y(k) /= ws.H(k, k);
                        else 
                            ws.y(k) = 0;
                    }
                    // 更新解
                    for (int j = 0; j <= i; ++j) {
                        sd.delta_x_local += ws.V[j].head(n) * ws.y(j);
                    }
                    return; // 完成
                }
            } // End Inner Loop
            
            // === Restart 阶段的回代 ===
            // 循环结束（达到 restart 限制或 maxIter），在当前子空间求解最优解
            int m = i - 1; // 最后有效列的索引
            if (m >= 0) {
                for (int k = m; k >= 0; --k) {
                    ws.y(k) = ws.s(k);
                    for (int j = k + 1; j <= m; ++j) {
                        ws.y(k) -= ws.H(k, j) * ws.y(j);
                    }
                    if (std::abs(ws.H(k, k)) > 1e-16)
                        ws.y(k) /= ws.H(k, k);
                    else
                        ws.y(k) = 0;
                }
                for (int j = 0; j <= m; ++j) {
                    sd.delta_x_local += ws.V[j].head(n) * ws.y(j);
                }
            }
            
            // 计算真实残差，准备下一次 restart
            // r = r_local - A * x_current
            // 注意：sd.delta_x_local 是累积的，所以这里是正确的
            r = sd.residual_local - sd.A_local * sd.delta_x_local;
            beta = r.norm();
            
            if (beta < tol * initial_res) return;
        }
    }
    
    void parallelSolveCorrections(const Vec& global_residual, 
                                   int maxIter, int restart, double tol) {
        // dynamic schedule 应对不同子域收敛速度差异
        // 确保 num_subdomains >> num_threads
        #pragma omp parallel for schedule(dynamic, 1)
        for (int sd = 0; sd < num_subdomains; ++sd) {
            restrictResidual(sd, global_residual);
            solveSubdomainCorrection(sd, maxIter, restart, tol);
        }
    }
    
 void prolongAndAddCorrections(Vec& x_global) {
        int base_size = global_size / num_subdomains;

        #pragma omp parallel for schedule(static)
        for (int sd_idx = 0; sd_idx < num_subdomains; ++sd_idx) {
            const auto& sd = subdomains[sd_idx];
            
            // 计算该子域 "负责" 的不重叠区域范围
            int core_start_global = sd_idx * base_size;
            int core_end_global = (sd_idx == num_subdomains - 1) ? global_size : (sd_idx + 1) * base_size;

            int n_local = sd.localToGlobal.size();
            
            // 遍历局部解向量
            for (int i = 0; i < n_local; ++i) {
                int global_idx = sd.localToGlobal[i];
                
                // === RAS 逻辑 ===
                // 只有当该节点属于当前子域的 "核心区域" 时，才更新全局解
                // 重叠区域的数据被计算用来辅助边界条件，但不会被写入全局解（由邻居写入）
                if (global_idx >= core_start_global && global_idx < core_end_global) {
                    // 无需 atomic，因为每个 global_idx 只属于唯一一个核心区域
                    x_global(global_idx) += sd.delta_x_local(i);
                }
            }
        }
    }
};

/**
 * 主求解函数
 */
Vec parallelDomainDecompositionGMRES(const SpMat& A, const Vec& b, 
                                     int num_subdomains_ratio = 4, // 建议子域数是线程数的4倍
                                     int overlap = 1,
                                     int outer_iter = 200,
                                     int inner_iter = 100,
                                     int restart = 20,
                                     double tol = 1e-8) {
    
    int n_threads = omp_get_max_threads();
    int num_subdomains = n_threads * num_subdomains_ratio; // 增加子域数量以平衡负载
    
    std::cout << "=== Optimized Parallel DD-GMRES ===" << std::endl;
    std::cout << "Threads: " << n_threads << ", Subdomains: " << num_subdomains << std::endl;
    
    DomainDecomposition dd(num_subdomains);
    dd.partitionMatrix(A, overlap, restart); // 传入 restart 用于预分配
    
    Vec x_global = Vec::Zero(b.size());
    Vec Ax = Vec::Zero(b.size());
    Vec residual = Vec::Zero(b.size());
    
    double b_norm = parallelNorm(b);
    
    for (int iter = 0; iter < outer_iter; ++iter) {
        parallelSpMV(A, x_global, Ax);
        parallelSubtract(b, Ax, residual);
        
        double res_norm = parallelNorm(residual);
        if (iter % 10 == 0) std::cout << "Iter " << iter << ": " << res_norm << std::endl;
        
        if (res_norm < tol) {
            std::cout << "Converged." << std::endl;
            return x_global;
        }
        
        // 求解并直接更新
        dd.parallelSolveCorrections(residual, inner_iter, restart, 0.1 * res_norm);
        dd.prolongAndAddCorrections(x_global);
    }
    
    return x_global;
}