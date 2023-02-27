#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <fast_gicp/gicp/gicp_settings.hpp>
#include <fast_gicp/cuda/covariance_estimation.cuh>

namespace fast_gicp {
  namespace cuda {

namespace {
  struct covariance_estimation_kernel {
    /**
     * @brief 输入点云，k值，近邻点索引，计算协方差
     * @param points
     * @param k
     * @param k_neighbors
     * @param covariances
     */
    covariance_estimation_kernel(const thrust::device_vector<Eigen::Vector3f>& points, int k, const thrust::device_vector<int>& k_neighbors, thrust::device_vector<Eigen::Matrix3f>& covariances)
        : k(k), points_ptr(points.data()), k_neighbors_ptr(k_neighbors.data()), covariances_ptr(covariances.data()) {}

    __host__ __device__ void operator()(int idx) const {
      // target points buffer & nn output buffer
      // points[i]可以指向点云中的某个点
      const Eigen::Vector3f* points = thrust::raw_pointer_cast(points_ptr);
      // k_neighbors[]关于源点云中第idx个点的近邻点索引
      const int* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;
      // 关于源点云中第idx个点的k个近邻点协方差，计算出来的协方差填到这里来
      Eigen::Matrix3f* cov = thrust::raw_pointer_cast(covariances_ptr) + idx;

      Eigen::Vector3f mean(0.0f, 0.0f, 0.0f);
      cov->setZero();
      // 遍历k次
      for(int i = 0; i < k; i++) {
        // k个近邻点坐标值求和
        const auto& pt = points[k_neighbors[i]];
        mean += pt;
        (*cov) += pt * pt.transpose();
      }
      mean /= k;
      // 近似协方差，忽略了小项
      (*cov) = (*cov) / k - mean * mean.transpose();
    }

    const int k;
    thrust::device_ptr<const Eigen::Vector3f> points_ptr;
    thrust::device_ptr<const int> k_neighbors_ptr;

    thrust::device_ptr<Eigen::Matrix3f> covariances_ptr;
  };
  }  // namespace

  /**
   * @brief
   * @param points
   * @param k
   * @param k_neighbors
   * @param covariances
   */
  void covariance_estimation(const thrust::device_vector<Eigen::Vector3f>& points, int k, const thrust::device_vector<int>& k_neighbors, thrust::device_vector<Eigen::Matrix3f>& covariances) {
    // 初始化 d_indices = {0,1,2,...}
    thrust::device_vector<int> d_indices(points.size());
    thrust::sequence(d_indices.begin(), d_indices.end());

    covariances.resize(points.size());
    // GPU并行计算每个点的近邻点协方差
    thrust::for_each(d_indices.begin(), d_indices.end(), covariance_estimation_kernel(points, k, k_neighbors, covariances));
  }
  }
}
