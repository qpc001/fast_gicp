#ifndef FAST_GICP_FAST_VGICP_CUDA_IMPL_HPP
#define FAST_GICP_FAST_VGICP_CUDA_IMPL_HPP

#include <atomic>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#include <fast_gicp/cuda/fast_vgicp_cuda.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace fast_gicp {

template<typename PointSource, typename PointTarget>
FastVGICPCuda<PointSource, PointTarget>::FastVGICPCuda() : LsqRegistration<PointSource, PointTarget>() {
  this->reg_name_ = "FastVGICPCuda";
  // 近邻点搜索数量
  k_correspondences_ = 20;
  // voxel大小
  voxel_resolution_ = 1.0;
  // 协方差计算模型
  regularization_method_ = RegularizationMethod::PLANE;
  // 近邻点搜索方式（TODO：增加GPU版本）
  neighbor_search_method_ = NearestNeighborMethod::CPU_PARALLEL_KDTREE;

  // 内部CUDA操作实例cuda::FastVGICPCudaCore
  vgicp_cuda_.reset(new cuda::FastVGICPCudaCore());
  vgicp_cuda_->set_resolution(voxel_resolution_);
  // ？？？
  vgicp_cuda_->set_kernel_params(0.5, 3.0);
}

template<typename PointSource, typename PointTarget>
FastVGICPCuda<PointSource, PointTarget>::~FastVGICPCuda() {}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {}

/**
 * @brief 调用内部CUDA操作实例cuda::FastVGICPCudaCore设置resolution
 * @param resolution
 */
template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setResolution(double resolution) {
  vgicp_cuda_->set_resolution(resolution);
}

template <typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setKernelWidth(double kernel_width, double max_dist) {
  if (max_dist <= 0.0) {
    max_dist = kernel_width * 5.0;
  }
  vgicp_cuda_->set_kernel_params(kernel_width, max_dist);
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setNeighborSearchMethod(NeighborSearchMethod method, double radius) {
  vgicp_cuda_->set_neighbor_search_method(method, radius);
}

template <typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setNearestNeighborSearchMethod(NearestNeighborMethod method) {
  neighbor_search_method_ = method;
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::swapSourceAndTarget() {
  vgicp_cuda_->swap_source_and_target();
  input_.swap(target_);
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::clearSource() {
  input_.reset();
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::clearTarget() {
  target_.reset();
}

/**
 * @brief 输入源点云（通常为新的一帧点云）
 * @param cloud
 */
template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  // the input cloud is the same as the previous one
  // 如果点云是之前输入的同一帧，则直接返回
  if(cloud == input_) {
    return;
  }

  // 调用基类pcl::Registration::setInputSource
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);

  // 遍历点云，将点云转成std::vector形式
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud->size());
  std::transform(cloud->begin(), cloud->end(), points.begin(), [=](const PointSource& pt) { return pt.getVector3fMap(); });

  // 传入std::vector形式的点云，将数据保存到显存上的数据thrust::device_vector
  vgicp_cuda_->set_source_cloud(points);

  // 使用对应的近邻搜索方法，搜索并计算协方差
  switch(neighbor_search_method_) {
    case NearestNeighborMethod::CPU_PARALLEL_KDTREE: {
      // 使用openmp，CPU多线程调用kdtree搜索近邻点，返回点云中每个点对应的近邻点索引
      std::vector<int> neighbors = find_neighbors_parallel_kdtree<PointSource>(k_correspondences_, cloud);
      // 设置源点云的近邻点的索引，并且保存到显存设备上thrust::device_vector
      vgicp_cuda_->set_source_neighbors(k_correspondences_, neighbors);
      // 计算并重组协方差矩阵 （GPU）
      vgicp_cuda_->calculate_source_covariances(regularization_method_);
    } break;
    case NearestNeighborMethod::GPU_BRUTEFORCE:
      // 对于源点云中的每个点x，遍历目标点云，找到关于源点云中的一个点x的目标点云近邻点索引，结果保存到成员变量（GPU）
      vgicp_cuda_->find_source_neighbors(k_correspondences_);
      // 计算并重组协方差矩阵（GPU）
      vgicp_cuda_->calculate_source_covariances(regularization_method_);
      break;
    case NearestNeighborMethod::GPU_RBF_KERNEL:
      vgicp_cuda_->calculate_source_covariances_rbf(regularization_method_);
      break;
  }
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  // the input cloud is the same as the previous one
  if(cloud == target_) {
    return;
  }

  // 1. 设置标识位：target_cloud_updated_
  // 2. 把输入点云保存到成员变量pcl::Registration::target_
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);

  // 遍历点云，将点云转成std::vector形式
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud->size());
  std::transform(cloud->begin(), cloud->end(), points.begin(), [=](const PointTarget& pt) { return pt.getVector3fMap(); });

  // 传入std::vector形式的点云，将数据保存到显存上的数据thrust::device_vector
  vgicp_cuda_->set_target_cloud(points);

  // 使用对应的近邻搜索方法，搜索并计算协方差
  switch(neighbor_search_method_) {
    case NearestNeighborMethod::CPU_PARALLEL_KDTREE: {
      // 使用openmp，CPU多线程调用kdtree搜索近邻点，返回点云中每个点对应的近邻点索引
      std::vector<int> neighbors = find_neighbors_parallel_kdtree<PointTarget>(k_correspondences_, cloud);
      // 设置源点云的近邻点的索引，并且保存到显存设备上thrust::device_vector
      vgicp_cuda_->set_target_neighbors(k_correspondences_, neighbors);
      // 计算并重组协方差矩阵 （GPU）
      vgicp_cuda_->calculate_target_covariances(regularization_method_);
    } break;
    case NearestNeighborMethod::GPU_BRUTEFORCE:
      // 对于源点云中的每个点x，遍历目标点云，找到关于源点云中的一个点x的目标点云近邻点索引，结果保存到成员变量（GPU）
      vgicp_cuda_->find_target_neighbors(k_correspondences_);
      // 计算并重组协方差矩阵（GPU）
      vgicp_cuda_->calculate_target_covariances(regularization_method_);
      break;
    case NearestNeighborMethod::GPU_RBF_KERNEL:
      vgicp_cuda_->calculate_target_covariances_rbf(regularization_method_);
      break;
  }
  //
  vgicp_cuda_->create_target_voxelmap();
}

template<typename PointSource, typename PointTarget>
void FastVGICPCuda<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  vgicp_cuda_->set_resolution(this->voxel_resolution_);

  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

/**
 * @brief 使用openmp，CPU多线程调用kdtree搜索近邻点
 * @param k 近邻点数
 * @param cloud
 * @return 返回输入点云每个点对应的近邻点
 */
template<typename PointSource, typename PointTarget>
template<typename PointT>
std::vector<int> FastVGICPCuda<PointSource, PointTarget>::find_neighbors_parallel_kdtree(int k, typename pcl::PointCloud<PointT>::ConstPtr cloud) const {
  // 构造KdTree（CPU）
  pcl::search::KdTree<PointT> kdtree;
  kdtree.setInputCloud(cloud);
  std::vector<int> neighbors(cloud->size() * k);

  // 使用openmp，CPU多线程调用kdtree搜索近邻点
#pragma omp parallel for schedule(guided, 8)
  for(int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k, k_indices, k_sq_distances);

    std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
  }

  return neighbors;
}

template<typename PointSource, typename PointTarget>
double FastVGICPCuda<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  vgicp_cuda_->update_correspondences(trans);
  return vgicp_cuda_->compute_error(trans, H, b);
}

template<typename PointSource, typename PointTarget>
double FastVGICPCuda<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  return vgicp_cuda_->compute_error(trans, nullptr, nullptr);
}

}  // namespace fast_gicp

#endif
