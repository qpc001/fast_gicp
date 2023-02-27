#include <Eigen/Core>

#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <nvbio/basic/vector_view.h>
#include <nvbio/basic/priority_queue.h>

namespace fast_gicp {
  namespace cuda {

namespace {
  /**
   * @brief
   */
  struct neighborsearch_kernel {
    /**
     * @brief 构造函数，注意，成员变量都用了指针类型
     * thrust::device_ptr<const Eigen::Vector3f> target_points_ptr; 存放点云指针
     * thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr; 存放近邻点搜索结果的容器指针
     * @param k
     * @param target
     * @param k_neighbors
     */
    neighborsearch_kernel(int k, const thrust::device_vector<Eigen::Vector3f>& target, thrust::device_vector<thrust::pair<float, int>>& k_neighbors)
        : k(k), num_target_points(target.size()), target_points_ptr(target.data()), k_neighbors_ptr(k_neighbors.data()) {}

    /**
     * @brief 并行操作的函数，并行的索引是源点云中的一个点x
     * 对于源点云中的一个点x:
     * 1. 遍历目标点云，找到关于源点云中的一个点x的目标点云近邻点索引
     * 2. 核心是使用了优先队列nvbio::priority_queue
     * @tparam Tuple
     * @param idx_x[in] thrust::for_each传进来的迭代器
     */
    template<typename Tuple>
    __host__ __device__ void operator()(const Tuple& idx_x) const {
      // threadIdx doesn't work because thrust split for_each in two loops
      // idx就是brute_force_knn_search函数中的序列{0,1,2,3,4,5 ... }
      int idx = thrust::get<0>(idx_x);
      // x 就是brute_force_knn_search函数中源点云中的一个点
      const Eigen::Vector3f& x = thrust::get<1>(idx_x);

      // 下面是对target_points进行并行遍历

      // target points buffer & nn output buffer
      const Eigen::Vector3f* pts = thrust::raw_pointer_cast(target_points_ptr);
      thrust::pair<float, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

      // priority queue
      struct compare_type {
        bool operator()(const thrust::pair<float, int>& lhs, const thrust::pair<float, int>& rhs) {
          return lhs.first < rhs.first;
        }
      };

      // nvbio::vector_view： 把内容封装成vector，使之可以使用[]来寻址
      // 使用nvbio::priority_queue优先队列，内部容器为nvbio::vector_view，而队列的索引从1开始，所以
      // nvbio::priority_queue优先队列推入第1个元素，即在 [k_neighbors-1+1]的地址上放入数据
      // 具体看一眼NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void priority_queue<Key,Container,Compare>::push这个函数就明白了
      typedef nvbio::vector_view<thrust::pair<float, int>*> vector_type;
      typedef nvbio::priority_queue<thrust::pair<float, int>, vector_type, compare_type> queue_type;
      // 优先队列的结果会对应的保存到k_neighbors，也就是k_neighbors_ptr
      queue_type queue(vector_type(0, k_neighbors - 1));

      // 首先运行k次，填充优先队列
      for(int i = 0; i < k; i++) {
        // 计算目标点云中的第i个点与源点云中的一个点x的距离
        float sq_dist = (pts[i] - x).squaredNorm();
        // 优先队列push数据
        queue.push(thrust::make_pair(sq_dist, i));
      }

      // 运行后面的num_target_points-k次，里面的弹出操作保证了优先队列的size为k
      for(int i = k; i < num_target_points; i++) {
        // 计算目标点云中的第i个点与源点云中的一个点x的距离
        float sq_dist = (pts[i] - x).squaredNorm();
        // 如果距离小于优先队列中的最小值，则弹出优先队列中的最大值，然后加入新的最小值
        if(sq_dist < queue.top().first) {
          queue.pop();
          queue.push(thrust::make_pair(sq_dist, i));
        }
      }

      // 运行完后，k_neighbors容器在对应的元素位置上就有关于源点云中的一个点x的k个近邻点索引
    }

    const int k;
    const int num_target_points;
    thrust::device_ptr<const Eigen::Vector3f> target_points_ptr;

    thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr;
  };

  struct sorting_kernel {
    sorting_kernel(int k, thrust::device_vector<thrust::pair<float, int>>& k_neighbors) : k(k), k_neighbors_ptr(k_neighbors.data()) {}

    __host__ __device__ void operator()(int idx) const {
      // target points buffer & nn output buffer
      thrust::pair<float, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

      // priority queue
      struct compare_type {
        bool operator()(const thrust::pair<float, int>& lhs, const thrust::pair<float, int>& rhs) {
          return lhs.first < rhs.first;
        }
      };

      typedef nvbio::vector_view<thrust::pair<float, int>*> vector_type;
      typedef nvbio::priority_queue<thrust::pair<float, int>, vector_type, compare_type> queue_type;
      queue_type queue(vector_type(k, k_neighbors - 1));
      queue.m_size = k;

      for(int i = 0; i < k; i++) {
        thrust::pair<float, int> poped = queue.top();
        queue.pop();

        k_neighbors[k - i - 1] = poped;
      }
    }

    const int k;
    thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr;
  };
}

/**
 * @brief 对于源点云中的每个点x，遍历目标点云，找到关于源点云中的一个点x的目标点云近邻点索引
 * @param source 源点云
 * @param target 近邻点搜索的目标点云（这里作为参数输入的仍然是点云配准中的源点云）
 * @param k 近邻点数量
 * @param k_neighbors 输出近邻点
 * @param do_sort 是否需要排序
 */
void brute_force_knn_search(const thrust::device_vector<Eigen::Vector3f>& source, const thrust::device_vector<Eigen::Vector3f>& target, int k, thrust::device_vector<thrust::pair<float, int>>& k_neighbors, bool do_sort=false) {
  // 在显存上初始化d_indices， d_indices 初始化为{0,1,2,3,4 ...}
  thrust::device_vector<int> d_indices(source.size());
  thrust::sequence(d_indices.begin(), d_indices.end());

  // 将d_indices和source的迭代器组合成一个新的迭代器
  auto first = thrust::make_zip_iterator(thrust::make_tuple(d_indices.begin(), source.begin()));
  auto last = thrust::make_zip_iterator(thrust::make_tuple(d_indices.end(), source.end()));

  // nvbio::priority_queue requires (k + 1) working space
  // 初始化thrust::device_vector<thrust::pair<float, int>>& k_neighbors
  k_neighbors.resize(source.size() * k, thrust::make_pair(-1.0f, -1));

  // 真正使用GPU的地方neighborsearch_kernel
  // 对于源点云中的每个点x，遍历目标点云，找到关于源点云中的一个点x的目标点云近邻点索引
  thrust::for_each(first, last, neighborsearch_kernel(k, target, k_neighbors));

  if(do_sort) {
    thrust::for_each(d_indices.begin(), d_indices.end(), sorting_kernel(k, k_neighbors));
  }
}

  }
} // namespace fast_gicp
