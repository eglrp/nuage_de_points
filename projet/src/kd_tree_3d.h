#pragma once
#include <memory>
#include <vector>
#include "nanoflann.h"
#include <Eigen/Dense>
namespace KdTreeDetails
{
    template <typename PointType>
    struct PointCloudAdaptor
    {
        typedef float_t coord_t;

        const PointType *begin; //!< A const ref to the data set origin
        const size_t n;

        /// The constructor that sets the data set source
        PointCloudAdaptor(const PointType *begin_, size_t n) : begin(begin_), n(n) { }

        /// CRTP helper method
        inline const PointType* derived() const { return begin; }

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return n; }

        // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
        coord_t kdtree_distance(const coord_t *p1, const size_t idx_p2, size_t /*size*/) const
        {
            const coord_t d0 = p1[0] - derived()[idx_p2][0];
            const coord_t d1 = p1[1] - derived()[idx_p2][1];
            const coord_t d2 = p1[2] - derived()[idx_p2][2];
            return d0*d0 + d1*d1 + d2*d2;
        }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline coord_t kdtree_get_pt(const size_t idx, int dim) const
        {
            if (dim == 0) return derived()[idx][0];
            else if (dim == 1) return derived()[idx][1];
            else return derived()[idx][2];
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

    };

}

class KdTree3D
{
    typedef KdTreeDetails::PointCloudAdaptor<Eigen::Vector3f> PointCloudAdaptor;
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3
    > my_kd_tree_t;
    std::unique_ptr<my_kd_tree_t> kdtree;
    std::unique_ptr<PointCloudAdaptor> points_adaptor;
public:
    KdTree3D() {
        kdtree = nullptr;
    }

    void set_points(const Eigen::Vector3f* pt_begin, int n) {
        kdtree = nullptr;
        points_adaptor = std::make_unique<PointCloudAdaptor>(pt_begin, n);
        kdtree = std::make_unique<my_kd_tree_t>(
            3, *points_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(20 /* max leaf */));
        kdtree->buildIndex();
    }

    int query(Eigen::Vector3f p) const {
        size_t num_results = 1;
        size_t ret_index;
        float out_dist_sqr;
        kdtree->knnSearch(&p[0], num_results, &ret_index, &out_dist_sqr);
        return int(ret_index);
    }

    void query_radius(Eigen::Vector3f p, float radius, std::vector<std::pair<size_t, float> >& ret_matches) const {
        nanoflann::SearchParams params;
        kdtree->radiusSearch(&p[0], radius, ret_matches, params);
    }
};