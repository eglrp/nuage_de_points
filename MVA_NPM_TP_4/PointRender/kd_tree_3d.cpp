#include "kd_tree_3d.h"
#include "nanoflann.h"

struct PointCloud
{
    struct SimplePoint
    {
        double  x, y, z;
        int index;
    };
    std::vector<SimplePoint>  pts;


    size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    double_t kdtree_distance(const double_t *p1, const size_t idx_p2, size_t /*size*/) const
    {
        const double d0 = p1[0] - pts[idx_p2].x;
        const double d1 = p1[1] - pts[idx_p2].y;
        const double d2 = p1[2] - pts[idx_p2].z;
        return d0*d0 + d1*d1 + d2*d2;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline double kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

};

class KdTree3D::Impl
{
    PointCloud cloud;
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 3> my_kd_tree_t;
    std::unique_ptr<my_kd_tree_t> kdtree;
public:
    KdTree3D::Impl() {
        kdtree = nullptr;
    }
    void set_points(const std::vector<Vec3f>& points) {
        kdtree = nullptr;
        cloud.pts.clear();
        for (int k = 0; k < points.size(); k++)
            cloud.pts.push_back(PointCloud::SimplePoint{ points[k][0],points[k][1],points[k][2], k });
        kdtree = std::make_unique<my_kd_tree_t>(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        kdtree->buildIndex();
    }

    std::vector<size_t> query(Vec3f p, int k) {
        PointCloud::SimplePoint query_pt{ p[0],p[1],p[2],-1 };
        size_t ret_index;
        std::vector<size_t> indices(k);
        std::vector<double> dists(k);
        double out_dist_sqr;
        kdtree->knnSearch(&query_pt.x, k, &indices[0], &dists[0]);
        std::vector<size_t> indices_out(k);
        for (int i = 0; i < k; i++) {
            indices_out[i] = cloud.pts[indices[i]].index;
        }
        return indices_out;
    }
};

KdTree3D::KdTree3D() {
    pImpl = std::make_unique<Impl>();
}

KdTree3D::~KdTree3D() {
}

void KdTree3D::set_points(const std::vector<Vec3f>& points) {
    pImpl->set_points(points);
}

std::vector<size_t> KdTree3D::query(Vec3f p, int k) const {
    return pImpl->query(p, k);
}
