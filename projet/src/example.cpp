#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <Eigen/Dense>
#include "tinyply.h"
#include "kd_tree_3d.h"
#include <numeric>
#include <random>
#include <memory>


typedef std::chrono::time_point<std::chrono::high_resolution_clock> timepoint;
std::chrono::high_resolution_clock c;

inline std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
    return c.now();
}

inline double difference_millis(timepoint start, timepoint end)
{
    return double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

inline double difference_second(timepoint start, timepoint end)
{
    return double(std::chrono::duration_cast<std::chrono::seconds>(end - start).count());
}

const double PI = 3.141592653589793238463;

void read_point_cloud(const std::string & filename, std::vector<Eigen::Vector3f>& verts, std::vector<int32_t>& ls, bool has_labels = true)
{
    std::ifstream ss(filename, std::ios::binary);

    if (ss.fail())
    {
        throw std::runtime_error("failed to open " + filename);
    }

    tinyply::PlyFile file;

    file.parse_header(ss);

    std::cout << "================================================================\n";

    for (auto c : file.get_comments()) std::cout << "Comment: " << c << std::endl;

    for (auto e : file.get_elements())
    {
        std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
        for (auto p : e.properties)
        {
            std::cout << "\tproperty - " << p.name << " (" << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
        }
    }

    std::cout << "================================================================\n";

    // Tinyply 2.0 treats incoming data as untyped byte buffers. It's now
    // up to users to treat this data as they wish. See below for examples.
    std::shared_ptr<tinyply::PlyData> vertices, labels;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the file header prior to reading the data. For brevity of this sample, properties 
    // like vertex position are hard-coded: 
    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
    if (has_labels) {
        try { labels = file.request_properties_from_element("vertex", { "class" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
    }

    timepoint before = now();
    file.read(ss);
    timepoint after = now();

    // Good place to put a breakpoint!
    std::cout << "Parsing took " << difference_millis(before, after) << " ms: " << std::endl;
    if (vertices) std::cout << "\tRead " << vertices->count << " total vertices " << std::endl;
    if (labels) std::cout << "\tRead " << labels->count << " total labels " << std::endl;


    const size_t numVerticesBytes = vertices->buffer.size_bytes();
    verts.resize(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);


    if (has_labels) {
        const size_t numLablesBytes = labels->buffer.size_bytes();
        ls.resize(labels->count);
        std::memcpy(ls.data(), labels->buffer.get(), numLablesBytes);
    }
}

void local_pca(std::vector<Eigen::Vector3f>& pts, Eigen::Matrix3f& evec, Eigen::Vector3f& evar) {
    Eigen::Vector3f center(0., 0., 0.);
    const int N = pts.size();
    if (N <= 1) {
        evec = Eigen::Matrix3f::Identity();
        evar = Eigen::Vector3f(0., 0., 0.);
        return;
    }
    for (auto& p : pts) {
        center += p;
    }
    center /= N;

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (auto& x : pts) {
        Eigen::Vector3f xc = x - center;
        cov += xc*xc.transpose();
    }
    cov /= N;


    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
    es.compute(cov);
    evec = es.eigenvectors();
    evar = es.eigenvalues();
}

struct ShapeFeature
{
    float vertical_spreads, total_spreads, verticality, linearity, planarity, sphericity;
};

std::pair<Eigen::Matrix3f, Eigen::Vector3f> neighborhood_PCA(const Eigen::Vector3f& query,
    const std::vector<Eigen::Vector3f>& pts, KdTree3D& tree, float radius, int& nb_count) {
    std::vector<std::pair<size_t, float> > ret_matches;
    tree.query_radius(query, radius, ret_matches);
    const int N = ret_matches.size();
    nb_count = N;
    std::vector<Eigen::Vector3f> nbs(N);
    for (int i = 0; i < N; i++)
        nbs[i] = pts[ret_matches[i].first];


    Eigen::Matrix3f evec;
    Eigen::Vector3f evar;
    local_pca(nbs, evec, evar);
    return std::make_pair(evec, evar);
}
ShapeFeature compute_features_with_eigen(const std::pair<Eigen::Matrix3f, Eigen::Vector3f>& eigens, float radius) {


    const Eigen::Matrix3f& evec = eigens.first;
    const Eigen::Vector3f& evar = eigens.second;

    float l1 = evar[2];
    float l2 = evar[1];
    float l3 = evar[0];
    float denom = l1 + l2 + l3 + 1e-6f;
    l1 /= denom;
    l2 /= denom;
    l3 /= denom;

    Eigen::Vector3f normal = evec.col(0);
    Eigen::Vector3f ez(0., 0., 1.);
    float verticality = 2. * std::asin(std::min(std::abs(normal.dot(ez)), 1.0f)) / PI;

    l1 += 1e-6f;
    float linearity = 1. - l2 / l1;
    float planarity = (l2 - l3) / l1;
    float sphericity = l3 / l1;

    const Eigen::Vector3f& v0 = evec.col(0);
    const Eigen::Vector3f& v1 = evec.col(1);
    const Eigen::Vector3f& v2 = evec.col(2);

    float vertical_spreads =
        v0[2] * v0[2] * evar[0] +
        v1[2] * v1[2] * evar[1] +
        v2[2] * v2[2] * evar[2];
    vertical_spreads /= denom;
    float total_spreads = denom / radius / radius;

    if (!std::isfinite(vertical_spreads)) throw "error";
    if (!std::isfinite(total_spreads)) throw "error";
    if (!std::isfinite(verticality)) throw "error";
    if (!std::isfinite(linearity)) throw "error";
    if (!std::isfinite(planarity)) throw "error";
    if (!std::isfinite(sphericity)) throw "error";

    return ShapeFeature{ vertical_spreads, total_spreads, verticality, linearity, planarity, sphericity };
}

//#pragma optimize( "", off )  

void process_file(const std::string& file, const std::string& output_name, bool is_test_file) {
    const int CLASS_NUMBER = 13;
    //read file
    const int max_sample_per_class = 3000;
    std::vector<Eigen::Vector3f> pts_origin;
    std::vector<int32_t> labels_origin;
    bool has_labels = !is_test_file;
    read_point_cloud(file, pts_origin, labels_origin, has_labels);
    const int point_count = pts_origin.size();

    //shuffle points
    std::vector<int> indices(point_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    if (!is_test_file)
        std::shuffle(indices.begin(), indices.end(), g);

    std::vector<Eigen::Vector3f> pts(point_count);
    std::vector<int32_t> labels(point_count);
#pragma omp parallel for  
    for (int i = 0; i < pts.size(); i++) {
        pts[i] = pts_origin[indices[i]];
        if (has_labels)
            labels[i] = labels_origin[indices[i]];
    }

    //select points
    std::vector<int> indices_taken;
    std::vector<int> labels_out;
    labels_out.reserve(pts.size());
    std::vector<int> class_count(CLASS_NUMBER, 0);
    std::vector<Eigen::Vector3f> pts_out;
    pts_out.reserve(pts.size());
    for (int i = 0; i < pts.size(); i++) {
        if (has_labels) {
            int label = labels[i];
            if (class_count[label] >= max_sample_per_class)
                continue;
            class_count[label]++;
            labels_out.push_back(label);
        }
        indices_taken.push_back(i);
        pts_out.push_back(pts[i]);
    }
    const int output_size = indices_taken.size();
    std::cout << output_size << " points selected for features" << std::endl;



    //parameters
    std::vector<float> radii{ 0.25, 0.5, 1, 2, 4 };
    const int SCALE_NUMBER = radii.size();
    const int PCA_FEATURE_N = 6;
    const int BALL_NUMBER = 7;
    const int FEATURE_N = BALL_NUMBER * (PCA_FEATURE_N + 1)*SCALE_NUMBER;
    const float offset_coeff = 1.0;
    std::vector<Eigen::Vector3f> offsets = { {-1,0,0},{1,0,0},{0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
    for (auto& offset : offsets) offset *= offset_coeff;
    //for (int i : {-1, 0, 1})
    //    for (int j : {-1, 0, 1})
    //        for (int k : {-1, 0, 1}) {
    //            if (i != 0 || j != 0 || k != 0) {
    //                Eigen::Vector3f o(i, j, k);
    //                offsets.emplace_back(o*offset_coeff);
    //            }
    //        }

    //build tree
    std::vector<std::unique_ptr<KdTree3D>> trees;
    std::vector<Eigen::Vector3f> pts_shuffle = pts;
    std::shuffle(pts_shuffle.begin(), pts_shuffle.end(), g);
    for (int k = 0; k < SCALE_NUMBER; k++) {
        trees.push_back(std::make_unique<KdTree3D>());
        const float base = 0.5f;
        float r = std::max(radii[k], base);
        float fraction = base*base / r / r;
        int pts_n = std::min(int(point_count * fraction), point_count);
        trees.back()->set_points(pts_shuffle.data(), pts_n);
    }


    std::vector<float> features(output_size * FEATURE_N);
    std::vector<float> features_one(output_size * PCA_FEATURE_N);

    std::cout << "Features calculation ..." << std::endl;
    timepoint before = now();



#pragma omp parallel for  
    for (int i = 0; i < output_size; i++) {
        auto& p = pts_out[i];

        float* f = &features[i*FEATURE_N];
        float* f_one = &features_one[i * PCA_FEATURE_N];

        for (int k = 0; k < radii.size(); k++) {
            float radius = radii[k];
            int nb_count;
            const auto eigen_pair = neighborhood_PCA(p, pts_shuffle, *trees[k], radius, nb_count);
            if (nb_count == 0) nb_count = 1;
            ShapeFeature sf = compute_features_with_eigen(eigen_pair, radius);
            float density_ratio = 1.;
            *f++ = density_ratio;
            *f++ = sf.vertical_spreads;
            *f++ = sf.total_spreads;
            *f++ = sf.verticality;
            *f++ = sf.linearity;
            *f++ = sf.planarity;
            *f++ = sf.sphericity;
            if (k == 1) {
                *f_one++ = sf.vertical_spreads;
                *f_one++ = sf.total_spreads;
                *f_one++ = sf.verticality;
                *f_one++ = sf.linearity;
                *f_one++ = sf.planarity;
                *f_one++ = sf.sphericity;
            }
            const Eigen::Matrix3f& evec = eigen_pair.first;
            Eigen::Vector3f normal = evec.col(0);
            Eigen::Vector3f direction1 = evec.col(2);
            Eigen::Vector3f direction2 = evec.col(1);

            if (normal[2] < 0) normal = -normal;
            if (direction1[2] < 0) direction1 = -direction1;
            if (direction2[2] < 0) direction2 = -direction2;

            //Eigen::Vector3f normal_xy = normal;
            //normal_xy[2] = 0.;
            //normal_xy.normalize();
            //Eigen::Vector3f left(-normal_xy[1], normal_xy[0], 0.);
            //Eigen::Vector3f up(0., 0., 1.);
            //std::uniform_real_distribution<> dis(0., 1.);
            //if (dis(g) > 0.5) {
            //    left *= -1.;
            //}

            //std::vector<Eigen::Vector3f> offsets = { normal,-normal,direction1,-direction1,direction2,-direction2 };
            std::vector<Eigen::Vector3f> axis = { normal, direction1,direction2 };
            for (auto& offset : offsets) {
                int nb_count_offset;
                auto offset_direction = offset[0] * axis[0] + offset[1] * axis[1] + offset[2] * axis[2];
                const auto eigens = neighborhood_PCA(p + radius * offset_direction, pts_shuffle, *trees[k], radius, nb_count_offset);
                ShapeFeature sf = compute_features_with_eigen(eigens, radius);
                float density_ratio = float(nb_count_offset) / nb_count;
                *f++ = density_ratio;
                *f++ = sf.vertical_spreads;
                *f++ = sf.total_spreads;
                *f++ = sf.verticality;
                *f++ = sf.linearity;
                *f++ = sf.planarity;
                *f++ = sf.sphericity;
            }
        }
    }

    timepoint after = now();

    std::cout << "...Done " << difference_second(before, after) << " s: " << std::endl;


    bool binary = true;
    std::ofstream out(output_name, binary ? std::ios::binary : 0);
    tinyply::PlyFile out_file;

    if (has_labels)
        out_file.add_properties_to_element("feature", { "label" }, tinyply::Type::UINT32, labels_out.size(),
            reinterpret_cast<uint8_t*>(labels_out.data()), tinyply::Type::INVALID, 0);

    out_file.add_properties_to_element("feature", { "entries" }, tinyply::Type::FLOAT32, features.size(),
        reinterpret_cast<uint8_t*>(features.data()), tinyply::Type::UINT32, FEATURE_N);
    out_file.write(out, binary);
    out.close();

    if (!is_test_file) {
        std::ofstream out2(std::string("check") + output_name, binary ? std::ios::binary : 0);
        tinyply::PlyFile out2_file;

        out2_file.add_properties_to_element("vertex", { "x", "y", "z" }, tinyply::Type::FLOAT32, pts_out.size() * 3,
            reinterpret_cast<uint8_t*>(pts_out.data()), tinyply::Type::INVALID, 0);
        out2_file.add_properties_to_element("vertex", { "vertical_spreads", "total_spreads", "linearity", "planarity", "sphericity" },
            tinyply::Type::FLOAT32, features_one.size(), reinterpret_cast<uint8_t*>(features_one.data()), tinyply::Type::INVALID, 0);
        out2_file.write(out2, binary);
        out2.close();
    }
}

//#pragma optimize( "", on )   

int main(int argc, char *argv[])
{
    std::string file1("D:/data/Area_3.ply");
    std::string name1("feature_area_3.ply");
    process_file(file1, name1, false);

    std::string filetest("D:/data/Area_4.ply");
    std::string nametest("feature_area_4.ply");
    process_file(filetest, nametest, false);
    return 0;
}
