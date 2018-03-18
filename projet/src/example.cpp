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
#include <array>
#include <map>
#include "scope_timer.h"
#include <set>
#include <omp.h>
typedef std::chrono::time_point<std::chrono::high_resolution_clock> timepoint;

inline std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
    return std::chrono::high_resolution_clock::now();
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

void read_point_cloud(const std::string & filename, std::vector<Eigen::Vector3f>& verts, std::vector<int32_t>& ls)
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
    std::shared_ptr<tinyply::PlyData> vertices, classes;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the file header prior to reading the data. For brevity of this sample, properties 
    // like vertex position are hard-coded: 
    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { classes = file.request_properties_from_element("vertex", { "class" }); }
    catch (const std::exception & e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
        std::cout << "load property 'scalar_class' as class instead.." << std::endl;
        try { classes = file.request_properties_from_element("vertex", { "scalar_class" }); }
        catch (const std::exception & e) {
            std::cerr << "tinyply exception: " << e.what() << std::endl;
            exit(1);
        }
    }


    timepoint before = now();
    file.read(ss);
    timepoint after = now();

    // Good place to put a breakpoint!
    std::cout << "Parsing took " << difference_millis(before, after) << " ms: " << std::endl;
    if (vertices) std::cout << "\tRead " << vertices->count << " total vertices " << std::endl;
    if (classes) std::cout << "\tRead " << classes->count << " total labels " << std::endl;

    const size_t numVerticesBytes = vertices->buffer.size_bytes();
    verts.resize(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
    {
        const size_t numLablesBytes = classes->buffer.size_bytes();
        ls.resize(classes->count);
        if (classes->t == tinyply::Type::UINT32 || classes->t == tinyply::Type::INT32) {
            std::memcpy(ls.data(), classes->buffer.get(), numLablesBytes);
        }
        else if (classes->t == tinyply::Type::FLOAT32) {
            std::cout << "convert float class to uint32" << std::endl;
            std::vector<float> float_classes(classes->count);
            std::memcpy(float_classes.data(), classes->buffer.get(), numLablesBytes);
            for (int i = 0; i < classes->count; i++) {
                ls[i] = int32_t(float_classes[i]);
            }
        }
        else {
            std::cout << "unacceptable class type" << std::endl;
            exit(1);
        }
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

enum Method { TIMO, LEMAN, LEMAN_2 };
const Method METHOD = LEMAN_2;

template <Method method>
struct MethodTraits
{
    static const int SCALE_N;
    static const int PCA_FEATURE_N;
    static const int BALL_FEATURE_N;
    using ShapeFeature = std::array<float, PCA_FEATURE_N>;

    static const bool nb_of_nb;
    static const bool use_knn; // k-NN or radius query
    static const int k_for_knn; // k of k-NN


    static std::vector<float> radii();
    static std::vector<float> grid_size();
    static ShapeFeature compute_features_with_eigen(
        const Eigen::Vector3f& p, const std::vector<Eigen::Vector3f>& nbs,
        const Eigen::Matrix3f& evec, const Eigen::Vector3f& evar, float radius);
};



template <>
struct MethodTraits<LEMAN>
{
    static const int SCALE_N = 4;
    static const int PCA_FEATURE_N = 6;
    static const int BALL_FEATURE_N = PCA_FEATURE_N + 1;
    using ShapeFeature = std::array<float, PCA_FEATURE_N>;

    static const bool nb_of_nb = true;
    static const bool use_knn = false;
    static const int k_for_knn = 0; // k of k-NN

    static std::vector<float> radii() {
        return{ 0.1f, 0.3f, 0.9f, 2.7f };
    }
    static std::vector<float> grid_size() {
        return{ 0.025f, 0.075f, 0.225f, 0.675f };
    }
    static ShapeFeature compute_features_with_eigen(
        const Eigen::Vector3f& p, const std::vector<Eigen::Vector3f>& nbs,
        const Eigen::Matrix3f& evec, const Eigen::Vector3f& evar, float radius) {
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

        return{ vertical_spreads, total_spreads, verticality, linearity, planarity, sphericity };
    }
};

template <>
struct MethodTraits<LEMAN_2>
{
    static const int SCALE_N = 4;
    static const int PCA_FEATURE_N = 5;
    static const int BALL_FEATURE_N = PCA_FEATURE_N + 1;
    using ShapeFeature = std::array<float, PCA_FEATURE_N>;

    static const bool nb_of_nb = true;
    static const bool use_knn = false;
    static const int k_for_knn = 0; // k of k-NN

    static std::vector<float> radii() {
        return{ 0.1f, 0.3f, 0.9f, 2.7f };
    }
    static std::vector<float> grid_size() {
        return{ 0.025f, 0.075f, 0.225f, 0.675f };
    }
    static ShapeFeature compute_features_with_eigen(
        const Eigen::Vector3f& p, const std::vector<Eigen::Vector3f>& nbs,
        const Eigen::Matrix3f& evec, const Eigen::Vector3f& evar, float radius) {
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
        float total_spreads = denom / radius / radius;

        if (!std::isfinite(total_spreads)) throw "error";
        if (!std::isfinite(verticality)) throw "error";

        return{ total_spreads, l1, l2, l3, verticality };
    }
};

template <>
struct MethodTraits<TIMO>
{
    static const int SCALE_N = 9;
    static const int PCA_FEATURE_N = 13;
    static const int BALL_FEATURE_N = PCA_FEATURE_N;
    using ShapeFeature = std::array<float, PCA_FEATURE_N>;

    static const bool nb_of_nb = false;
    static const bool use_knn = true;
    static const int k_for_knn = 10; // k of k-NN

    static std::vector<float> radii() {
        return{ 0,0,0,0,0,0,0,0,0 };
    }
    static std::vector<float> grid_size() {
        return{ 0.025f, 0.05f, 0.1f, 0.2f, 0.4f, 0.8f, 1.6f, 3.2f, 6.4f };
    }
    static ShapeFeature compute_features_with_eigen(
        const Eigen::Vector3f& p, const std::vector<Eigen::Vector3f>& nbs,
        const Eigen::Matrix3f& evec, const Eigen::Vector3f& evar, float radius) {
        float l1 = evar[2];
        float l2 = evar[1];
        float l3 = evar[0];
        l3 = std::max(0.f, l3);
        float l_sum = l1 + l2 + l3 + 1e-6f;

        l1 /= l_sum;
        l2 /= l_sum;
        l3 /= l_sum;

        float eigen_entropy = -l1*std::log(l1) - l2*std::log(l2) - l3*std::log(l3 + 1e-6f);
        float omnivariance = std::cbrt(l1*l2*l3);

        Eigen::Vector3f normal = evec.col(0);
        Eigen::Vector3f ez(0., 0., 1.);
        float verticality = 1.0 - std::min(std::abs(normal.dot(ez)), 1.0f);

        l1 += 1e-6f;
        float anisotropy = (l1 - l3) / l1;
        float surface_variation = l3 / (l1 + l2 + l3);
        float linearity = 1. - l2 / l1;
        float planarity = (l2 - l3) / l1;
        float sphericity = l3 / l1;

        Eigen::Vector3f e1 = evec.col(2);
        Eigen::Vector3f e2 = evec.col(1);
        float moment_1o_1a = 0.;
        float moment_1o_2a = 0.;
        float moment_2o_1a = 0.;
        float moment_2o_2a = 0.;
        for (const auto& pi : nbs) {
            auto d = pi - p;
            float m1 = d.dot(e1);
            float m2 = d.dot(e2);
            moment_1o_1a += m1;
            moment_1o_2a += m2;
            moment_2o_1a += m1*m1;
            moment_2o_2a += m2*m2;
        }

        if (!std::isfinite(moment_1o_1a)) throw "error";
        if (!std::isfinite(moment_1o_2a)) throw "error";
        if (!std::isfinite(moment_2o_1a)) throw "error";
        if (!std::isfinite(moment_2o_2a)) throw "error";
        if (!std::isfinite(verticality)) throw "error";
        if (!std::isfinite(linearity)) throw "error";
        if (!std::isfinite(planarity)) throw "error";
        if (!std::isfinite(sphericity)) throw "error";
        if (!std::isfinite(eigen_entropy)) throw "error";
        if (!std::isfinite(omnivariance)) throw "error";
        if (!std::isfinite(anisotropy)) throw "error";
        if (!std::isfinite(surface_variation)) throw "error";

        return{ l_sum, omnivariance, eigen_entropy, anisotropy,
            planarity, linearity, surface_variation, sphericity, verticality,
            moment_1o_1a, moment_1o_2a, moment_2o_1a, moment_2o_2a };
    }
};




struct FeatureReport
{
    int nb_count;
    MethodTraits<METHOD>::ShapeFeature f;
    Eigen::Matrix3f evec;
    Eigen::Vector3f evar;
};



void neighborhood_PCA_and_feature(
    const Eigen::Vector3f& query, KdTree3D& tree, int k_for_knn, float radius, FeatureReport& feature_report) {
    std::vector<Eigen::Vector3f> nbs;
    if (MethodTraits<METHOD>::use_knn) {
        tree.query_k(query, k_for_knn, nbs);
    }
    else {
        tree.query_radius(query, radius, nbs);
    }
    feature_report.nb_count = nbs.size();
    local_pca(nbs, feature_report.evec, feature_report.evar);
    const Eigen::Matrix3f& evec = feature_report.evec;
    const Eigen::Vector3f& evar = feature_report.evar;

    //features 
    feature_report.f = MethodTraits<METHOD>::compute_features_with_eigen(query, nbs, evec, evar, radius);
}



void grid_subsample(const std::vector<Eigen::Vector3f>& pts, const std::vector<int32_t>& labels,
    std::vector<Eigen::Vector3f>& sub_pts, std::vector<int32_t>& sub_labels, float size, int class_number) {
    sub_pts.clear();
    sub_labels.clear();

    typedef std::array<int, 3> Index;
    typedef int32_t label_t;
    std::vector<std::map<Index, std::vector<Eigen::Vector3f> > >  grids(class_number);
    {
        ScopeTimer t("Assigning grids");
        for (int a = 0; a < pts.size(); a++) {
            Eigen::Vector3f ijk = (pts[a] / size).array().floor();
            int i = ijk[0], j = ijk[1], k = ijk[2];
            grids[labels[a]][{i, j, k}].push_back(pts[a]);
        }
    }
    {
        ScopeTimer t("merge grids");
        for (int c = 0; c < class_number; c++)
            for (auto& it : grids[c]) {
                const auto& points = it.second;
                Eigen::Vector3f s(0.f, 0.f, 0.f);
                for (const auto& p : points)
                    s += p;
                s /= points.size();
                sub_pts.push_back(s);
                sub_labels.push_back(c);
            }
    }
}

//#pragma optimize( "", off )  

void process_file(const std::string& file, const std::string& output_name) {
    const int CLASS_NUMBER = 12;

    std::random_device rd;
    std::mt19937 g(rd());

    //read file
    const int max_sample_per_class = 10000;
    std::vector<Eigen::Vector3f> pts;
    std::vector<int32_t> classes;
    {
        ScopeTimer t("read file", true, false);
        read_point_cloud(file, pts, classes);
    }

    //select points
    std::vector<int> classes_select;
    std::vector<Eigen::Vector3f> pts_select;
    {
        ScopeTimer t("select pts per class", true, false);


        std::vector<int> indices(pts.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        classes_select.reserve(pts.size());
        std::vector<int> class_count(CLASS_NUMBER, 0);
        pts_select.reserve(pts.size());
        for (int i : indices) {
            int class_ = classes[i];
            if (class_ == 0) {
                continue; // unclassified point
            }
            if (class_ >= CLASS_NUMBER || class_ < 0) {
                std::cout << "wrong class number : " << class_ << std::endl;
                exit(1);
            }
            if (class_count[class_] >= max_sample_per_class)
                continue;

            class_count[class_]++;
            classes_select.push_back(class_);
            pts_select.push_back(pts[i]);
        }
        std::cout << pts_select.size() << " points selected for features : " << std::endl;
        for (int i = 0; i < CLASS_NUMBER; i++)
            std::cout << class_count[i] << " points for class " << i << std::endl;
        std::cout << std::endl;

    }

    //parameters
    std::vector<float> radii = MethodTraits<METHOD>::radii();
    std::vector<float> grid_size = MethodTraits<METHOD>::grid_size();
    const int PCA_FEATURE_N = MethodTraits<METHOD>::PCA_FEATURE_N;
    const int SCALE_NUMBER = MethodTraits<METHOD>::SCALE_N;
    const int BALL_NUMBER = MethodTraits<METHOD>::nb_of_nb ? 7 : 1;
    const int FEATURE_N = BALL_NUMBER * MethodTraits<METHOD>::BALL_FEATURE_N * SCALE_NUMBER;
    const float offset_coeff = 1.5;
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
    {
        ScopeTimer t("build tree", true, false);
        std::cout << pts.size() << " points in total" << std::endl;
        for (int scale = 0; scale < SCALE_NUMBER; scale++) {
            trees.push_back(std::make_unique<KdTree3D>());
            std::vector<Eigen::Vector3f> sub_pts;
            std::vector<int32_t> sub_labels;
            grid_subsample(pts, classes, sub_pts, sub_labels, grid_size[scale], CLASS_NUMBER);
            std::cout << sub_pts.size() << " points for scale " << scale << std::endl;

            {
                ScopeTimer t("build tree");
                trees.back()->set_points(sub_pts);
            }
        }
    }


    std::vector<float> features(pts_select.size() * FEATURE_N);
    std::vector<float> features_one(pts_select.size() * PCA_FEATURE_N);

    {
        ScopeTimer t("Features calculation", true, false);
        int points_in_ball[SCALE_NUMBER] = { 0 };
        int balls_n[SCALE_NUMBER] = { 0 };
#pragma omp parallel for  
        for (int i = 0; i < pts_select.size(); i++) {
            auto& p = pts_select[i];

            float* f = &features[i*FEATURE_N];
            float* f_one = &features_one[i * PCA_FEATURE_N];

            for (int scale = 0; scale < SCALE_NUMBER; scale++) {
                float radius = radii[scale];
                FeatureReport fr;
                neighborhood_PCA_and_feature(p, *trees[scale], MethodTraits<METHOD>::k_for_knn, radius, fr);
                int nb_count = fr.nb_count;
                points_in_ball[scale] += nb_count;
                balls_n[scale]++;
                if (nb_count == 0) nb_count = 1;
                if (!MethodTraits<METHOD>::use_knn) {
                    float density_ratio = 1.;
                    *f++ = density_ratio;
                }
                for (int s = 0; s < PCA_FEATURE_N; s++)
                    *f++ = fr.f[s];
                if (scale == 3) {
                    for (int s = 0; s < PCA_FEATURE_N; s++)
                        *f_one++ = fr.f[s];
                }

                if (MethodTraits<METHOD>::nb_of_nb) {
                    Eigen::Vector3f normal = fr.evec.col(0);
                    Eigen::Vector3f direction1 = fr.evec.col(2);
                    Eigen::Vector3f direction2 = fr.evec.col(1);

                    if (normal[2] < 0) normal = -normal;
                    if (direction1[2] < 0) direction1 = -direction1;
                    if (direction2[2] < 0) direction2 = -direction2;


                    std::vector<Eigen::Vector3f> axis = { normal, direction1, direction2 };
                    bool flat = false;
                    if (flat) {
                        Eigen::Vector3f normal_xy = normal;
                        normal_xy[2] = 0.;
                        normal_xy.normalize();
                        Eigen::Vector3f left(-normal_xy[1], normal_xy[0], 0.);
                        Eigen::Vector3f up(0., 0., 1.);
                        std::uniform_real_distribution<> dis(0., 1.);
                        if (dis(g) > 0.5) {
                            left *= -1.;
                        }
                        axis = { normal_xy, left, up };
                    }

                    for (auto& offset : offsets) {
                        FeatureReport fr_off;
                        auto offset_direction = offset[0] * axis[0] + offset[1] * axis[1] + offset[2] * axis[2];
                        neighborhood_PCA_and_feature(p + radius * offset_direction, *trees[scale], MethodTraits<METHOD>::k_for_knn, radius, fr_off);
                        if (!MethodTraits<METHOD>::use_knn) {
                            float density_ratio = float(fr_off.nb_count) / nb_count;
                            *f++ = density_ratio;
                        }

                        for (int s = 0; s < PCA_FEATURE_N; s++)
                            *f++ = fr_off.f[s];
                    }
                }
            }


        }
        std::cout << "averaged points in ball for scales :" << std::endl;
        for (int i = 0; i < SCALE_NUMBER; i++) {
            std::cout << "scale " << i << " " << double(points_in_ball[i]) / balls_n[i] << std::endl;
        }
    }

    {
        ScopeTimer t("Write to file", true, false);

        bool binary = true;
        std::ofstream out(output_name, binary ? std::ios::binary : 0);
        tinyply::PlyFile out_file;

        {
            out_file.add_properties_to_element("feature", { "class" }, tinyply::Type::UINT32, classes_select.size(),
                reinterpret_cast<uint8_t*>(classes_select.data()), tinyply::Type::INVALID, 0);

            out_file.add_properties_to_element("feature", { "entries" }, tinyply::Type::FLOAT32, features.size(),
                reinterpret_cast<uint8_t*>(features.data()), tinyply::Type::UINT32, FEATURE_N);
            out_file.write(out, binary);
            out.close();
        }
        {
            std::ofstream out2(std::string("check_") + output_name, binary ? std::ios::binary : 0);
            tinyply::PlyFile out2_file;

            out2_file.add_properties_to_element("vertex", { "x", "y", "z" }, tinyply::Type::FLOAT32, pts_select.size() * 3,
                reinterpret_cast<uint8_t*>(pts_select.data()), tinyply::Type::INVALID, 0);
            out2_file.add_properties_to_element("feature", { "class" }, tinyply::Type::UINT32, classes_select.size(),
                reinterpret_cast<uint8_t*>(classes_select.data()), tinyply::Type::INVALID, 0);
            if (METHOD == TIMO)
                out2_file.add_properties_to_element("vertex", { "l_sum", "omnivariance", "eigen_entropy", "anisotropy",
                    "planarity", "linearity", "surface_variation", "sphericity", "verticality",
                    "moment_1o_1a", "moment_1o_2a", "moment_2o_1a", "moment_2o_2a" },
                    tinyply::Type::FLOAT32, features_one.size(), reinterpret_cast<uint8_t*>(features_one.data()), tinyply::Type::INVALID, 0);
            if (METHOD == LEMAN)
                out2_file.add_properties_to_element("vertex", { "vertical_spreads", "total_spreads", "verticality", "linearity", "planarity", "sphericity" },
                    tinyply::Type::FLOAT32, features_one.size(), reinterpret_cast<uint8_t*>(features_one.data()), tinyply::Type::INVALID, 0);
            out2_file.write(out2, binary);
            out2.close();
        }
    }
}

//#pragma optimize( "", on )   
std::map<int, int> get_class_mapping_for_Paris_rue_Madame() {
    std::map<int, int> class_map;
    class_map[1] = 1;//Facade
    class_map[2] = 2;//Ground
    class_map[4] = 3;//Cars
    class_map[10] = 4;//Moto
    class_map[14] = 5;//Traffic signs
    class_map[9] = 6;//Pedestrians
    class_map[22] = 6;//Pedestrians
    class_map[24] = 6;//Pedestrians
    return class_map;
}
std::map<int, int> get_class_mapping_for_Paris_Lille_3D_data() {
    std::map<int, int> class_map;
    class_map[0] = 0; //unknown
    class_map[100000000] = 0; //unknown
    class_map[202020000] = 1; //road => ground
    class_map[202030000] = 1; //sidewalk => ground
    class_map[202050000] = 1; //island => ground
    class_map[202060000] = 1; //vegetal ground => ground
    class_map[203000000] = 2; //building
    class_map[302020300] = 3; //bollard
    class_map[302020400] = 4; //floor lamp
    class_map[302020500] = 5; //traffic light
    class_map[302020600] = 6; //traffic sign => sign
    class_map[302020700] = 6; //signboard => sign
    class_map[302030300] = 7; // roasting
    class_map[302030600] = 8; //wire
    class_map[303040000] = 9; //4+ wheels
    class_map[303040100] = 9; //4+ wheels
    class_map[303040200] = 9; //4+ wheels
    class_map[303040201] = 9; //4+ wheels
    class_map[303040202] = 9; //4+ wheels
    class_map[303040203] = 9; //4+ wheels
    class_map[303040204] = 9; //4+ wheels
    class_map[303040205] = 9; //4+ wheels
    class_map[303040206] = 9; //4+ wheels
    class_map[303040207] = 9; //4+ wheels
    class_map[303040208] = 9; //4+ wheels
    class_map[303040209] = 9; //4+ wheels
    class_map[303040300] = 9; //4+ wheels
    class_map[303040304] = 9; //4+ wheels
    class_map[303050500] = 10; //trash can
    class_map[304000000] = 11; //natural
    class_map[304020000] = 11; //tree => natural
    class_map[304040000] = 11; //potted plant => natural
    return class_map;
}

void class_mapping(const std::string& file, const std::string& output_name) {
    // for Paris-Lille-3D dataset 
    //std::map<int, int> class_map = get_class_mapping_for_Paris_Lille_3D_data();
    std::map<int, int> class_map = get_class_mapping_for_Paris_rue_Madame();


    std::set<int> unclassified_class;

    int class_count = 0;
    std::vector<Eigen::Vector3f> pts;
    std::vector<int32_t> labels;
    std::vector<int32_t> new_labels;
    {
        ScopeTimer t("read file", true, false);
        read_point_cloud(file, pts, labels);
        new_labels.resize(labels.size());
    }
    for (int i = 0; i < pts.size(); i++) {
        int32_t l = labels[i];
        auto it = class_map.find(l);
        if (it == class_map.end()) {
            new_labels[i] = 0;

            auto it2 = unclassified_class.find(l);
            if (it2 == unclassified_class.end()) {
                unclassified_class.insert(l);
                std::cout << "unclassified class : " << l << std::endl;
            }
        }
        else {
            new_labels[i] = class_map[l];
        }
    }
    std::cout << class_count << " classes found" << std::endl;

    {
        std::ofstream out(std::string("class_remap_") + output_name, std::ios::binary);
        tinyply::PlyFile out_file;

        out_file.add_properties_to_element("vertex", { "x", "y", "z" }, tinyply::Type::FLOAT32, pts.size() * 3,
            reinterpret_cast<uint8_t*>(pts.data()), tinyply::Type::INVALID, 0);
        out_file.add_properties_to_element("vertex", { "class" }, tinyply::Type::UINT32, new_labels.size(),
            reinterpret_cast<uint8_t*>(new_labels.data()), tinyply::Type::INVALID, 0);

        out_file.write(out, true);
        out.close();
    }
    {
        std::ofstream out(std::string("class_remap_annotation_") + output_name);
        for (auto it : class_map) {
            out << it.first << " => " << it.second << std::endl;
        }
        out << "unclassified:" << std::endl;
        for (auto it : unclassified_class) {
            out << it << std::endl;
        }
        out.close();
    }
}
void test() {

    const int N = 8;
    std::vector<std::vector<int>> numbers(N);
    std::cout << N << "threads in total" << std::endl;
#pragma omp parallel for num_threads(N)
    for (int i = 0; i < 100; i++) {
        int threadnum = omp_get_thread_num();
        numbers[threadnum].push_back(i);
    }

    for (auto& v : numbers) {
        for (auto& n : v)
            std::cout << n << " ";
        std::cout << std::endl;
    }
}
int main(int argc, char *argv[])
{
    //test();
    //class_mapping("D:/data/rueMadame/GT_Madame1_2.ply", "madame_1.ply");
    //class_mapping("D:/data/rueMadame/GT_Madame1_3.ply", "madame_2.ply");

    std::string file1("D:/data/LilleStreetClass/Lille1_part1.ply");
    std::string name1("Lille1_part1.ply");
    process_file(file1, name1);

    std::string file2("D:/data/LilleStreetClass/Lille1_part2.ply");
    std::string name2("Lille1_part2.ply");
    process_file(file2, name2);
    return 0;
}
