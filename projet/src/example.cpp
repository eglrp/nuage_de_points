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
    std::shared_ptr<tinyply::PlyData> vertices, labels;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the file header prior to reading the data. For brevity of this sample, properties 
    // like vertex position are hard-coded: 
    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { labels = file.request_properties_from_element("vertex", { "class" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


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

    const size_t numLablesBytes = labels->buffer.size_bytes();
    ls.resize(labels->count);
    std::memcpy(ls.data(), labels->buffer.get(), numLablesBytes);

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

std::pair<Eigen::Matrix3f, Eigen::Vector3f> neighborhood_PCA(
    const Eigen::Vector3f& query, KdTree3D& tree, float radius, int& nb_count) {
    std::vector<Eigen::Vector3f> nbs;
    tree.query_radius(query, radius, nbs);
    const int N = nbs.size();
    nb_count = N;
    Eigen::Matrix3f evec;
    Eigen::Vector3f evar;
    local_pca(nbs, evec, evar);
    return std::make_pair(evec, evar);
}

const int PCA_FEATURE_N = 6;
typedef std::array<float, PCA_FEATURE_N> ShapeFeature;
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

void grid_subsample(const std::vector<Eigen::Vector3f>& pts, const std::vector<int32_t>& labels,
    std::vector<Eigen::Vector3f>& sub_pts, std::vector<int32_t>& sub_labels, float size) {
    sub_pts.clear();
    sub_labels.clear();
    //Eigen::Vector3f max_corner(-1e100, -1e100, -1e100);
    //Eigen::Vector3f min_corner(1e100, 1e100, 1e100);
    //for (const auto& p : pts) {
    //    for (int i = 0; i < 3; i++) {
    //        max_corner[i] = std::max(max_corner[i], p[i]);
    //        min_corner[i] = std::min(min_corner[i], p[i]);
    //    }
    //}
    typedef std::array<int, 3> Index;
    typedef int32_t label_t;
    std::map<Index, std::map<label_t, std::vector<Eigen::Vector3f> > > grids;
    for (int a = 0; a < pts.size(); a++) {
        Eigen::Vector3f ijk = (pts[a] / size).array().floor();
        int i = ijk[0], j = ijk[1], k = ijk[2];
        grids[{i, j, k}][labels[a]].push_back(pts[a]);
    }
    for (auto& it : grids) {
        const auto& points_by_classes = it.second;
        for (auto& itc : points_by_classes) {
            label_t cls = itc.first;
            const std::vector<Eigen::Vector3f>& points = itc.second;
            Eigen::Vector3f s(0.f, 0.f, 0.f);
            for (const auto& p : points)
                s += p;
            s /= points.size();
            sub_pts.push_back(s);
            sub_labels.push_back(cls);
        }
    }
}

//#pragma optimize( "", off )  

void process_file(const std::string& file, const std::string& output_name) {
    const int RAW_CLASS_NUMBER = 27;
    const int CLASS_NUMBER = 6;
    std::vector<int> class_mapping(RAW_CLASS_NUMBER, -1);
    class_mapping[1] = 0;//Facade
    class_mapping[2] = 1;//Ground
    class_mapping[4] = 2;//Cars
    class_mapping[10] = 3;//Moto
    class_mapping[14] = 4;//Traffic signs
    class_mapping[9] = 5;//Pedestrians
    class_mapping[22] = 5;//Pedestrians
    class_mapping[24] = 5;//Pedestrians

    //read file
    const int max_sample_per_class = 3000;
    std::vector<Eigen::Vector3f> pts;
    std::vector<int32_t> labels;
    {
        ScopeTimer t("read file", true, false);
        read_point_cloud(file, pts, labels);
    }

    //select points
    std::vector<int> labels_select;
    std::vector<Eigen::Vector3f> pts_select;
    {
        ScopeTimer t("select pts per class", true, false);

        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<int> indices(pts.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        labels_select.reserve(pts.size());
        std::vector<int> class_count(CLASS_NUMBER, 0);
        pts_select.reserve(pts.size());
        for (int i : indices) {
            int label = class_mapping[labels[i]];
            if (label == -1) continue;
            if (label >= CLASS_NUMBER) {
                std::cout << "wrong class number : " << label << std::endl;
                exit(1);
            }
            if (class_count[label] >= max_sample_per_class)
                continue;

            class_count[label]++;
            labels_select.push_back(label);
            pts_select.push_back(pts[i]);
        }
        std::cout << pts_select.size() << " points selected for features : " << std::endl;
        for (int i = 0; i < CLASS_NUMBER; i++)
            std::cout << class_count[i] << " points for class " << i << std::endl;
        std::cout << std::endl;

    }

    //parameters
    std::vector<float> radii{ 0.25, 0.5, 1, 2, 4 };
    std::vector<float> grid_size{ 0.0125f, 0.025f, 0.05f, 0.1f, 0.2f };
    const int SCALE_NUMBER = radii.size();
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
    {
        ScopeTimer t("build tree", true, false);
        std::cout << pts.size() << " points in total" << std::endl;
        for (int k = 0; k < SCALE_NUMBER; k++) {
            trees.push_back(std::make_unique<KdTree3D>());
            std::vector<Eigen::Vector3f> sub_pts;
            std::vector<int32_t> sub_labels;
            grid_subsample(pts, labels, sub_pts, sub_labels, grid_size[k]);
            std::cout << sub_pts.size() << " points for scale " << radii[k] << std::endl;
            trees.back()->set_points(sub_pts);
        }
    }


    std::vector<float> features(pts_select.size() * FEATURE_N);
    std::vector<float> features_one(pts_select.size() * PCA_FEATURE_N);

    std::cout << "Features calculation ..." << std::endl;
    timepoint before = now();



#pragma omp parallel for  
    for (int i = 0; i < pts_select.size(); i++) {
        auto& p = pts_select[i];

        float* f = &features[i*FEATURE_N];
        float* f_one = &features_one[i * PCA_FEATURE_N];

        for (int k = 0; k < radii.size(); k++) {
            float radius = radii[k];
            int nb_count;
            const auto eigen_pair = neighborhood_PCA(p, *trees[k], radius, nb_count);
            if (nb_count == 0) nb_count = 1;
            ShapeFeature sf = compute_features_with_eigen(eigen_pair, radius);
            float density_ratio = 1.;
            *f++ = density_ratio;
            for (int s = 0; s < PCA_FEATURE_N; s++)
                *f++ = sf[s];
            if (k == 1) {
                for (int s = 0; s < PCA_FEATURE_N; s++)
                    *f_one++ = sf[s];
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
                const auto eigens = neighborhood_PCA(p + radius * offset_direction, *trees[k], radius, nb_count_offset);
                ShapeFeature sf = compute_features_with_eigen(eigens, radius);
                float density_ratio = float(nb_count_offset) / nb_count;
                *f++ = density_ratio;
                for (int s = 0; s < PCA_FEATURE_N; s++)
                    *f++ = sf[s];
            }
        }
    }

    timepoint after = now();

    std::cout << "...Done " << difference_second(before, after) << " s: " << std::endl;


    bool binary = true;
    std::ofstream out(output_name, binary ? std::ios::binary : 0);
    tinyply::PlyFile out_file;

    {
        out_file.add_properties_to_element("feature", { "label" }, tinyply::Type::UINT32, labels_select.size(),
            reinterpret_cast<uint8_t*>(labels_select.data()), tinyply::Type::INVALID, 0);

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
        out2_file.add_properties_to_element("feature", { "label" }, tinyply::Type::UINT32, labels_select.size(),
            reinterpret_cast<uint8_t*>(labels_select.data()), tinyply::Type::INVALID, 0);
        out2_file.add_properties_to_element("vertex", { "vertical_spreads", "total_spreads", "verticality", "linearity", "planarity", "sphericity" },
            tinyply::Type::FLOAT32, features_one.size(), reinterpret_cast<uint8_t*>(features_one.data()), tinyply::Type::INVALID, 0);
        out2_file.write(out2, binary);
        out2.close();
    }
}

//#pragma optimize( "", on )   

int main(int argc, char *argv[])
{
    //std::string file1("D:/data/rueMadame/GT_Madame1_2.ply");
    //std::string name1("madame_1.ply");
    //process_file(file1, name1);

    std::string file2("D:/data/rueMadame/GT_Madame1_3.ply");
    std::string name2("madame_2.ply");
    process_file(file2, name2);
    return 0;
}
