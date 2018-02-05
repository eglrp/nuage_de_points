#pragma once
#include <memory>
#include <vector>
#include "Vec3.h"

class KdTree3D
{
    class Impl;
    std::unique_ptr<Impl> pImpl;
public:
    KdTree3D();
    ~KdTree3D();
    void set_points(const std::vector<Vec3f>& points);
    std::vector<size_t> query(Vec3f p, int k) const;
};
