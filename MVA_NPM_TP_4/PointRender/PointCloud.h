// -------------------------------------------
// Simple point cloud object.
// Copyright (C) 2015 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "Exception.h"
#include "Vec3.h"
/// A basic point cloud class.
class PointCloud {
public:
    class Point {
    public:
        inline Point (const Vec3f & position = Vec3f (0.f, 0.f, 0.f), 
                      const Vec3f & normal = Vec3f (1.f, 0.f, 0.f)) : 
                      _position (position), _normal (normal) {}
        inline Point (const Point & p) : _position (p._position), _normal (p._normal) {}
        inline ~Point () {}
        inline Vec3f & position () { return _position; }
        inline const Vec3f & position () const { return _position; }
        inline void setPosition (const Vec3f & p) { _position = p; }
        inline Vec3f & normal () { return _normal; }
        inline const Vec3f & normal () const { return _normal; }
        inline void setNormal (const Vec3f & n) { _normal = n; }
    private:
        Vec3f _position;
        Vec3f _normal;
    };
    inline Point & operator() (unsigned int i) { return _points[i]; }
    inline const Point & operator() (unsigned int i) const { return _points[i]; }
    inline unsigned int size () const { return _points.size () ;}
    inline void loadPN (const std::string & filename);
    inline void centerAndScaleToUnit ();
private:
    std::vector<Point> _points;
};

static const unsigned int READ_BUFFER_SIZE = 1024;

void PointCloud::loadPN (const std::string & filename) {
    unsigned int surfelSize = 6;
#ifdef _MSC_VER
    FILE * in;
    if (fopen_s(&in, filename.c_str(), "rb"))
        throw Exception (filename + " is not a valid PN file.");
#else
    FILE * in = fopen (filename.c_str (), "rb");
    if (in == NULL)
        throw Exception (filename + " is not a valid PN file.");
#endif
    float * pn = new float[surfelSize*READ_BUFFER_SIZE];
    _points.clear ();
    while (!feof (in)) {
        unsigned numOfPoints = fread (pn, 4, surfelSize*READ_BUFFER_SIZE, in);
        for (unsigned int i = 0; i < numOfPoints; i += surfelSize) 
            _points.push_back (Point (Vec3f (pn[i], pn[i+1], pn[i+2]), 
                                      Vec3f (pn[i+3], pn[i+4], pn[i+5])));
        
        if (numOfPoints < surfelSize*READ_BUFFER_SIZE)
            break;
    }
    fclose (in);
    delete [] pn;
    centerAndScaleToUnit ();
}

void PointCloud::centerAndScaleToUnit () {
    Vec3f c;
    for  (unsigned int i = 0; i < _points.size (); i++)
        c += _points[i].position ();
    c /= _points.size ();
    float maxD = dist (_points[0].position (), c);
    for (unsigned int i = 0; i < _points.size (); i++){
        float m = dist (_points[i].position (), c);
        if (m > maxD)
            maxD = m;
    }
    for  (unsigned int i = 0; i < _points.size (); i++) {
        const Vec3f &p = _points[i].position ();
        _points[i].setPosition ((p - c) / maxD);
    }
}
