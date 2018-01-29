// -------------------------------------------
// Simple point cloud object.
// Copyright (C) 2015 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

#pragma once
#include <string>

/// Exceptions should be thrown for any system issue, such as I/O or memory problems.
class Exception {
private:
    std::string msg;
public:
    inline Exception ( const std::string & message) : msg (message) {}
    virtual ~Exception () {}
    inline const std::string & message () const { return msg; }
};