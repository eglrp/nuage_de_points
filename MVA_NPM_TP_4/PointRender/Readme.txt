PointRender
-----------
Author: Tamy Boubekeur (http://www.telecom-paristech.fr/~boubek)        

Description
------------
PointRender  is a minimal 3D viewer for PN files.

PN file format
------------
PN files are binary 3D point sets with position and normal vector for
each 3D point. There is no header.
A PN file is made of a list of 6 32-bits float chunk (binary) :
  x0,y0,z0,nx0,ny0,nz0,x1,y1,z1,nx1,ny1,nz1,...
with {x,y,z} the position and {nx,ny,nz} the normal vector of each point sample.

Installation
------------
Requirement: c++ compiler, OpenGL 1.2, GLUT library
Compile on Linux : cd <path-to-PointRender> ; make
Compile on Windows/Cygwin : edit the Makefile to load opengl32, GLU32 and Glut32, then same as Linux.
Run: ./PointRender <file.pn>
The archive contains few PN files for testing in the data directory.

Keyboard commands
------------------
 ?: Print help
 f: Toggle full screen mode
 +/-: Increase/Decrease screen point size
 <drag>+<left button>: rotate model
 <drag>+<right button>: move model
 <drag>+<middle button>: zoom
 q, <esc>: Quit

