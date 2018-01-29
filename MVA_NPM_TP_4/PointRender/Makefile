# Makefile pour un unique exécutable

# liste des variables à renseigner
#   CIBLE : nom du programme ( $(CIBLE).c doit contenir main() )
#   SRCS : ensemble des fichiers sources 
#   LIBS : liste des bibliothèques utiles à l'édition des liens 
#          (format : -lnom1 -lnom2 ...) 
#   PREFIX : chemin de la hiérarchie 
#
# NE PAS OUBLIER D'AJOUTER LA LISTE DES DEPENDANCES A LA FIN DU FICHIER

CIBLE = PointRender
SRCS =  Camera.cpp Trackball.cpp PointRender.cpp

## Commenter la ligne suivante et décommenter celle qui suit pour compiler sous Windows/Cygwin
LIBS =  -lglut -lGLU -lGL -lm
#LIBS =  -lopengl32 -lGLU32 -lglut32 -lm

#########################################################"

INCDIR = .
LIBDIR = .
BINDIR = .

# nom du compilateur
CC = g++
CPP = g++

# options du compilateur          
CFLAGS = -Wall -O3 
CXXFLAGS = -Wall -O3 

# option du preprocesseur
CPPFLAGS =  -I$(INCDIR) 

# options du linker et liste des bibliothèques à charger
LDFLAGS = -L/usr/X11R6/lib              
LDLIBS = -L$(LIBDIR) $(LIBS)  

# construire la liste des fichiers objets une nouvelle chaine à partir
# de SRCS en substituant les occurences de ".c" par ".o" 
OBJS = $(SRCS:.cpp=.o)   

# cible par défaut
$(CIBLE): $(OBJS)

clean:
	rm -f  *~  $(CIBLE) $(OBJS)

Trackball.o: Trackball.cpp Trackball.h
Camera.o: Camera.cpp Camera.h Vec3.h Trackball.h
PointRender.o: PointRender.cpp Vec3.h Camera.h Trackball.h PointCloud.h Exception.h



