#A script for plotting arrow, released as a part of Multiwfn
#Written by Tian Lu (sobereva@sina.com, Beijing Kein Research Center for Natural Sciences), 2024-Feb-4

#This script differs with drawarrow.tcl by meaning of "atmrange"

#atmrange: Selection of atoms, whose geometry center defines arrow beginning position
#fragdx/dy/dz: Cartesian component of the vector to be plotted
#scl: Scale factor of vector length
#rad: Radius of arrow
#showgoc: If printing geometry center in console window

proc drawarrow2 {atmrange fragdx fragdy fragdz {scl 1} {rad 0.2} {showgoc 1}} {
#Determine arrow center
set sel [atomselect top $atmrange]
set cen [measure center $sel]
set cenx [lindex $cen 0]
set ceny [lindex $cen 1]
set cenz [lindex $cen 2]
if {$showgoc==1} {puts "Geometry center: $cenx $ceny $cenz"}
#Scale vector
set fragdx [expr $fragdx*$scl]
set fragdy [expr $fragdy*$scl]
set fragdz [expr $fragdz*$scl]
#Draw arrow
set body 0.75
set begx [expr $cenx]
set begy [expr $ceny]
set begz [expr $cenz]
set endx [expr $cenx+$fragdx*$body]
set endy [expr $ceny+$fragdy*$body]
set endz [expr $cenz+$fragdz*$body]
draw cylinder "$begx $begy $begz" "$endx $endy $endz" radius $rad filled yes resolution 20
set endx2 [expr $cenx+$fragdx]
set endy2 [expr $ceny+$fragdy]
set endz2 [expr $cenz+$fragdz]
draw cone "$endx $endy $endz" "$endx2 $endy2 $endz2" radius [expr $rad*2.5] resolution 20
}

