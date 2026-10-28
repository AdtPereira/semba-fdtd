set term x11 persist           1
plot 'noda2002.fdtd_Wire probe_Wz_10_10_1_s1.dat' using 1:2 every 1::2 with lines
set term x11 persist           2
plot 'noda2002.fdtd_Wire probe001_Wz_10_10_3_s5.dat' using 1:2 every 1::2 with lines
set term x11 persist           3
plot 'noda2002.fdtd_Wire probe002_Wz_10_10_2_s4.dat' using 1:2 every 1::2 with lines
set term x11 persist           4
plot 'noda2002.fdtd_Bulk probe_Jz_9_9_1__10_10_1.dat' using 1:2 every 1::2 with lines
