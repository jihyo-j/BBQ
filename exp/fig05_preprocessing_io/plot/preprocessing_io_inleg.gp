# preprocessing_io_inleg.gp -- generated in-panel legend; caller sets LX, LT (pt)
set object 120 rect from screen sx(LX),sy(LT+-6.25) to screen sx(LX+8.0),sy(LT+-1.25) fc rgb MBLUE fs solid 1.0 border lc rgb "#000000" lw MLW front
set label 130 "query preprocessing" at screen sx(LX+10.5),sy(LT+-3.75) left font ",6" tc rgb "#000000" front
set object 121 rect from screen sx(LX),sy(LT+-13.75) to screen sx(LX+8.0),sy(LT+-8.75) fc rgb MBLUE fs empty border lc rgb "#000000" lw MLW front
eval HATCHPT(LX, LT-13.75, LX+8.0, LT-8.75)   # same hatch as the bars
set label 131 "exact-FP I/O wait" at screen sx(LX+10.5),sy(LT+-11.25) left font ",6" tc rgb "#000000" front
