mex -DDEFINEUNIX -largeArrayDims  -lopenblas mtimesx.c
%mex CFLAGS="\$CFLAGS -std=c99" -DDEFINEUNIX -largeArrayDims -lopenblas mtimesx.c
