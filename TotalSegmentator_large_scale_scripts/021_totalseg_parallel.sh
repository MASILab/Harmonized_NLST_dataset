#!/bin/bash

/usr/bin/time -v parallel --slf /hosts.txt \
              --memfree 20G \
              --load 200% \
              --eta \
              --progress \
              --joblog /totalseg_parallel_joblog.txt \
              'bash /020_parallelize_totalseg.sh {1}' :::: /path/to/filepaths/txt