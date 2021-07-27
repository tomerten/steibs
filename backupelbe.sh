#!/bin/bash

scp -r /home/tmerten/cuda-workspace/\{ *.cu  *.h  *.cpp  *.cuh *.in \}  mertens@repo.acc.bessy.de:/home/mertens
