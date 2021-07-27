#!/bin/bash

nvcc -ccbin clang-3.8 -lstdc++ -lm tester.cu
