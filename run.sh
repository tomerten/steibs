nvcc -ccbin clang-3.8 -lstdc++ -lm  STE.cu STE_ReadInput_double.cpp STE_TFS_double.cu STE_Radiation_double.cu STE_Longitudinal_Hamiltonian_double.cu STE_Synchrotron_double.cu STE_ReadBunchFiles_double.cu STE_IBS_double.cu STE_Bunch_double.cu -run

# nvcc -ccbin clang-3.8 -lstdc++ -lm  STE.cu STE_ReadInput.cpp STE_TFS.cu STE_Radiation.cu STE_Longitudinal_Hamiltonian.cu STE_Synchrotron.cu STE_ReadBunchFiles.cu STE_IBS.cu STE_Bunch.cu -run
nvcc -ccbin clang-3.8 -lstdc++ -lm tester.cu 
time ./a.out
python  plotscript.py
