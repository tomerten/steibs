mkdir -p build 
mkdir -p cpp/build
mkdir -p cpp/tests/build
cd cpp/build
cmake .. #-DCMAKE_INSTALL_PREFIX=~/.local
make
sudo make install
cd ..
cd tests/build
cmake ..
make
cd ../../../build
cmake ..
make
make install 
cd ..
# poetry install