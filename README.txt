1. Build library:

  mkdir build; cd build
  cmake ..
  make

2. Edit .bashrc:

export PYTHONPATH=/home/[user]/Projects/PointCloudsPython:$PYTHONPATH

3. Test library:

cd /home/[user]/Projects/PointCloudsPython
chmod u+x test.py
./test.py
