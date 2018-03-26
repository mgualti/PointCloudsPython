#!/usr/bin/env python
'''Script for testing point_cloud library.'''

from time import time
from numpy import array, tile, zeros
import point_cloud

def main():
  '''Entrypoint to the program.'''
  
  X = point_cloud.LoadPcd("test.pcd")  
  
  items = globals()
  for item in items:
    if len(item) > 4 and item[:4] == "Test":
      
      try:
        result = eval(item + "(X)")
      except:
        result = False
      
      if result: print(item + " PASSED")
      else: print(item + " FAILED")

def TestComputeNormals(X):
  point_cloud.ComputeNormals(X)
  V = tile(array([[-1,0,1]]), (X.shape[0],1))
  point_cloud.ComputeNormals(X, V, kNeighbors=30, rNeighbors=0)
  return True

def TestIcp(X):
  T = point_cloud.Icp(X, X)
  return T[3,0] == 0 and T[3,1] == 0 and T[3,2] == 0 and T[3,3] == 1
  
def TestRemoveStatisticalOutliers(X):
   point_cloud.RemoveStatisticalOutliers(X, 50, 1.0)
   return True

def TestSavePcd(X):
  point_cloud.SavePcd("save.pcd", X)
  X = point_cloud.LoadPcd("save.pcd")
  #point_cloud.Plot(X)
  return X.shape[0] == 2977 and X.shape[1] == 3

def TestSegmentPlane(X):
  point_cloud.SegmentPlane(X, 0.008)
  return True

def TestVoxelize(X):
  point_cloud.Voxelize(X, 0.01)
  return True

if __name__ == "__main__":
  main()
  exit()