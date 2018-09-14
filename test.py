#!/usr/bin/env python
'''Script for testing point_cloud library.'''

from time import time
from numpy import abs, array, eye, max, tile, vstack, zeros
from numpy.linalg import norm
import point_cloud

def main():
  '''Runs each of the functions in this module that begin with the name Test. Prints pass/fail info.'''

  try:
    X = point_cloud.LoadPcd("test.pcd")
    # point_cloud.Plot(X)
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'
  except Exception as e:
    print("Failed to initialize testing.")
    print str(e)
    return

  items = globals()
  for item in items:
    if len(item) > 4 and item[:4] == "Test":

      try:
        result = eval(item + "(X)")
      except Exception as e:
        print str(e)
        result = False

      if result: print("[" + GREEN +"PASS" + END + "] " + item[4:])
      else: print("[" + RED +"FAIL" + END + "] " + item[4:])

def TestComputeNormals(X):
  N = point_cloud.ComputeNormals(X)
  if N.shape[0] != X.shape[0] or N.shape[1] != 3: return False
  if max(abs(norm(N, axis=1)-1.0)) > 0.001: return False
  V = tile(array([[-1,0,1]]), (X.shape[0],1))
  N = point_cloud.ComputeNormals(X, V, kNeighbors=30, rNeighbors=0)
  if N.shape[0] != X.shape[0] or N.shape[1] != 3: return False
  if max(abs(norm(N, axis=1)-1.0)) > 0.001: return False
  return True

def TestIcp(X):
  R = eye(4); R[0, 3] = 0.01
  Y = point_cloud.Transform(R, X)
  T = point_cloud.Icp(Y, X)
  if max(abs(T-R)) > 0.001:
    print T
    print R
    return False
  return True

def TestRemoveStatisticalOutliers(X):
  outlier = array([0, 0, 10])
  Y = vstack((X, outlier))
  Z = point_cloud.RemoveStatisticalOutliers(Y, 50, 0.05)
  if Z.shape[0] != X.shape[0] or Z.shape[1] != 3:
    print X.shape, Z.shape
    return False
  return True

def TestSavePcd(X):
  initCount = X.shape[0]
  point_cloud.SavePcd("save.pcd", X)
  X = point_cloud.LoadPcd("save.pcd")
  return X.shape[0] == initCount and X.shape[1] == 3

def TestSegmentPlane(X):
  point_cloud.SegmentPlane(X, 0.008)
  return True

def TestVoxelize(X):
  A = point_cloud.Voxelize(0.010, X)
  B = point_cloud.Voxelize(0.005, X)
  C = point_cloud.Voxelize(0.002, X)
  if A.shape[0] > B.shape[0] or B.shape[0] > C.shape[0] or C.shape[0] > X.shape[0]:
    print X.shape, A.shape, B.shape, C.shape
    return False
  if A.shape[1] != 3 or B.shape[1] != 3 or C.shape[1] != 3: return False
  return True

def TestVoxelizeWithNormals(X):
  N = point_cloud.ComputeNormals(X)
  XX, NN = point_cloud.Voxelize(0.005, X, N)
  norms = norm(NN, axis=1)
  if XX.shape[0] > X.shape[0] or NN.shape[0] > N.shape[0]:
    print X.shape, XX.shape, N.shape, NN.shape
    return False
  if XX.shape[0] != NN.shape[0] or XX.shape[1] != NN.shape[1]:
    return False
  if XX.shape[1] != 3 or NN.shape[1] != 3: return False
  if (norms > 1.001).any() or (norms < 0.999).any():
    print max(norms), min(norms)
    return False
  return True

if __name__ == "__main__":
  main()
  exit()