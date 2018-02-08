#!/usr/bin/env python
'''Script for testing point_cloud library.'''

from time import time
from numpy import array, tile, zeros
import point_cloud

def main():
  '''Entrypoint to the program.'''

  print("Loading cloud.")
  X = point_cloud.LoadPcd("test.pcd")
  point_cloud.Plot(X)

  print("Saving cloud and loading saved cloud.")
  point_cloud.SavePcd("save.pcd", X)
  X = point_cloud.LoadPcd("save.pcd")
  point_cloud.Plot(X)

  print("Voxelizing.")
  V = point_cloud.Voxelize(X, 0.01)
  point_cloud.Plot(V)

  print("Computing normals.")
  startTime = time()
  N = point_cloud.ComputeNormals(X)
  print("Normals computation took {}s.".format(time()-startTime))
  point_cloud.Plot(X, N, 5)

  print("Normals with viewpoints.")
  V = tile(array([[-1,0,1]]), (X.shape[0],1))
  N = point_cloud.ComputeNormals(X, V, kNeighbors=30, rNeighbors=0)
  point_cloud.Plot(X, N, 5)

if __name__ == "__main__":
  main()
  exit()
