'''A module with utilities for manipulating a point cloud (nx3 numpy array).'''

# IMPORTS ==========================================================================================

# python
import ctypes
from copy import copy
from ctypes import c_char, c_int, c_float, c_uint8, pointer, POINTER
# scipy
from numpy.linalg import norm
from scipy.io import loadmat, savemat
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.ctypeslib import ndpointer
from numpy import array, ascontiguousarray, dot, empty, eye, frombuffer, integer, isinf, isnan, issubdtype, \
  logical_and, logical_not, logical_or, ones, repeat, reshape, sum, vstack, zeros

# C BINDINGS =======================================================================================

PointCloudsPython = ctypes.cdll.LoadLibrary(__file__[:-15] + "/build/libPointCloudsPython.so")

PclComputeNormals = PointCloudsPython.PclComputeNormals
PclComputeNormals.restype = c_int
PclComputeNormals.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_float, POINTER(POINTER(c_float))]

CopyAndFree = PointCloudsPython.CopyAndFree
CopyAndFree.restype = c_int
CopyAndFree.argtypes = [POINTER(c_float), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]

CopyAndFreeColors = PointCloudsPython.CopyAndFreeColors
CopyAndFreeColors.restype = c_int
CopyAndFreeColors.argtypes = [POINTER(c_uint8), ndpointer(c_uint8, flags="C_CONTIGUOUS"), c_int]

CopyAndFreeInt = PointCloudsPython.CopyAndFreeInt
CopyAndFreeInt.restype = c_int
CopyAndFreeInt.argtypes = [POINTER(c_int), ndpointer(c_int, flags="C_CONTIGUOUS"), c_int]

PclExtractEuclideanClusters = PointCloudsPython.PclExtractEuclideanClusters
PclExtractEuclideanClusters.restype = c_int
PclExtractEuclideanClusters.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_float, c_int, c_int, ndpointer(c_int, flags="C_CONTIGUOUS")]

PclIcp = PointCloudsPython.PclIcp
PclIcp.restype = c_int
PclIcp.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, ndpointer(c_float, flags="C_CONTIGUOUS")]

PclLoadPcd = PointCloudsPython.PclLoadPcd
PclLoadPcd.restype = c_int
PclLoadPcd.argtypes = [POINTER(c_char), POINTER(POINTER(c_float)), POINTER(c_int)]

PclPointCloud2MsgToXyzRgb = PointCloudsPython.PclPointCloud2MsgToXyzRgb
PclPointCloud2MsgToXyzRgb.restype = c_int
PclPointCloud2MsgToXyzRgb.argtypes = [ndpointer(c_uint8, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_uint8, flags="C_CONTIGUOUS")]

PclPointCloud2MsgToXyz = PointCloudsPython.PclPointCloud2MsgToXyz
PclPointCloud2MsgToXyz.restype = c_int
PclPointCloud2MsgToXyz.argtypes = [ndpointer(c_uint8, flags="C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_int, c_int, c_int, ndpointer(c_float, flags="C_CONTIGUOUS")]

PclSavePcd = PointCloudsPython.PclSavePcd
PclSavePcd.restype = c_int
PclSavePcd.argtypes = [POINTER(c_char), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]

PclSaveOrganizedPcd = PointCloudsPython.PclSaveOrganizedPcd
PclSaveOrganizedPcd.restype = c_int
PclSaveOrganizedPcd.argtypes = [POINTER(c_char), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_int]

PclRemoveStatisticalOutliers = PointCloudsPython.PclRemoveStatisticalOutliers
PclRemoveStatisticalOutliers.restype = c_int
PclRemoveStatisticalOutliers.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_int, c_float, POINTER(POINTER(c_float)), POINTER(c_int)]

PclSegmentPlane = PointCloudsPython.PclSegmentPlane
PclSegmentPlane.restype = c_int
PclSegmentPlane.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_float, POINTER(POINTER(c_int)), POINTER(c_int)]

PclVoxelize = PointCloudsPython.PclVoxelize
PclVoxelize.restype = c_int
PclVoxelize.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_float, POINTER(POINTER(c_float)), POINTER(c_int)]

PclVoxelizeWithColors = PointCloudsPython.PclVoxelizeWithColors
PclVoxelizeWithColors.restype = c_int
PclVoxelizeWithColors.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_uint8, flags="C_CONTIGUOUS"), c_int, c_float, POINTER(POINTER(c_float)), POINTER(POINTER(c_uint8)), POINTER(c_int)]

PclVoxelizeWithColorsAndNormals = PointCloudsPython.PclVoxelizeWithColorsAndNormals
PclVoxelizeWithColorsAndNormals.restype = c_int
PclVoxelizeWithColorsAndNormals.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_uint8, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_float, POINTER(POINTER(c_float)), POINTER(POINTER(c_uint8)), POINTER(POINTER(c_float)), POINTER(c_int)]

PclVoxelizeWithNormals = PointCloudsPython.PclVoxelizeWithNormals
PclVoxelizeWithNormals.restype = c_int
PclVoxelizeWithNormals.argtypes = [ndpointer(c_float, flags="C_CONTIGUOUS"), ndpointer(c_float, flags="C_CONTIGUOUS"), c_int, c_float, POINTER(POINTER(c_float)), POINTER(POINTER(c_float)), POINTER(c_int)]

# FUNCTIONS ========================================================================================

def ComputeNormals(cloud, viewPoints=None, kNeighbors=0, rNeighbors=0.03):
  '''Calls PCL to compute surface normals for the input cloud.
  
  - Input cloud: nx3 point cloud to compute normals for.
  - Input viewPoints: nx3 list of view points from which each cloud point was observed.
  - Input kNeighbors: Number of neighbors to consider in normals calculation. Set to negative if
    using rNeighbors instead.
  - Input rNeighbors: Radius of area to consider in normals calculation. Set to negative if using
    kNeighbors instead.
  - Returns normalsOut: nx3, c-contiguous, float32 numpy array.
  '''

  cloud = ascontiguousarray(cloud, dtype='float32')
  ppnormals = pointer(pointer(c_float(0)))

  errorCode = PclComputeNormals(cloud, cloud.shape[0], kNeighbors, rNeighbors, ppnormals)

  if errorCode == -1:
    raise Exception(f"Invalid argument to Compute Normals: kNeighbors={kNeighbors}, rNeighbors={rNeighbors}.")
  elif errorCode == -2:
    raise Exception("Size of normals output from PCL did not match size of input cloud.")

  pnormals = ppnormals.contents
  normals = empty((cloud.shape[0], 3), dtype='float32', order='C')
  CopyAndFree(pnormals, normals, cloud.shape[0])

  # sometimes NaNs appear in the normals (perhaps when no points in neighborhood?)
  mask = logical_or(isnan(normals).any(axis=1), isinf(normals).any(axis=1))
  normals[mask] = array([1,0,0], dtype='float32')

  # flip if viewpoints are provided
  if viewPoints is not None:
    pc = viewPoints - cloud
    pcMag = reshape(norm(pc, axis=1), (pc.shape[0], 1))
    pc /= pcMag
    theta = sum(pc*normals, axis=1)
    flip = theta < 0
    if isnan(theta).any():
      SavePcd("cloud_error.pcd", cloud)
      SavePcd("normals_error.pcd", normals)
      raise Exception("Theta contains invalid values. Saved cloud and normals.")
    normals[flip,:] = -normals[flip,:]

  return normals
  
def ExtractEuclideanClusters(cloud, searchRadius, minClusterSize = 0, maxClusterSize = None):
  '''Clusters the point cloud using PCL's Euclidean clustering method.
  
  - Input cloud: nx3 numpy array.
  - Input searchRadius: Maximum distance between neighboring points in the same cluster.
  - Input minClusterSize: Minimum number of points in a single cluster.
  - Input maxClusterSize: Maximum number of points in a single cluster.
  - Returns clouds: A list of clouds with points from cloud, 1 for each cluster found. The number of
    clusters is len(clouds).
  - Returns clusterId: An n-element array with indices indicating to which cluster each point in
    cloud belongs. An index of 0 indicates that the point does not belong to any cluster. The number
    of clusters is max(clusterId).
  '''
  
  if maxClusterSize is None:
    maxClusterSize = cloud.shape[0]
  
  cloud = ascontiguousarray(cloud, dtype='float32')
  clusterId = zeros(cloud.shape[0], dtype='int32')

  errorCode = PclExtractEuclideanClusters(cloud, cloud.shape[0], searchRadius, minClusterSize,
    maxClusterSize, clusterId)
  
  if errorCode == -1:
    raise Exception(f"Invalid searchRadius {searchRadius}.")
  if errorCode == -2:
    raise Exception(f"Invalid minClusterSize: {minClusterSize}.")
  if errorCode == -3:
    raise Exception(f"Invalid maxClusterSize: {maxClusterSize}.")
  
  clouds = []
  nClusters = max(clusterId)
  for i in range(1, nClusters + 1):
    clouds.append(cloud[clusterId == i, :])
    
  return clouds, clusterId

def FilterNans(cloud):
  '''Removes points that are (NaN, NaN, NaN).
  
  - Input cloud: nx3 numpy array.
  - Returns cloud: nx3 numpy array without points that are all NaNs.
  '''

  mask = logical_not(isnan(cloud).any(axis=1))
  cloud = cloud[mask]
  return cloud

def FilterNearAndFarPoints(axis, minDist, maxDist, cloud, normals=None):
    '''Filters points outside min and max distances for a given coordinate.
    
    - Input axis: Coordinate to check min and max distance for filtering.
    - Input minDist: Points less than this along the given axis are filtered.
    - Input maxDist: Points grater than this along the given axis are filtered.
    - Input cloud: nx3 numpy array.
    - Input normals: (optional) nx3 numpy array.
    - Returns cloud: mx3 numpy array.
    - Returns normals: (optional) mx3 numpy array.
    '''

    mask = logical_and(cloud[:, axis] >= minDist, cloud[:, axis] <= maxDist)
    cloud = cloud[mask, :]
    if normals is None: return cloud

    normals = normals[mask, :]
    return cloud, normals

def FilterWorkspace(workspace, cloud, normals=None):
  '''Removes points that are outside of a workspace defined in terms of standard basis axes.
  
  - Input workspace: List of tuples of the form [(minX, maxX), (minY, maxY), (minZ, maxZ)]
  - Input cloud: nx3 numpy array.
  - Input normals: (optional) nx3 numpy array.
  - Returns cloud: mx3 numpy array.
  - Returns normals: (optional) mx3 numpy array.
  '''

  mask = (((((cloud[:,0] >= workspace[0][0]) &  (cloud[:,0] <= workspace[0][1])) \
           & (cloud[:,1] >= workspace[1][0])) & (cloud[:,1] <= workspace[1][1])) \
           & (cloud[:,2] >= workspace[2][0])) & (cloud[:,2] <= workspace[2][1])
  cloud = cloud[mask, :]
  if normals is None: return cloud

  normals = normals[mask, :]
  return cloud, normals

def Icp(cloud1, cloud2):
  '''Runs iterative closest point to align cloud2 to cloud1.
  
  - Input cloud1: nx3 numpy array, the target cloud.
  - Input cloud2: nx3 numpy array, the source cloud.
  - Returns T: 4x4 homogenous transform that should be applied to cloud2 to make it similar to cloud1.
  '''

  cloud1 = ascontiguousarray(cloud1, dtype='float32')
  cloud2 = ascontiguousarray(cloud2, dtype='float32')
  T = zeros(16, dtype='float32')

  PclIcp(cloud1, cloud1.shape[0], cloud2, cloud2.shape[0], T)
  T = reshape(T, (4,4)).T
  return T
  
def InverseTransform(T):
    '''Quick inverse of homogeneous transform. Faster than linalg.inv.
    
    - Input T: 4x4 matrix, assumed to be in SE(3) (i.e. det(T[0:3, 0:3]) = 1 and
      T[3, 0:3] = [0, 0, 0, 1]). Assumption is not checked (for speed), and if the assumption does
      not hold, the result is not guaranteed to be the matrix inverse.
    - Returns Tinv: T^{-1}.
    '''
    
    R = T[0:3, 0:3].T
    Tinv = eye(4)
    Tinv[0:3, 0:3] = R
    Tinv[0:3, 3] = -dot(R, T[0:3, 3])
    return Tinv

def LoadMat(fileName):
  '''Loads a point cloud from a Matlab .mat file.
  
  - Input fileName: Name of the Matlab file to load.
  - Returns cloud: nx3, c-contigous, float32 numpy array.
  - Returns normals: nx3, c-contiguous, normals array, or None if normals not present.
  '''

  data = loadmat(fileName)
  cloud = data["cloud"]
  normals = data["normals"] if "normals" in data else None
  return cloud, normals

def LoadPcd(fileName):
  '''Calls PCL to load the PCD file.
  
  - Input fileName: Full file name to load.
  - Returns cloud: nx3, c-contiguous, float32 numpy array.
  '''

  fileName = fileName.encode('utf-8')
  nPoints = pointer(c_int(0))
  ppoints = pointer(pointer(c_float(0)))

  errorCode = PclLoadPcd(fileName, ppoints, nPoints)
  points = ppoints.contents
  nPoints = nPoints.contents.value

  if errorCode < 0:
    raise Exception("Loading file {} failed.".format(fileName))

  cloud = empty((nPoints, 3), dtype='float32', order='C')
  errorCode = CopyAndFree(points, cloud, nPoints)

  return cloud

def Plot(cloud, normals=None, nthNormal=0):
  '''Uses matplotlib to plot the points in 3D.
  
  - Input cloud: nx3 numpy array of points.
  - Input normals: (Optional) nx3 numpy array of normals.
  - Input nthNormal: (Optional) Only plot every nthNormal normals.
  - Returns None.
  '''

  fig = pyplot.figure()
  ax = fig.add_subplot(111, projection="3d")

  # points
  x = []; y = []; z = []
  for point in cloud:
    x.append(point[0])
    y.append(point[1])
    z.append(point[2])

  ax.scatter(x, y, z, c='k', s=5, depthshade=False)
  extents = UpdatePlotExtents(x,y,z)

  # normals
  if normals is not None and nthNormal > 0:
    xx=[0,0]; yy=[0,0]; zz=[0,0]
    for i in range(len(cloud)):
      if i % nthNormal != 0: continue
      xx[0] = x[i]; xx[1] = x[i] + 0.02 * normals[i][0]
      yy[0] = y[i]; yy[1] = y[i] + 0.02 * normals[i][1]
      zz[0] = z[i]; zz[1] = z[i] + 0.02 * normals[i][2]
      ax.plot(xx, yy, tuple(zz), 'g')

  # bounding cube
  l = (extents[1]-extents[0], extents[3]-extents[2], extents[5]-extents[4])
  c = (extents[0]+l[0]/2.0, extents[2]+l[1]/2.0, extents[4]+l[2]/2.0)
  d = 1.10*max(l) / 2.0

  ax.plot((c[0]+d, c[0]+d, c[0]+d, c[0]+d, c[0]-d, c[0]-d, c[0]-d, c[0]-d), \
          (c[1]+d, c[1]+d, c[1]-d, c[1]-d, c[1]+d, c[1]+d, c[1]-d, c[1]-d), \
          (c[2]+d, c[2]-d, c[2]+d, c[2]-d, c[2]+d, c[2]-d, c[2]+d, c[2]-d), \
           c='k', linewidth=0)

  # labels
  ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
  ax.set_title(f"Point cloud with {len(cloud)} points.")

  pyplot.show(block=True)

def PointCloud2MsgToArray(msg):
  '''Converts a PointCloud2 ROS message to numpy arrays.
  
  - Input: PointCloud2 ROS message.
  - Returns cloud: nx3 numpy array of points: [..., (x_i, y_i, z_i), ...].
  - Returns rgb: (Optional) If the message has a color field, returns colors:
    [..., (r_i, g_i, b_i), ...]. Each r, g, b component is a byte with values in the range [0, 255].
  '''

  # TODO: Assumes native endianness.
  # TODO: Data type assumed to be float.

  n = msg.width * msg.height
  cloud = empty((n, 3), dtype='float32', order='C')
  rgb = empty((n, 3), dtype='uint8', order='C')

  fieldToOffset = {msg.fields[i].name:msg.fields[i].offset for i in range(len(msg.fields))}
  has_color = "rgb" in fieldToOffset.keys()
  data = frombuffer(msg.data, dtype="uint8")

  if has_color:
    errorCode = PclPointCloud2MsgToXyzRgb(data, msg.height, msg.width, msg.row_step, msg.point_step,
      fieldToOffset["x"], fieldToOffset["y"], fieldToOffset["z"], fieldToOffset["rgb"], cloud, rgb)
  else:
    errorCode = PclPointCloud2MsgToXyz(data, msg.height, msg.width, msg.row_step, msg.point_step,
      fieldToOffset["x"],   fieldToOffset["y"], fieldToOffset["z"], cloud)

  if errorCode < 0:
    raise Exception(f"Error {errorCode} when calling PclRemoveStatisticalOutliers.")

  if has_color:
    return cloud, rgb
  return cloud

def RemoveStatisticalOutliers(cloud, meanK, stddevMulThresh):
  '''Calls PCL to remove statistical outliers from the cloud.
  
  - Input cloud: nx3 point cloud from which to remove outliers.
  - Input meanK: Scalar number of neighbors to analyze.
  - Input stddevMulThresh: Scalar standard deviation multiplier.
  - Returns cloud: nx3, c-contiguous, float32 numpy array.
  '''
  
  # input checking
  if cloud.shape[0] == 0:
    return copy(cloud)

  # call C++ wrapper
  cloud = ascontiguousarray(cloud, dtype='float32')
  
  nPoints = pointer(c_int(0))
  ppoints = pointer(pointer(c_float(0)))

  errorCode = PclRemoveStatisticalOutliers(
    cloud, cloud.shape[0], meanK, stddevMulThresh, ppoints, nPoints)

  # unpack output    
  points = ppoints.contents
  nPoints = nPoints.contents.value
  cloud = empty((nPoints, 3), dtype='float32', order='C')
  CopyAndFree(points, cloud, nPoints)
  
  # check for errors
  if errorCode < 0:
    raise Exception(f"Error {errorCode} when calling PclRemoveStatisticalOutliers.")

  return cloud

def SaveMat(fileName, cloud, normals=None):
  '''Saves cloud to Matlab .mat file.
  
  - Input fileName: Name of the file to save (including extension).
  - Input cloud: nx3 numpy array to save to the mat file.
  - Input normals: Optionally nx3 array of surface normals to save.
  - Returns None.
  '''

  data = {'cloud':cloud}
  if normals is not None: data["normals"] = normals
  savemat(fileName, data)

def SavePcd(fileName, cloud):
  '''Saves cloud to (ASCII) PCD file.'''

  fileName = fileName.encode('utf-8')
  cloud = ascontiguousarray(cloud, dtype='float32')
  errorCode = PclSavePcd(fileName, cloud, cloud.shape[0])

  if errorCode < 0:
    raise Exception(f"Failed to save {fileName}.")

def SaveOrganizedPcd(fileName, cloud, height, width):
  '''Reorganizes cloud and saves it to (ASCII) PCD file.'''

  cloud = ascontiguousarray(cloud, dtype='float32')
  errorCode = PclSaveOrganizedPcd(fileName, cloud, cloud.shape[0], height, width)

  if errorCode < 0:
    raise Exception(f"Failed to save {fileName}.")

def SegmentPlane(cloud, distanceThreshold):
  '''Calls PCL to segment the largest plane from the cloud.
  
  - Input cloud: nx3 point cloud from which to segment the plane.
  - Input distanceThreshold: Scalar distance threshold from plane.
  - Returns indicesOut: nx1, c-contiguous, int32 numpy array.
  '''

  cloud = ascontiguousarray(cloud, dtype='float32')

  nIndices = pointer(c_int(0))
  pindices = pointer(pointer(c_int(0)))

  PclSegmentPlane(cloud, cloud.shape[0], distanceThreshold, pindices, nIndices)
  indices = pindices.contents
  nIndices = nIndices.contents.value

  indicesOut = empty((nIndices,1), dtype='int32', order='C')
  CopyAndFreeInt(indices, indicesOut, nIndices)

  return indicesOut

def Transform(T, cloud, normals=None):
  '''Applies homogeneous transform T to the cloud: y = Tx, for each x in cloud.
  
  - Input T: 4x4 matrix, consisting of a 3x3 rotation matrix and 3x1 translation vector.
  - Input cloud: nx3 points to apply transform to.
  - Input normals: (optional) nx3 normalized vectors which will only be rotated.
  '''

  X = vstack((cloud.T, ones(cloud.shape[0])))
  X = dot(T, X).T
  X = X[:, 0:3]

  if normals is None:
    return X

  T = T[0:3, 0:3]
  N = dot(T, normals.T).T
  return X, N

def UpdatePlotExtents(x, y, z, extents=None):
  '''Extends the current extents in a plot by the given values.
  
  - Input x: List of x-coordiantes.
  - Input y: List of y-coordinates.
  - Input z: Lizt of z-coordinates.
  - Input extents: Extents of all other points in the plot as (minX, maxX, minY, maxY, minZ, maxZ).
  - Returns newExtents: The max/min of existing extents with the input coordinates.
  '''

  x = copy(x); y = copy(y); z = copy(z)

  if type(x) == type(array([])):
    x = x.flatten().tolist()
    y = y.flatten().tolist()
    z = z.flatten().tolist()

  if extents != None:
    x.append(extents[0]); x.append(extents[1])
    y.append(extents[2]); y.append(extents[3])
    z.append(extents[4]); z.append(extents[5])

  extents = (min(x),max(x), min(y),max(y), min(z),max(z))

  return extents

def Voxelize(voxelSize, cloud, normals=None, colors=None):
  '''Calls PCL to load the voxelize the cloud.
  
  - Input voxelSize: Scalar size of the voxels to use.
  - Input cloud: nx3 point cloud to voxelize.
  - Input normals: (Optional) nx3 array of surface normals.
  - Input colors: (Optional) nx3 array of colors. If the type is a floating point type, assumes
    values are in [0, 1]. If the type an integer type, assumes values are in [0, 255].
  - Returns cloud: mx3, c-contiguous, float32 numpy array.
  - Returns normals: (Optional) mx3 normalized vectors corresponding to points in cloud.
  - Returns colors: (Optional) mx3 array of colors. If the input type is a floating point type, the
    output type is also floating point in [0, 1] (but input values could be changed by as much as
    1/255, due to roundoff errors when converting to a byte). If the input type is an integer type,
    the output is also integer in [0, 255] (and values are preserved).
  '''

  # Input checking and type conversions.

  if cloud.shape[1] != 3:
    raise Exception(f"Expected 3 columns in cloud, got {cloud.shape[1]}.")

  cloud = ascontiguousarray(cloud, dtype='float32')
  nPoints = pointer(c_int(0))
  ppoints = pointer(pointer(c_float(0)))

  if normals is not None:

    if normals.shape[0] != cloud.shape[0]:
      raise Exception(f"Cloud has {cloud.shape[0]} points and normals has {normals.shape[0]} points!")
    if normals.shape[1] != 3:
      raise Exception(f"Expected 3 columns in normals, got {normals.shape[1]}.")

    normals = ascontiguousarray(normals, dtype='float32')
    pnormals = pointer(pointer(c_float(0)))

  if colors is not None:

    if colors.shape[0] != cloud.shape[0]:
      raise Exception(f"Cloud has {cloud.shape[0]} points and colors has {colors.shape[0]} points!")
    if colors.shape[1] != 3:
      raise Exception(f"Expected 3 columns in colors, got {colors.shape[1]}.")

    if issubdtype(colors.dtype, integer):
      colors = ascontiguousarray(colors, dtype='uint8')
      convert_colors_to_float = False
    else:
      colors = ascontiguousarray(255 * colors, dtype='uint8')
      convert_colors_to_float = True
    
    pcolors = pointer(pointer(c_uint8(0)))

  # Call the appropriate function.

  if normals is None and colors is None:
    errorCode = PclVoxelize(
      cloud, cloud.shape[0], voxelSize, ppoints, nPoints)
  elif colors is None:
    errorCode = PclVoxelizeWithNormals(
      cloud, normals, cloud.shape[0], voxelSize, ppoints, pnormals, nPoints)
  elif normals is None:
    errorCode = PclVoxelizeWithColors(
      cloud, colors, cloud.shape[0], voxelSize, ppoints, pcolors, nPoints)
  else:
    errorCode = PclVoxelizeWithColorsAndNormals(
      cloud, colors, normals, cloud.shape[0], voxelSize, ppoints, pcolors, pnormals, nPoints)

  if errorCode < 0:
    raise Exception(f"Voxelization failed with code {errorCode}.")

  # Copy out the result and free memory.

  points = ppoints.contents
  nPoints = nPoints.contents.value
  cloud = empty((nPoints, 3), dtype='float32', order='C')
  CopyAndFree(points, cloud, nPoints)
  output = [cloud]

  if normals is not None:
    norms = pnormals.contents
    normals = empty((nPoints, 3), dtype='float32', order='C')
    CopyAndFree(norms, normals, nPoints)
    # correct for errors introduced by PCL averaging normals
    magnitudes = repeat(reshape(norm(normals, axis=1), (nPoints, 1)), 3, axis=1)
    normals = normals / magnitudes
    output.append(normals)

  if colors is not None:
    cols = pcolors.contents
    colors = empty((nPoints, 3), dtype='uint8', order='C')
    CopyAndFreeColors(cols, colors, nPoints)
    if convert_colors_to_float:
      colors = colors.astype('float') / 255.0
    output.append(colors)

  # Return result.

  if len(output) == 1:
    return output[0]
  return tuple(output)
  
def WorkspaceCenter(workspace):
  '''Returns the center of a rectangular workspace.
  
  - Input workspace: List of pairs or 2D array of [(minX, maxX), (minY, maxY), (minZ, maxZ)].
  - Returns center: Numpy array of length 3.
  '''
  
  return array([
    workspace[0][0] + (workspace[0][1] - workspace[0][0]) / 2.0,
    workspace[1][0] + (workspace[1][1] - workspace[1][0]) / 2.0,
    workspace[2][0] + (workspace[2][1] - workspace[2][0]) / 2.0])
