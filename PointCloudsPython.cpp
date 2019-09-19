#include <cstring>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
using namespace pcl;
using namespace pcl::io;

// HELPERS =========================================================================================

int PclArrayToPointCloud(float* points, int nPoints, PointCloud<PointXYZ>& cloud)
{
  for (int i = 0; i < nPoints; i++)
    cloud.push_back(PointXYZ(points[3*i+0], points[3*i+1], points[3*i+2]));

  return 0;
}

int PclArrayToPointCloudPtr(float* points, int nPoints, PointCloud<PointXYZ>::Ptr& cloud)
{
  for (int i = 0; i < nPoints; i++)
    cloud->push_back(PointXYZ(points[3*i+0], points[3*i+1], points[3*i+2]));

  return 0;
}

int PclArraysToPointCloudNormalPtr(float* points, float* normals, int nPoints, PointCloud<PointNormal>::Ptr& cloud)
{
  PointNormal pn;
  for (int i = 0; i < nPoints; i++)
  {
    pn.x = points[3*i+0];
    pn.y = points[3*i+1];
    pn.z = points[3*i+2];
    pn.normal_x = normals[3*i+0];
    pn.normal_y = normals[3*i+1];
    pn.normal_z = normals[3*i+2];
    cloud->push_back(pn);
  }

  return 0;
}

int PclNormalsToNewArray(PointCloud<Normal>& cloud, float** pnormals)
{
  int nNormals = cloud.size();
  float* normals = new float[nNormals * 3];
  *pnormals = normals;

  for (int i = 0; i < nNormals; i++)
  {
    normals[3*i+0] = cloud[i].normal_x;
    normals[3*i+1] = cloud[i].normal_y;
    normals[3*i+2] = cloud[i].normal_z;
  }

  return 0;
}

int PclPointCloudToNewArray(PointCloud<PointXYZ>& cloud, float** ppoints, int* nPoints)
{
  *nPoints = cloud.size();
  float* points = new float[*nPoints * 3];
  *ppoints = points;

  for (int i = 0; i < *nPoints; i++)
  {
    points[3*i+0] = cloud[i].x;
    points[3*i+1] = cloud[i].y;
    points[3*i+2] = cloud[i].z;
  }

  return 0;
}

int PclPointCloudNormalToNewArrays(PointCloud<PointNormal>& cloud, float** ppoints, float** pnormals, int* nPoints)
{
  *nPoints = cloud.size();
  float* points = new float[*nPoints * 3];
  float* normals = new float[*nPoints * 3];
  *ppoints = points;
  *pnormals = normals;

  for (int i = 0; i < *nPoints; i++)
  {
    points[3*i+0] = cloud[i].x;
    points[3*i+1] = cloud[i].y;
    points[3*i+2] = cloud[i].z;
    normals[3*i+0] = cloud[i].normal_x;
    normals[3*i+1] = cloud[i].normal_y;
    normals[3*i+2] = cloud[i].normal_z;
  }

  return 0;
}

int PclPointIndicesToNewArray(PointIndices::Ptr& pointIndicesPtr, int** pindices, int* nIndices)
{
  *nIndices = pointIndicesPtr->indices.size();
  int* indices = new int[*nIndices];
  *pindices = indices;

  for (int i = 0; i < *nIndices; i++)
  {
    indices[i] = pointIndicesPtr->indices[i];
  }

  return 0;
}

// EXTERN ==========================================================================================

extern "C" int CopyAndFree(float* in, float* out, int nPoints)
{
  memcpy(out, in, sizeof(float)*nPoints*3);
  delete[] in;
  return 0;
}

extern "C" int CopyAndFreeInt(int* in, int* out, int nIndices)
{
  memcpy(out, in, sizeof(int)*nIndices);
  delete[] in;
  return 0;
}

extern "C" int PclComputeNormals(float* pointsIn, int nPointsIn, int kNeighborhood,
  float radiusNeighborhood, float** normalsOut)
{
  PointCloud<PointXYZ>::Ptr cloudIn(new PointCloud<PointXYZ>);
  PclArrayToPointCloudPtr(pointsIn, nPointsIn, cloudIn);

  PointCloud<Normal> normals;
  search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ> ());

  //NormalEstimation<PointXYZ, Normal> ne;
  NormalEstimationOMP<PointXYZ, Normal> ne(4);
  ne.setInputCloud(cloudIn);
  ne.setSearchMethod(tree);

  if ((kNeighborhood <= 0) && (radiusNeighborhood > 0))
    ne.setRadiusSearch(radiusNeighborhood);
  else if ((kNeighborhood > 0) && (radiusNeighborhood <= 0))
    ne.setKSearch(kNeighborhood);
  else
    return -1;

  ne.compute(normals);
  if (normals.size() != cloudIn->size())
    return -2;

  PclNormalsToNewArray(normals, normalsOut);
  return 0;
}

extern "C" int PclExtractEuclideanClusters(float* points, int nPoints, float searchRadius,
  int minClusterSize, int maxClusterSize, int* clusterIndices)
{
  if (searchRadius < 0)
    return -1;
  if (minClusterSize < 0)
    return -2;
  if (maxClusterSize > nPoints)
    return -3;
  
  PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
  PclArrayToPointCloudPtr(points, nPoints, cloud);
  
  search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>);
  tree->setInputCloud(cloud);

  vector<PointIndices> idxs;
  EuclideanClusterExtraction<PointXYZ> ec;
  ec.setClusterTolerance(searchRadius);
  ec.setMinClusterSize(minClusterSize);
  ec.setMaxClusterSize(maxClusterSize);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(idxs);
  
  for (int i = 0; i < idxs.size(); i++)
  {
    for (int j = 0; j < idxs[i].indices.size(); j++)
      clusterIndices[idxs[i].indices[j]] = i + 1;
  }
  
  return 0;
}

extern "C" int PclIcp(float* points1, int nPoints1, float* points2, int nPoints2, float* T)
{
  PointCloud<PointXYZ>::Ptr cloud1(new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr cloud2(new PointCloud<PointXYZ>);

  PclArrayToPointCloudPtr(points1, nPoints1, cloud1);
  PclArrayToPointCloudPtr(points2, nPoints2, cloud2);
  
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cloud2);
  icp.setInputTarget(cloud1);
  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);

  Eigen::Matrix4f M = icp.getFinalTransformation();
  
  for (int i = 0; i < 16; i++)
	T[i] = M(i);
  
  return 0;
}

extern "C" int PclLoadPcd(char* fileName, float** ppoints, int* nPoints)
{
  PointCloud<PointXYZ> cloud;
  if (loadPCDFile<PointXYZ>(fileName, cloud) < 0)
    return -1;

  PclPointCloudToNewArray(cloud, ppoints, nPoints);
  return 0;
}

extern "C" int PclSavePcd(char* fileName, float* points, int nPoints)
{
  PointCloud<PointXYZ> cloud;
  PclArrayToPointCloud(points, nPoints, cloud);

  if (savePCDFileASCII(fileName, cloud) < 0)
    return -1;

  return 0;
}

extern "C" int PclSaveOrganizedPcd(char* fileName, float* points, int nPoints, int height, int width)
{
  PointCloud<PointXYZ> cloud;
  PclArrayToPointCloud(points, nPoints, cloud);

  PointCloud<PointXYZ> cloud_organized;
  cloud_organized.height = height;
  cloud_organized.width = width;
  cloud_organized.points.resize(height*width);

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      cloud_organized.at(j,i) = cloud.points[i*width + j];
    }
  }

  if (savePCDFileASCII(fileName, cloud_organized) < 0)
    return -1;

  return 0;
}

extern "C" int PclRemoveStatisticalOutliers(float* pointsIn, int nPointsIn, int meanK, float stddevMulThresh, float** pointsOut, int* nPointsOut)
{
  PointCloud<PointXYZ>::Ptr cloudIn(new PointCloud<PointXYZ>);
  PointCloud<PointXYZ> cloudOut;

  PclArrayToPointCloudPtr(pointsIn, nPointsIn, cloudIn);

  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloudIn);
  sor.setMeanK(meanK);
  sor.setStddevMulThresh(stddevMulThresh);
  sor.filter(cloudOut);

  PclPointCloudToNewArray(cloudOut, pointsOut, nPointsOut);
  return 0;
}

extern "C" int PclSegmentPlane(float* pointsIn, int nPointsIn, float distanceThreshold, int** indicesOut, int* nIndicesOut)
{
  PointCloud<PointXYZ>::Ptr cloudIn(new PointCloud<PointXYZ>);
  PointCloud<PointXYZ> cloudOut;

  PclArrayToPointCloudPtr(pointsIn, nPointsIn, cloudIn);

  ModelCoefficients::Ptr coefficients (new ModelCoefficients);
  PointIndices::Ptr inliers (new PointIndices);
  SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (SACMODEL_PLANE);
  seg.setMethodType (SAC_RANSAC);
  seg.setDistanceThreshold (distanceThreshold);
  seg.setInputCloud (cloudIn);
  seg.segment (*inliers, *coefficients);

  PclPointIndicesToNewArray(inliers, indicesOut, nIndicesOut);

  return 0;
}

extern "C" int PclVoxelize(float* pointsIn, int nPointsIn, float voxelSize, float** pointsOut, int* nPointsOut)
{
  PointCloud<PointXYZ>::Ptr cloudIn(new PointCloud<PointXYZ>);
  PointCloud<PointXYZ> cloudOut;

  PclArrayToPointCloudPtr(pointsIn, nPointsIn, cloudIn);

  VoxelGrid<PointXYZ> grid;
  grid.setInputCloud(cloudIn);
  grid.setLeafSize(voxelSize, voxelSize, voxelSize);
  grid.filter(cloudOut);

  PclPointCloudToNewArray(cloudOut, pointsOut, nPointsOut);
  return 0;
}

extern "C" int PclVoxelizeWithNormals(float* pointsIn, float* normalsIn, int nPointsIn, float voxelSize, float** pointsOut, float** normalsOut, int* nPointsOut)
{
  PointCloud<PointNormal>::Ptr cloudIn(new PointCloud<PointNormal>);
  PointCloud<PointNormal> cloudOut;

  PclArraysToPointCloudNormalPtr(pointsIn, normalsIn, nPointsIn, cloudIn);

  VoxelGrid<PointNormal> grid;
  grid.setInputCloud(cloudIn);
  grid.setDownsampleAllData(true);
  grid.setLeafSize(voxelSize, voxelSize, voxelSize);
  grid.filter(cloudOut);

  PclPointCloudNormalToNewArrays(cloudOut, pointsOut, normalsOut, nPointsOut);
  return 0;
}
