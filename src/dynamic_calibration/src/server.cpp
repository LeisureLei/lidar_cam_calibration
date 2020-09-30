#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include "image_geometry/pinhole_camera_model.h"
#include <dynamic_reconfigure/server.h>
#include <dynamic_calibration/calibrationConfig.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"
#include <queue>
#include <map>
#include <ceres/ceres.h>

using namespace std;

int size_buf = 1;
int reject_dis;
queue<sensor_msgs::ImageConstPtr> ImgQueue;
queue<sensor_msgs::PointCloud2ConstPtr> LidarQueue;
vector<cv::Point2d> ImgPoints;
map<vector<double>, Eigen::Vector3d> search_map; //2D点与对应的3D点

ros::Subscriber ImgSub;
ros::Subscriber LidarSub;
ros::Subscriber CameraInfoSub;
ros::Publisher CamPointCloudPub;
ros::Publisher PointCloudImagePub;

Eigen::Matrix4d cam_T_lidar = Eigen::Matrix4d::Identity();  //外参
Eigen::Matrix4d K;  //相机内参

bool flag = false;
bool filter = false;
bool mousecall = false;
bool first = true;
Eigen::Matrix<double,4,4> R_rect_00;
Eigen::Matrix<double,3,4> P_rect_00;

cv::VideoWriter outputVideo;

void callback(dynamic_calibration::calibrationConfig &config, uint32_t level)
{	
	//get para
	cout<<"config.roll:"<<config.roll<<endl
		<<"config.pitch:"<<config.pitch<<endl
		<<"config.yaw:"<<config.yaw<<endl
		<<"config.x:"<<config.x<<endl
		<<"config.y:"<<config.y<<endl
		<<"config.z:"<<config.z<<endl<<endl;
	Eigen::Vector3d eulerAngle(config.yaw, config.pitch, config.roll); //eulerAngle:YPR
	Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(2),Eigen::Vector3d::UnitX()));
	Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1),Eigen::Vector3d::UnitY()));
	Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle(0),Eigen::Vector3d::UnitZ()));
 	Eigen::Matrix3d R;
 	R = yawAngle*pitchAngle*rollAngle;
	cam_T_lidar.block<3,3>(0,0) = R;
	cam_T_lidar.block<3,1>(0,3) = Eigen::Vector3d(config.x, config.y, config.z);
	cout<<"cam_T_lidar:"<<cam_T_lidar<<endl;
	flag = config.next;
	filter = config.dis_filter;
	mousecall = config.get_point;
	reject_dis = config.reject_distance;
}

void ImgCallback(const sensor_msgs::ImageConstPtr& ImgMsg)
{	
	if(ImgQueue.size()==size_buf)
	{	
		//cout<<"Get img size:"<<ImgQueue.size()<<endl;
		return;
	}
	else
	{
		ImgQueue.push(ImgMsg);
	}	
}

void LidarCallback(const sensor_msgs::PointCloud2ConstPtr& LidarMsg)
{
	if(LidarQueue.size()==size_buf)
	{
		//cout<<"Get lidar size: "<<LidarQueue.size()<<endl;
		return;
	}
	else
	{
		LidarQueue.push(LidarMsg);
	}
}

void CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info)
{
	image_geometry::PinholeCameraModel camera_model;
  	camera_model.fromCameraInfo(camera_info);
	double fx = camera_model.fx(); 
  	double fy = camera_model.fy(); 
  	double cx = camera_model.cx(); 
  	double cy = camera_model.cy(); 
	
	K << fx, 0, cx, 0,
		 0, fy, cy, 0,
		 0, 0, 1, 0,
		 0, 0, 0, 1;
}

void On_mouse(int event, int x, int y, int flags, void*)
{
	if(event == CV_EVENT_LBUTTONDOWN)
	{
		ImgPoints.push_back(cv::Point(x,y));
	}
}

struct ReprojectedError
{
    private:
        cv::Point3d p3;
        cv::Point2d p2;

    public:
        ReprojectedError(cv::Point3d P3, cv::Point2d P2):p3(P3),p2(P2){}

    template<typename T>
    bool operator()(const T* const q4x1, const T* const t3x1, T* residuals) const{
        Eigen::Quaternion<T> Rx(q4x1[0], q4x1[1],q4x1[2],q4x1[3]);
        Eigen::Matrix<T,3,1> tx;
        tx<<t3x1[0],t3x1[1],t3x1[2];
  
        Eigen::Vector3d p3_t(Eigen::Vector3d(p3.x, p3.y, p3.z));
        Eigen::Matrix<T,3,1> p3_T = p3_t.cast<T>();
        Eigen::Vector2d p2_t(Eigen::Vector2d(p2.x, p2.y));
        Eigen::Matrix<T,2,1> p2_T = p2_t.cast<T>();

        Eigen::Matrix<T,3,3> K_T = K.block<3,3>(0,0).cast<T>();

        Eigen::Matrix<T,3,1> Puv = K_T * (Rx.matrix()*p3_T + tx);
        residuals[0] = Puv[0]/Puv[2] - p2_T[0];
        residuals[1] = Puv[1]/Puv[2] - p2_T[1];

        return true;

    }
};

int main(int argc, char** argv)
{
	R_rect_00 << 9.999239e-01, 9.837760e-03, -7.445048e-03, 0,
		-9.869795e-03 ,	9.999421e-01 ,-4.278459e-03 ,0,
		7.402527e-03, 4.351614e-03 ,9.999631e-01,0,
		0,0,0,1;
	P_rect_00 << 7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00 ,
			0.000000e+00, 7.215377e+02 ,1.728540e+02 ,0.000000e+00 ,
			0.000000e+00, 0.000000e+00, 1.000000e+00 ,0.000000e+00;
		
	
    ros::init(argc, argv, "dynamic_calibration_node");
	ros::NodeHandle nh;

	ImgSub = nh.subscribe("/image_raw", 100, ImgCallback);
	LidarSub = nh.subscribe("/pointcloud", 100, LidarCallback);
	CameraInfoSub = nh.subscribe("/camera_info", 1, CameraInfoCallback);
	CamPointCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/camrea_pointcloud", 100);
	PointCloudImagePub = nh.advertise<sensor_msgs::Image>("/PointCloudImg", 100);

    dynamic_reconfigure::Server<dynamic_calibration::calibrationConfig> server;
    dynamic_reconfigure::Server<dynamic_calibration::calibrationConfig>::CallbackType f;
    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);

	while(ros::ok())
	{	
		if( !ImgQueue.empty() && !LidarQueue.empty() )
		{
			sensor_msgs::ImageConstPtr ImgMsgPtr = ImgQueue.front();
			sensor_msgs::PointCloud2ConstPtr LidarMsgPtr = LidarQueue.front();

			pcl::PointCloud<pcl::PointXYZ>::Ptr CamPointCloud(new pcl::PointCloud<pcl::PointXYZ>);

			//time sync
			double Img_time = ImgMsgPtr->header.stamp.toSec();
			double Lidar_time = LidarMsgPtr->header.stamp.toSec();
			if(Lidar_time - Img_time <= 0.020 && Lidar_time - Img_time >= -0.020)
			{
				//cout<<"lidar time:"<<to_string(Lidar_time)<<endl;
				//cout<<"Img_time:"<<to_string(Img_time)<<endl;
				//handle img, convert ros image to opencv image
				cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(*ImgMsgPtr, sensor_msgs::image_encodings::BGR8);
				cv::Mat img = cv_ptr -> image;
				//convert ros pointcloud to pcl pointcloud
				pcl::PointCloud<pcl::PointXYZ>::Ptr pclPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::fromROSMsg(*LidarMsgPtr, *pclPointCloud); 
				std::vector<int> index;
				pcl::removeNaNFromPointCloud(*pclPointCloud,*pclPointCloud,index);  	

				int size = pclPointCloud->points.size();
				// cout<<"pointcloud size:"<<size<<endl;
				if(first)
				{
					outputVideo.open("/home/leibing/kitti_calibration.avi",CV_FOURCC('M','J','P','G'),30,cv::Size(img.cols,img.rows),true);	
					first = false;
				}
	
				vector<cv::Point2d> points;
				for(size_t i=0; i<size; i++)  //project every point to image plane and visualize using opencv
				{
					cv::Point2d cv_point;
					Eigen::Vector4d P_lidar(
						pclPointCloud->points[i].x,
						pclPointCloud->points[i].y,
						pclPointCloud->points[i].z,
						1);
					Eigen::Vector3d Puv = P_rect_00*R_rect_00*cam_T_lidar*P_lidar;  //project pointcloud to image plane
					//reject points that are not in image
					if(  Puv[0] > 0 && Puv[1] > 0 && Puv[2] > 0 && Puv[0]/Puv[2] > 0 && Puv[1]/Puv[2] > 0 && Puv[0]/Puv[2] < img.cols && Puv[1]/Puv[2] < img.rows)
					{
						cv_point.x = Puv[0]/Puv[2];
						cv_point.y = Puv[1]/Puv[2];
						if(filter)  
						{	
							//reject too far points
							Eigen::Vector3d p(P_lidar[0], P_lidar[1], P_lidar[2]);
							double dis = p.norm();
							if( dis < reject_dis )
							{	
								vector<double> vectorpoint{cv_point.x,cv_point.y};
								search_map[vectorpoint] = p;
								cv::circle(img,cv_point,1,cv::Scalar(0,255,0));
								points.push_back(cv_point);
								CamPointCloud->points.push_back(pclPointCloud->points[i]);
							}
						}
						else
						{
							cv::circle(img,cv_point,0.8,cv::Scalar(0,255,255));
							points.push_back(cv_point);
							CamPointCloud->points.push_back(pclPointCloud->points[i]);
						}
					}	
				}
				// cout<<"img points size:"<<points.size()<<endl;
				//cout<<"reject_dis:"<<reject_dis<<endl;
				//publish campointcloud
				sensor_msgs::PointCloud2 pointcloud;
				pcl::toROSMsg(*CamPointCloud, pointcloud);
				pointcloud.header = ImgMsgPtr->header;
				CamPointCloudPub.publish(pointcloud);

				//publish image with pointclouds projected to img
				sensor_msgs::ImagePtr projectedImgMsg = cv_bridge::CvImage(ImgMsgPtr->header, "bgr8", img).toImageMsg();
				PointCloudImagePub.publish(projectedImgMsg);

				cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
				if(mousecall)
				{		
					cv::setMouseCallback("image", On_mouse);
					cout<<"get "<<ImgPoints.size()<<" points"<<endl;
					if(ImgPoints.size()==12)
					{
						//find 3D-2D points
						vector<cv::Point3d > objectPoints;
						vector<cv::Point2d > imagePoints;
					
						for(int i=0;i<ImgPoints.size();i++)
						{
							map< vector<double>, Eigen::Vector3d>::iterator iter;
							for(iter=search_map.begin();iter!=search_map.end();iter++)
							{
								Eigen::Vector2d d(iter->first[0]-ImgPoints[i].x, iter->first[1]-ImgPoints[i].y);
								if( d.norm() < 5 )
								{
									objectPoints.push_back( cv::Point3d( iter->second[0],iter->second[1],iter->second[2] ) );
									imagePoints.push_back( cv::Point2d(iter->first[0], iter->first[1] ) );
								}
							}
						}
						
						//pnp
						cv::Matx33d cameraMatrix(K(0,0),  0,    K(0,2),
                           						  0,    K(1,1), K(1,2),
                           						  0,      0,       1);
						cv::Vec4f distCoeffs(0,0,0,0);
						cv::Mat rvec, tvec;
						// if(objectPoints.size()==9)
						// {
							cout<<"use 9 points to solvePnP!"<<endl;
							cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
							Eigen::Matrix3d R;
  							cv::cv2eigen(rvec, R);
							cout<<"before opt R is:"<<endl<<R.eulerAngles(2,1,0)<<endl;
							cout<<"t is:"<<endl<<tvec.at<double>(0)<<" "<<tvec.at<double>(1)<<" "<<tvec.at<double>(2)<<endl;
							//ceres opt
							Eigen::Quaterniond q(R);
							double q_array[4] = {q.w(),q.x(),q.y(),q.z()};
							double t_array[3] = {tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};

							ceres::Problem problem;
							for(int i=0;i<objectPoints.size();i++)
							{
								ceres::CostFunction *costfunction = new ceres::AutoDiffCostFunction<ReprojectedError,2,4,3>(
									new ReprojectedError(objectPoints[i], imagePoints[i])
								);
								ceres::LossFunction * loss_function = new ceres::HuberLoss(1.0);
			        			problem.AddResidualBlock(costfunction, loss_function, q_array, t_array);
							}
							ceres::LocalParameterization* quaternionParameterization = new ceres::QuaternionParameterization;
  		        			problem.SetParameterization(q_array,quaternionParameterization);
							ceres::Solver::Options options;
  		        			options.linear_solver_type = ceres::DENSE_SCHUR;
							ceres::Solver::Summary summary;
 		        			ceres::Solve(options, &problem, & summary);
							q = Eigen::Quaterniond(q_array[0],q_array[1],q_array[2],q_array[3]);
							cout<<"After optmize R is:"<<endl<<q.matrix().eulerAngles(2,1,0)<<endl;
							cout<<"t is:"<<endl<<t_array[0]<<" "<<t_array[1]<<" "<<t_array[2]<<endl;

  							cam_T_lidar.block<3,3>(0,0) = q.matrix();
							cam_T_lidar.block<3,1>(0,3) << t_array[0], t_array[1], t_array[2];
						// }

						ImgPoints.clear();

						objectPoints.clear();
					}
					
					for(int i=0; i<ImgPoints.size();i++)
					{
						cv::circle(img,ImgPoints[i],2,cv::Scalar(255,255,0));
					}
				}
				cout <<"R:"<<endl<< cam_T_lidar.block<3,3>(0,0).eulerAngles(2,1,0)<<endl;
				cout<< "t:"<<endl<<cam_T_lidar.block<3,1>(0,3)<<endl;
				//opencv visualization
				cv::imshow("image", img);
				cv::waitKey(1);
				outputVideo.write(img);
				if(flag)
				{
					ImgQueue.pop();
					LidarQueue.pop();
					search_map.clear();
					continue;
				}
			}
			else if(Lidar_time - Img_time > 0.020)
			{
				ImgQueue.pop();
				continue;
			}
			else
			{
				LidarQueue.pop();
				continue;
			}
		}
		ros::spinOnce();
	}
    outputVideo.release();
    return 0;
}
