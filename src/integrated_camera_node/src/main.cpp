#include <iostream>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "publish_image");
    ros::NodeHandle n;
    
    ros::Publisher pb = n.advertise<sensor_msgs::Image>("/integrated_camera/image_2D",1);

    cv::VideoCapture cap(0);
    cv::Mat frame;

    while(ros::ok)
    {
        try
        {      
            cap.read(frame);
            cv_bridge::CvImage msg;
            if(frame.empty()) std::cout << "No image available" << std::endl;
            else 
            {
                msg.image = frame;
                msg.encoding = "bgr8";
            }
            pb.publish(msg.toImageMsg());
            cv::imshow("Camera",frame);
            int k = cv::waitKey(1);
            if (k=='q') break;
            ros::spinOnce();
        }
        catch (cv_bridge::Exception& e) 
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return 0;
        }
    }

    return 0;
}