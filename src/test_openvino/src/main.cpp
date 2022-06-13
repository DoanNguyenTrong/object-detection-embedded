#include <iostream>
#include <string>
#include <vector>

// clang-format off
#include <opencv2/opencv.hpp>
#include "openvino/openvino.hpp"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "test_openvino/BoundingBoxArray.h"

// clang-format on

class HumanDetection
{
    private:
    ros::Publisher publisher;
    ros::Subscriber subcriber;
    std::string pub_topic = "/boxes_data";
    std::string sub_topic = "/integrated_camera/image_2D";
    // Config the model and device here
    ov::CompiledModel compiled_model;
    const std::string model_path = "/home/nguyen/project_detection/src/test_openvino/model/intel/person-detection-0203/FP32/person-detection-0203.xml";
    const std::string device_name = "CPU";
    //image properties
    size_t batch_size = 1;
    size_t image_channels = 3;
    size_t image_width=1280;
    size_t image_height=720;
    
    public:
    HumanDetection(ros::NodeHandle nh)
    {
        this->setup_model();
        publisher = nh.advertise<test_openvino::BoundingBoxArray>(pub_topic,1);
        subcriber = nh.subscribe(sub_topic,1,&HumanDetection::get_results,this);
        ros::spin();
    }

    void setup_model()
    {
        try
        {
            // Get OpenVINO runtime version
            std::cout << ov::get_openvino_version() << std::endl;

            // -------- Step 1. Initialize OpenVINO Runtime Core --------
            ov::Core core;

            // -------- Step 2. Read a model --------
            std::cout << "Loading model files: " << model_path << std::endl;
            std::shared_ptr<ov::Model> model = core.read_model(model_path);

            // -------- Step 3. Set up input

            // -------------------------------------------------------------------

            // Reshape model to image size and batch size
            // assume model layout NCHW
            const ov::Layout model_layout{"NCHW"};

            ov::Shape tensor_shape = model->input().get_shape();

            tensor_shape[ov::layout::batch_idx(model_layout)] = batch_size;
            tensor_shape[ov::layout::channels_idx(model_layout)] = image_channels;
            tensor_shape[ov::layout::height_idx(model_layout)] = image_height;
            tensor_shape[ov::layout::width_idx(model_layout)] = image_width;

            std::cout << "Reshape network to the image size = [" << image_height << "x" << image_width << "] " << std::endl;
            model->reshape({{model->input().get_any_name(), tensor_shape}});

            // just wrap image data by ov::Tensor without allocating of new memory

            const ov::Layout tensor_layout{"NHWC"};

            // -------- Step 4. Configure preprocessing --------

            ov::preprocess::PrePostProcessor ppp(model);
            
            // 1) input() with no args assumes a model has a single input
            ov::preprocess::InputInfo& input_info = ppp.input();
            // 2) Set input tensor information:
            // - precision of tensor is supposed to be 'u8'
            // - layout of data is 'NHWC'
            input_info.tensor().
                set_element_type(ov::element::u8).
                set_layout(tensor_layout);
            // 3) Adding explicit preprocessing steps:
            // - convert u8 to f32
            // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
            ppp.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW");
            // 4) Here we suppose model has 'NCHW' layout for input
            input_info.model().set_layout("NCHW");
            // 5) output () with no args assumes a model has a single output
            ov::preprocess::OutputInfo& output_info = ppp.output(0);
            // 6) declare output element type as FP32
            output_info.tensor().set_element_type(ov::element::f32);

            // 7) Apply preprocessing modifing the original 'model'
            model = ppp.build();

            // -------- Step 5. Loading a model to the device --------
            compiled_model = core.compile_model(model, device_name);
        }
        catch (const std::exception &e)
        {
            std::cout << "Couldn't load the model" << std::endl;
            std::cerr << e.what() << std::endl;
            return;
        }
        std::cout << "Load model sucessfully" << std::endl;
    }
  
    void get_results(sensor_msgs::Image msg)
    {
        //get the color frame from camera by using cv_bridge
        cv_bridge::CvImagePtr bridge;
        try 
        {
            bridge = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e) 
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    
        ov::Shape input_shape = {batch_size, image_height, image_width, image_channels};

        ov::element::Type input_type = ov::element::u8;
        // just wrap image data by ov::Tensor without allocating of new memory
        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, bridge->image.data);

        // -------- Step 6. Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Step 7. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output
        //0 is "boxes", 1 is "labels"
        ov::Tensor output_tensor = infer_request.get_output_tensor(0);//output_tensor is tensor "boxes" with 5 values

        // -------- Step 10. Visualize output and publish results
        test_openvino::BoundingBoxArray results;
        if (output_tensor.get_size()>0) for (int i = 0; i < output_tensor.get_size(); i+=5)
        {
            if (output_tensor.data<float>()[i+4]<=0.3) break; 

            results.boxes.resize(output_tensor.get_size());
            results.boxes[i/5].x = static_cast<int>(output_tensor.data<float>()[i]); //x of the upper left conner of the box
            results.boxes[i/5].y = static_cast<int>(output_tensor.data<float>()[i+1]); //y of the upper left conner of the box
            results.boxes[i/5].w = static_cast<int>(output_tensor.data<float>()[i+2] - output_tensor.data<float>()[i]); //width of the box
            results.boxes[i/5].h = static_cast<int>(output_tensor.data<float>()[i+3] - output_tensor.data<float>()[i+1]); //height of the box
            results.boxes[i/5].confidence = output_tensor.data<float>()[i+4]; //confidence

            // cv::Rect rect(results.x,results.y,results.w-results.x,results.h-results.y);
            cv::Rect rect(results.boxes[i/5].x,results.boxes[i/5].y,results.boxes[i/5].w,results.boxes[i/5].h);
            cv::rectangle(bridge->image,rect,cv::Scalar(0,255,0),1,8);
            cv::putText(bridge->image,std::to_string(results.boxes[i/5].confidence),cv::Point(results.boxes[i/5].x,results.boxes[i/5].y),1,cv::FONT_HERSHEY_COMPLEX,cv::Scalar(0,0,255),1,8);
        }

        publisher.publish(results);
        results.boxes.clear();
        cv::Mat final_result = bridge->image;    
        cv::imshow("Results", final_result);

        int k = cv::waitKey(30);
        if (k == 'q') 
        {
            cv::destroyAllWindows();
            ros::shutdown();      
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_openvino");
    ros::NodeHandle nh;
    HumanDetection dt = HumanDetection(nh);
     
    return 0;
}