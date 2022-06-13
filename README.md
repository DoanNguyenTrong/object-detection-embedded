# Project Object Detection

## Cài đặt OpenVINO
-  Cài OpenVINO Runtime bằng bộ cài [tại đây](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#doxid-openvino-docs-install-guides-installing-openvino-linux). Lưu ý khi chạy lệnh mở bộ cài 
```shell
  ./l_openvino_toolkit_p_<version>.sh
```
có thể chạy với quyền sudo để cài vào /opt

- Mỗi lần làm việc với OpenVINO cần khởi tạo môi trường bằng lệnh:
```shell
  source <đường dẫn đến file cài đặt OpenVINO>/setupvars.sh
```
có thể lưu lệnh này vào file .bashrc

## Cài đặt OpenCV
Hướng dẫn cài đặt OpenCV tại document của OpenCV [link](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) 

## Chạy project
- cd đến workspace chứa file src vừa clone, chạy catkin_make
- Chạy lệnh:
```shell
  source devel/setup.bash
```
- Mở terminal, chạy lệnh ``roscore``
- Mở terminal khác, chạy 
```shell
  rosrun integrated_camera_node integrated_camera_node 
```
để chạy rosnode dùng webcam của laptop và publish image cho node dùng OpenVINO
- Mở terminal khác, chạy 
```shell
  rosrun test_openvino vino_node 
```
để chạy rosnode subcribe image từ node trên và dùng OpenVINO suy luận lấy kết quả và publish message với data là BoundingBoxArray chứa tọa độ của hình chữ nhật và độ chính xác

## Lưu ý
Chỉnh lại thông số ảnh đầu vào tại dòng 29, 30 và đường dẫn file đến model trong file main.cpp của package test_openvino
Nhấn q vào cửa sổ để quit các app

## Enjoy :>
