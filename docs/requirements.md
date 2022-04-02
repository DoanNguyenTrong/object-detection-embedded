# Problem: Object detection (human) in factory
- Re-train (validate) model using custom data (data collected from Rostek customer’s warehouse) or come up with our own architecture for NN. If using open source data, collect it and dive in to understand how they organized their data.
    - Ask Rostek team helping collect some data 
    - Understand how open source projects organize data. Do creating our own data if necessary
    - Play with models, code, inference
- Optimize/Quantize using openvino c++ framework
- Run inference on a ROS node with input from Intel RealSense (D435/D455) and publish in a defined message format
