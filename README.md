# Sam2+yolo26_Automated_labeling
SAM2+yolo26 achieves semi-automatic annotation


2026.2.4
# 实现功能：
  - 1.yolo26+sam2 sam2根据yolo26的检测框自动化标注语义分割mask（只标注yolo26识别到的，可根据需求微调yolo26），保存yolo格式、labelme的json格式、标准的json格式、分割标注图
  - 2.通过配置文件统一设置参数，标准的json格式正确与否待验证
  - 3.调用千问大模型对分割质量打分
  - 4.根据评分对标注数据进行分类，对需要的数据集人工微调

  ## yolo26与sam2权重根据精度需求可调，本项目用的是yolo26和sam2.1_hiera_large

    - yolo26权重放入 weights/pretrained
    - sam权重放入 segment-anything-2/checkpoints
    
dependencies:
  - python=3.11.14
  - pytorch=2.5.1=py3.11_cuda12.4_cudnn9.1.0_0
  - torchvision=0.20.1=py311_cu124
  - cudatoolkit=12.4.0
  - numpy=1.26.4
  - opencv=4.10.0
  - pip:
    - ultralytics==8.3.0
    - supervision==0.25.0
    - omegaconf==2.3.0
    - hydra-core==1.3.2
    - tqdm==4.66.5
    - matplotlib==3.9.2
    - pyyaml==6.0.2

# 自动标注脚本
执行命令时不要进入cuda环境

#1：对ros2 录制的bag标注
source /opt/ros/humble/setup.bash
运行run.sh
./run.sh \
    --bag /home/why/rezone_space/bags/rosbag2_2026_03_10-10_46_58_停车场 \   标注包
    --output /home/why/rezone_space/my_output \    输出地址
    --topic /camera/color/image_raw/compressed \   节点
    --step 2.0 \                                   每隔几秒抽取图片
    --conf 0.3 \                                   标注置信度,低于的不标注
    --device cpu                                   若启动gpu 更改为cuda
    
#运行完整的一键脚本
  - ./run.sh \
    --bag /home/why/rezone_space/bags/rosbag2_2026_03_10-10_46_58_停车场 \
    --output /home/why/rezone_space/my_output \
    --topic /camera/color/image_raw/compressed \
    --step 2.0 \
    --conf 0.3 \
    --device cpu
    
#2：对图片进行标注
  - python Automated_labeling.py \
    --input input \                  需要评标注的数据）
    --output quality_scores.json \   输出地址
    --conf 0.3 \                     标注置信度,低于的不标注
    --device cpu                     若启动gpu 更改为cuda
    
#运行完整的一键脚本
  - python Automated_labeling.py \
    --input input \                  
    --output quality_scores.json \   
    --conf 0.3 \                     
    --device cpu                    

# 评分脚本
 评分时在cuda环境下

  - python annotation_quality_evaluation.py \
    --composite_dir API_test \       需要评分的数据（mask与原图叠加）
    --api_key "xxxxxxxxxxxxxx" \     API秘钥
    --output quality_scores.json \   输出地址
    --max_samples 3                  最多要处理的图片数量

  - python annotation_quality_evaluation.py \
    --composite_dir API_test \
    --api_key "sk-XXXXXXXXXXXXX" \
    --output quality_scores.json \
    --max_samples 10
    
    
 # 分类脚本

  - python classify_by_score.py \
    --results quality_scores.json \  打分输出文件
    --source test_API \              数据集文件
    --output ./classified_images \   输出地址文件
    --low 5.0 \                      低质量分数，低于该分数的归为一类，根据需要调整
    --high 7.0
     --move                          可选参数move：移动图片文件进入新文件夹，不设置默认是复制   --report：生成报告

  - python classify_by_score.py \
    --results quality_scores.json \
    --source API_test \
    --output ./classified_images \
    --low 5.0 \
    --high 7.0 \
    --move 
    
    
# 其他文件及其作用
  - sam2_interactive.py        实现sam2半自动提示点标注（人工点击需要标注的物体）
  - yolo26_test.py             测试yolo26
  - yolowold_test.py           测试yolowold
  - extract_images_from_bag.py 将ros2bag 提取为png图片
