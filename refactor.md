sam3-main/
├── sam3/                          # 原始核心库（不动）
├── sam3_ext/                      # 你的扩展层
│   ├── __init__.py
│   ├── api/                       # API层
│   │   ├── __init__.py
│   │   └── app.py                 # Flask服务
│   ├── inference/                 # 推理模块
│   │   ├── __init__.py
│   │   ├── test_picture_to_picture_optimizer_batch_inference.py
│   │   ├── test_picture_to_picture_optimizer.py
│   │   ├── test_words_picture.py
│   │   ├── word_picture_batch_inference.py
│   │   └── picture_to_picture_optimizer.py
│   ├── services/                  # 业务服务
│   │   ├── __init__.py
│   │   ├── detector_service.py
│   │   └── feature_store.py
│   └── utils/                     # 工具函数
│       ├── __init__.py
│       ├── box_transform.py
│       └── image_utils.py
├── configs/                       # 配置文件目录
├── tests/                         # 测试文件目录
├── run_api.py                     # 启动入口
└── [其他原有文件]
