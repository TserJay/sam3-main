"""
SAM3 扩展模块入口
启动 Flask API 服务
"""
from sam3_ext.api.app import app, get_api_instance

if __name__ == '__main__':
    # 启动时初始化API
    get_api_instance()
    
    # 启动Flask服务
    print("Flask API服务启动中...（支持Base64+JSON请求）")
    app.run(host='0.0.0.0', port=5000, debug=True)
