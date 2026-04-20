# Gunicorn 适配SAM3大模型的终极配置
import multiprocessing

# 基础绑定
bind = "0.0.0.0:5000"
workers = 1  # 单Worker，避免多模型加载
threads = 8  # 增加线程数，提升并发处理能力（根据CPU核心数调整）
worker_class = "gevent"  # 关键：改用gevent异步Worker，更好处理长连接
worker_connections = 1000  # 允许更多并发连接

# 超时配置（拉满到30分钟，覆盖所有场景）
timeout = 1800  # 30分钟：请求处理最大超时
keepalive = 60  # 保持连接60秒，避免客户端频繁重连
graceful_timeout = 300  # 优雅退出超时
sendfile = False  # 关闭sendfile，避免长连接异常
tcp_nodelay = True  # 开启TCP_NODELAY，提升长连接稳定性

# 日志配置（便于排查超时问题）
pidfile = "/mnt/pengyi-sam3/sam3-main/logs/sam3.pid"
accesslog = "/mnt/pengyi-sam3/sam3-main/logs/access.log"
errorlog = "/mnt/pengyi-sam3/sam3-main/logs/error.log"
loglevel = "debug"  # 调试级别，记录所有超时/连接日志
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'  # 记录请求耗时

# 后台运行（调试时可改为False）
daemon = True

# 额外优化：限制内存使用，避免OOM
limit_request_line = 4096
limit_request_fields = 100
