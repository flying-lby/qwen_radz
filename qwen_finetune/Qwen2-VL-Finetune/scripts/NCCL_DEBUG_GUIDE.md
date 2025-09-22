# NCCL分布式训练故障排除指南

## 常见NCCL错误及解决方案

### 1. "failed to recv, got 0 bytes" 错误

#### 症状
- 多个rank无法从rank 0获取ncclUniqueId
- TCPStore通信失败
- 可能提示应用程序崩溃或网络设置问题

#### 解决方案
1. **检查网络配置**
   ```bash
   # 检查网络接口
   ip addr show
   # 设置NCCL网络接口
   export NCCL_SOCKET_IFNAME=eth0  # 或你的实际网络接口
   ```

2. **端口冲突检查**
   ```bash
   # 检查端口是否被占用
   netstat -tuln | grep 29500
   # 或使用其他端口
   deepspeed --master_port=29501 ...
   ```

3. **GPU设备配置**
   ```bash
   # 明确指定GPU设备
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   deepspeed --num_gpus=8 ...
   ```

### 2. NCCL环境变量配置

#### 必要环境变量
```bash
# GPU设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL网络配置
export NCCL_SOCKET_IFNAME=eth0        # 网络接口
export NCCL_IB_DISABLE=1              # 禁用InfiniBand
export NCCL_DEBUG=INFO                # 调试级别
export NCCL_TREE_THRESHOLD=0          # 树形通信阈值
export NCCL_TIMEOUT=600               # 超时设置(秒)
```

#### 高级配置
```bash
# 性能优化
export NCCL_BUFFSIZE=8388608
export NCCL_NTHREADS=1
export NCCL_MAX_NCHANNELS=32

# 故障排除
export NCCL_DEBUG_SUBSYS=ALL         # 详细调试
export NCCL_LAUNCH_MODE=PARALLEL     # 启动模式
```

### 3. DeepSpeed配置优化

#### ZeRO-3参数调整
```json
{
  "zero_optimization": {
    "stage": 3,
    "sub_group_size": 1e8,              // 减小避免内存问题
    "stage3_max_live_parameters": 1e8,  // 限制活跃参数
    "stage3_max_reuse_distance": 1e8    // 限制重用距离
  },
  "communication_data_type": "fp32",    // 稳定的通信类型
  "gradient_clipping": 1.0,             // 梯度裁剪
  "steps_per_print": 1                  // 增加日志输出
}
```

### 4. 分步调试方法

#### Step 1: 单GPU测试
```bash
CUDA_VISIBLE_DEVICES=0 python src/train/clip_train_improved.py ...
```

#### Step 2: 双GPU测试
```bash
export CUDA_VISIBLE_DEVICES=0,1
deepspeed --num_gpus=2 --master_port=29500 ...
```

#### Step 3: 逐步增加GPU数量
```bash
# 4GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
deepspeed --num_gpus=4 ...

# 8GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed --num_gpus=8 ...
```

### 5. 故障诊断命令

#### 系统检查
```bash
# GPU状态
nvidia-smi

# 网络接口
ip addr show

# 端口占用
netstat -tuln | grep 29500

# 进程检查
ps aux | grep deepspeed
```

#### NCCL测试
```bash
# NCCL通信测试
mpirun -np 8 $NCCL_HOME/build/test/single/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### 6. 常见问题FAQ

**Q: 为什么rank 0总是先崩溃？**
A: rank 0通常负责协调，可能因为内存不足或网络配置问题率先失败。

**Q: 如何确定正确的网络接口？**
A: 使用`ip route get 8.8.8.8`查看默认路由接口。

**Q: DeepSpeed启动卡住怎么办？**
A: 检查防火墙设置，确保端口可访问，设置NCCL_DEBUG=INFO查看详细日志。

### 7. 预防措施

1. **使用一致的配置模板**
2. **定期检查GPU和网络状态**
3. **监控端口占用情况**
4. **设置合理的超时参数**
5. **保留详细的训练日志**

