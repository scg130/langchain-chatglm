# 并发请求限流和超时保护

## ✅ 已实现的功能

### 1. 并发请求限流

**功能说明：**
- 使用 `Semaphore` 控制同时处理的请求数量
- 同步请求使用 `threading.Semaphore`
- 异步请求使用 `asyncio.Semaphore`
- 默认最大并发数：3
- 获取槽位超时：30秒

**配置参数：**
```python
max_concurrent_requests: int = 3  # 最大并发请求数
```

**使用示例：**
```python
from core.llm_chatglm import StableLLM

# 创建带限流的 LLM（默认最大3个并发）
llm = StableLLM(max_concurrent_requests=3)

# 或者使用工厂方法
from core.llm_chatglm import LLMFactory
llm = LLMFactory.create_llm(max_concurrent_requests=5)
```

### 2. 超时保护

**功能说明：**
- 同步请求使用 `ThreadPoolExecutor` + `future.result(timeout)`
- 异步请求使用 `asyncio.wait_for()`
- 流式请求在生成过程中检查超时
- 默认超时时间：120秒

**配置参数：**
```python
request_timeout: float = 120.0  # 请求超时时间（秒）
```

**使用示例：**
```python
# 设置超时时间为60秒
llm = StableLLM(request_timeout=60.0)

# 或者使用工厂方法
llm = LLMFactory.create_llm(request_timeout=90.0)
```

## 实现细节

### 同步请求（invoke）

```python
def invoke(self, input: Any, config: Optional[dict] = None) -> str:
    with self._rate_limit_context():  # 并发限流
        return self._execute_with_timeout(_invoke_internal)  # 超时保护
```

**保护机制：**
1. 获取信号量槽位（最多等待30秒）
2. 执行请求（最多等待 `request_timeout` 秒）
3. 自动释放信号量槽位

### 异步请求（astream）

```python
async def astream(self, input: Any, config: Optional[dict] = None):
    async with self._async_rate_limit_context():  # 异步并发限流
        stream = await asyncio.wait_for(  # 超时保护
            loop.run_in_executor(None, _stream_wrapper),
            timeout=self.request_timeout
        )
        for chunk in stream:
            yield chunk
```

**保护机制：**
1. 获取异步信号量槽位（最多等待30秒）
2. 在线程池中执行流式生成
3. 使用 `asyncio.wait_for` 设置总超时
4. 流式输出过程中检查超时
5. 自动释放信号量槽位

### 同步流式请求（stream）

```python
def stream(self, input: Any, config: Optional[dict] = None):
    with self._rate_limit_context():  # 并发限流
        # 流式生成
        for new_text in streamer:
            if time.time() - start_time > self.request_timeout:  # 超时检查
                break
            yield new_text
```

**保护机制：**
1. 获取信号量槽位
2. 流式生成过程中持续检查超时
3. 超时后自动中断生成
4. 自动释放信号量槽位

## 错误处理

### 限流错误

当无法获取处理槽位时：
```python
RuntimeError: "请求限流：无法获取处理槽位，请稍后重试"
```

### 超时错误

当请求超过超时时间时：
```python
TimeoutError: "请求超时：超过 {request_timeout} 秒未完成"
```

### 流式超时

流式生成超时时会：
- 记录警告日志
- 中断生成循环
- 返回错误消息（异步流式）

## 配置建议

### 生产环境

```python
llm = StableLLM(
    max_concurrent_requests=2,  # 保守的并发数
    request_timeout=180.0,      # 3分钟超时
)
```

### 开发环境

```python
llm = StableLLM(
    max_concurrent_requests=5,  # 更高的并发数
    request_timeout=60.0,       # 1分钟超时
)
```

### 高负载环境

```python
llm = StableLLM(
    max_concurrent_requests=1,  # 单请求处理
    request_timeout=300.0,       # 5分钟超时
)
```

## 监控和日志

所有限流和超时事件都会记录日志：

```python
# 限流日志
logger.warning("请求限流：无法获取处理槽位")

# 超时日志
logger.error("请求超时: 超过 120.0 秒未完成")
logger.warning("流式生成超时: 超过 120.0 秒")
```

## 性能影响

- **并发限流**：轻微延迟（获取槽位时）
- **超时保护**：几乎无性能影响
- **流式超时检查**：每次 yield 时检查，开销极小

## 最佳实践

1. **根据硬件配置调整并发数**
   - CPU: 1-2
   - GPU: 2-5
   - 多GPU: 5-10

2. **根据模型大小调整超时**
   - 小模型（<1B）: 60秒
   - 中模型（1-7B）: 120秒
   - 大模型（>7B）: 180-300秒

3. **监控限流和超时频率**
   - 如果频繁限流，考虑增加并发数
   - 如果频繁超时，考虑增加超时时间或优化模型

