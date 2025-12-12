# Qwen2.5 代码升级说明

## 参考 Qwen3 示例的改进

### 1. 模型加载优化
**改进前：**
```python
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch_dtype,
    device_map="auto"
)
```

**改进后（参考 Qwen3）：**
```python
torch_dtype = "auto"  # CUDA 模式下自动推断
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch_dtype,
    device_map="auto"
)
```

### 2. Tokenizer 使用优化
**改进前：**
```python
prompt = tokenizer.apply_chat_template(...)
inputs = tokenizer(prompt, return_tensors="pt")
```

**改进后（参考 Qwen3）：**
```python
text = tokenizer.apply_chat_template(...)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
```

### 3. 生成输出提取优化
**改进前：**
```python
response = tokenizer.decode(
    outputs[0][inputs['input_ids'].shape[-1]:],
    skip_special_tokens=True
)
```

**改进后（参考 Qwen3）：**
```python
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
response = tokenizer.decode(output_ids, skip_special_tokens=True)
```

### 4. 主要区别

| 特性 | Qwen3 | Qwen2.5 |
|------|-------|---------|
| Thinking 模式 | ✅ 支持 `enable_thinking=True` | ❌ 不支持 |
| Thinking 解析 | ✅ 需要解析 `</think>` | ❌ 不需要 |
| Chat Template | ✅ 支持 | ✅ 支持 |
| Auto dtype | ✅ 支持 `torch_dtype="auto"` | ✅ 支持 |

## 已完成的改进

1. ✅ 优化模型加载参数（使用 `torch_dtype="auto"`）
2. ✅ 改进 tokenizer 输入格式（使用列表包装）
3. ✅ 优化输出提取方式（参考 Qwen3 示例）
4. ✅ 统一变量命名（`model_inputs` 替代 `inputs`）
5. ✅ 移除 thinking 相关代码（Qwen2.5 不支持）

## 代码对比

### Qwen3 示例代码
```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Qwen3 特有
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
# 解析 thinking content（Qwen3 特有）
```

### Qwen2.5 升级后代码
```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    # 不包含 enable_thinking（Qwen2.5 不支持）
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
response = tokenizer.decode(output_ids, skip_special_tokens=True)
# 不需要解析 thinking content
```

## 优势

1. **更现代的 API 使用**：参考 Qwen3 的最佳实践
2. **更好的性能**：使用 `torch_dtype="auto"` 自动优化
3. **更清晰的代码**：统一的变量命名和结构
4. **更好的兼容性**：适配 Qwen2.5 的特性

## 注意事项

- Qwen2.5 不支持 thinking 模式，所以移除了相关代码
- `max_new_tokens` 保持为 512（可根据需要调整）
- 保留了原有的停止条件和清理逻辑

