# Function Calling 与协作示例

这个目录包含了使用阿里云通义千问（Qwen）模型进行Function Calling的完整示例。

## 文件说明

### 1. `simple_function_call.py` - 简化版Function Calling示例
这是一个完整的Function Calling演示，展示了如何：
- 定义工具函数
- 配置工具描述
- 处理工具调用
- 生成最终回复

**功能特性：**
- 天气查询工具
- 数据库查询工具
- 支持多个测试用例
- 完整的错误处理

### 2. `qwen_agent_functincall.py` - Qwen Agent版本
使用qwen-agent框架的版本（由于依赖问题暂时注释了GUI组件）。

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv 安装依赖
uv sync

# 或者使用 pip 安装
pip install dashscope python-dotenv
```

### 2. 配置API Key

创建 `.env` 文件：

```bash
# DashScope API Key (阿里云通义千问)
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 3. 运行示例

```bash
# 运行简化版示例
uv run python simple_function_call.py

# 选择测试模式：
# 1. 单个测试 - 运行一个完整的Function Calling流程
# 2. 多个测试 - 运行多个测试用例
```

## 核心概念

### Function Calling 流程

1. **用户提问** - 用户提出需要调用工具的问题
2. **模型分析** - Qwen模型分析问题并决定调用哪些工具
3. **工具调用** - 执行相应的工具函数
4. **结果整合** - 将工具结果整合到对话中
5. **生成回复** - 生成最终的用户友好回复

### 工具定义

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定地点的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "要查询天气的地点名称"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

### 工具函数实现

```python
def get_weather(location: str) -> str:
    """查询指定地点的天气信息"""
    try:
        # 调用实际的天气API
        return f"{location}的天气：晴天，温度25°C，湿度60%，风力2级"
    except Exception as e:
        return f"查询天气失败：{str(e)}"
```

## 测试用例

### 天气查询
- "北京今天天气怎么样？"
- "深圳的天气如何？"

### 数据库查询
- "帮我查询用户表中的所有数据"
- "查询订单表的数据"

### 综合查询
- "请查询上海的天气，然后查询订单表的数据"

## 错误处理

代码包含了完善的错误处理机制：
- API调用失败处理
- 工具执行异常处理
- 参数验证
- 网络超时处理

## 扩展功能

### 添加新工具

1. 定义工具函数
2. 在tools列表中添加工具描述
3. 在工具调用处理逻辑中添加对应的执行代码

### 集成真实API

将模拟数据替换为真实的API调用：
- 天气API（如和风天气、高德天气）
- 数据库连接（如MySQL、PostgreSQL）
- 其他外部服务

## 注意事项

1. **API Key安全** - 使用环境变量存储API Key，不要硬编码
2. **错误处理** - 为每个工具函数添加适当的错误处理
3. **参数验证** - 验证工具函数的输入参数
4. **性能优化** - 考虑工具调用的性能影响

## 依赖版本

- dashscope >= 1.11.0
- python-dotenv >= 1.0.0
- requests >= 2.25.0

## 许可证

MIT License