import os
import json
import requests
from dotenv import load_dotenv
import dashscope
from dashscope import Generation

load_dotenv()

# 设置 DashScope API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30

# 和风天气API配置
QWEATHER_API_KEY = os.environ.get('QWEATHER_API_KEY', "687ed20ce0ce4567b032fde58a5938a8")
CITY_LOOKUP_URL = "https://geoapi.qweather.com/v2/city/lookup"
WEATHER_NOW_URL = "https://devapi.qweather.com/v7/weather/now"

def get_weather(location: str) -> str:
    """查询指定地点的天气信息"""
    try:
        # 第一步：查找 location ID
        lookup_params = {"location": location, "key": QWEATHER_API_KEY}
        resp_lookup = requests.get(CITY_LOOKUP_URL, params=lookup_params, timeout=5)
        resp_lookup.raise_for_status()
        data_lookup = resp_lookup.json()
        
        if data_lookup.get("location") and len(data_lookup["location"]) > 0:
            city_id = data_lookup["location"][0]["id"]
        else:
            return f"未找到地点: {location}"
        
        # 第二步：根据 location ID 获取实时天气
        now_params = {"location": city_id, "key": QWEATHER_API_KEY}
        resp_now = requests.get(WEATHER_NOW_URL, params=now_params, timeout=5)
        resp_now.raise_for_status()
        data_now = resp_now.json()
        
        if data_now.get("code") != "200":
            return f"天气数据异常: {data_now.get('message', '未知错误')}"
        
        now_data = data_now.get("now", {})
        if not now_data:
            return "天气数据为空"
        
        # 转换为用户友好的格式
        location_info = data_lookup["location"][0]
        return f"{location}的天气：{now_data['text']}，温度{now_data['temp']}°C，湿度{now_data['humidity']}%，{now_data['windDir']}{now_data['windScale']}级"
        
    except Exception as e:
        return f"查询天气失败：{str(e)}"

def query_database(query: str) -> str:
    """执行SQL查询"""
    try:
        # 这里可以连接实际数据库
        # 为了演示，我们返回模拟数据
        return f"执行查询：{query}\n结果：查询成功，返回3条记录"
    except Exception as e:
        return f"数据库查询失败：{str(e)}"

# 定义工具列表
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
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "执行SQL数据库查询",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要执行的SQL查询语句"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def run_function_call_demo():
    """运行Function Calling演示"""
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以帮助用户查询天气信息和数据库信息。"},
        {"role": "user", "content": "请帮我查询北京的天气，然后查询用户表中的所有数据"}
    ]
    
    print("=== Function Calling 演示 ===")
    print(f"用户问题：{messages[1]['content']}")
    
    # 第一次调用 - 获取工具调用
    response = Generation.call(
        model="qwen-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    if response.status_code == 200:
        message = response.output.choices[0].message
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print("\n检测到工具调用：")
            
            # 处理工具调用
            tool_messages = []
            for tool_call in message.tool_calls:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                
                print(f"调用工具：{func_name}")
                print(f"参数：{func_args}")
                
                # 执行工具函数
                if func_name == "get_weather":
                    result = get_weather(**func_args)
                elif func_name == "query_database":
                    result = query_database(**func_args)
                else:
                    result = f"未知工具：{func_name}"
                
                print(f"工具结果：{result}")
                
                # 添加工具调用和结果到消息列表
                tool_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": func_name,
                    "content": result
                })
            
            # 第二次调用 - 生成最终回复
            print("\n生成最终回复：")
            final_messages = messages + tool_messages
            final_response = Generation.call(
                model="qwen-turbo",
                messages=final_messages,
                tools=tools,
                tool_choice="none"  # 不需要再次调用工具
            )
            
            if final_response.status_code == 200:
                final_content = final_response.output.choices[0].message.content
                print(f"最终回复：{final_content}")
            else:
                print(f"最终回复失败：{final_response.message}")
        else:
            print("没有检测到工具调用")
            print(f"直接回复：{message.content}")
    else:
        print(f"请求失败：{response.message}")

def run_interactive_mode():
    """交互式模式"""
    print("=== 交互式 Function Calling 模式 ===")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助")
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            elif user_input.lower() == 'help':
                print("帮助信息：")
                print("- 可以询问天气，如：'北京今天天气怎么样？'")
                print("- 可以查询数据库，如：'查询用户表数据'")
                print("- 输入 'quit' 或 'exit' 退出")
                continue
            elif not user_input:
                continue
            
            messages = [
                {"role": "system", "content": "你是一个智能助手，可以帮助用户查询天气信息和数据库信息。"},
                {"role": "user", "content": user_input}
            ]
            
            # 调用模型
            response = Generation.call(
                model="qwen-turbo",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            if response.status_code == 200:
                message = response.output.choices[0].message
                
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    print("检测到工具调用...")
                    
                    # 处理工具调用
                    tool_messages = []
                    for tool_call in message.tool_calls:
                        func_name = tool_call["function"]["name"]
                        func_args = json.loads(tool_call["function"]["arguments"])
                        
                        # 执行工具函数
                        if func_name == "get_weather":
                            result = get_weather(**func_args)
                        elif func_name == "query_database":
                            result = query_database(**func_args)
                        else:
                            result = f"未知工具：{func_name}"
                        
                        # 添加工具调用和结果到消息列表
                        tool_messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call]
                        })
                        tool_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": func_name,
                            "content": result
                        })
                    
                    # 生成最终回复
                    final_messages = messages + tool_messages
                    final_response = Generation.call(
                        model="qwen-turbo",
                        messages=final_messages,
                        tools=tools,
                        tool_choice="none"
                    )
                    
                    if final_response.status_code == 200:
                        final_content = final_response.output.choices[0].message.content
                        print(f"助手: {final_content}")
                    else:
                        print(f"回复失败：{final_response.message}")
                else:
                    print(f"助手: {message.content}")
            else:
                print(f"请求失败：{response.message}")
                
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"处理请求时出错: {str(e)}")

if __name__ == "__main__":
    print("选择运行模式：")
    print("1. 演示模式 - 运行预设的演示")
    print("2. 交互模式 - 与助手对话")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        run_function_call_demo()
    elif choice == "2":
        run_interactive_mode()
    else:
        print("无效选择，运行演示模式")
        run_function_call_demo()
