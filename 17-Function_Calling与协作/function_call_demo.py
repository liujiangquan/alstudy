import requests
from http import HTTPStatus
import dashscope
import os
import json
import logging
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# 设置 DashScope API Key
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
#api_key = os.environ.get('DASHSCOPE_API_KEY')

# 高德天气 API 的 天气工具定义（JSON 格式）
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 北京",
                },
                "adcode": {
                    "type": "string",
                    "description": "The city code, e.g. 110000 (北京)",
                }
            },
            "required": ["location"],
        },
    },
}

# 高德天气 API 的 天气工具定义（JSON 格式）
weather_tool_qweather = {
    "type": "function",
    "function": {
        "name": "get_current_weather_qweather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 北京",
                },
            },
            "required": ["location"],
        },
    },
}

# 配置（请替换成你自己的 API Key）
QWEATHER_API_KEY = os.environ.get('QWEATHER_API_KEY', "687ed20ce0ce4567b032fde58a5938a8")

# 和风天气接口地址（注意：无空格！）
CITY_LOOKUP_URL = "https://geoapi.qweather.com/v2/city/lookup"
WEATHER_NOW_URL = "https://devapi.qweather.com/v7/weather/now"

def get_current_weather_qweather(location: str):
    """
    使用和风天气查询指定地点的实时天气（支持区县）
    支持：深圳市光明区、广州天河区、北京中关村等
    """

    # 第一步：查找 location ID
    lookup_params = {"location": location, "key": QWEATHER_API_KEY}
    try:
        logger.info(f"正在查询地点: {location}")
        resp_lookup = requests.get(CITY_LOOKUP_URL, params=lookup_params, timeout=5)
        resp_lookup.raise_for_status()
        data_lookup = resp_lookup.json()
        
        logger.info(f"地点查询响应: {data_lookup}")

        if data_lookup.get("location") and len(data_lookup["location"]) > 0:
            city_id = data_lookup["location"][0]["id"]
            logger.info(f"找到城市ID: {city_id}")
        else:
            logger.warning(f"未找到地点: {location}")
            return {
                "status": "0",
                "count": "0",
                "info": "未找到该地点",
                "infocode": "10001",
                "lives": []
            }

        # 第二步：根据 location ID 获取实时天气
        now_params = {"location": city_id, "key": QWEATHER_API_KEY}
        logger.info(f"正在获取天气数据，城市ID: {city_id}")
        resp_now = requests.get(WEATHER_NOW_URL, params=now_params, timeout=5)
        resp_now.raise_for_status()
        data_now = resp_now.json()
        
        logger.info(f"天气数据响应: {data_now}")

        if data_now.get("code") != "200":
            logger.error(f"天气API返回错误: {data_now.get('message', '未知错误')}")
            return {
                "status": "0",
                "count": "0",
                "info": f"天气数据异常: {data_now.get('message', '未知错误')}",
                "infocode": "10002",
                "lives": []
            }

        now_data = data_now.get("now", {})
        if not now_data:
            logger.error("天气数据为空")
            return {
                "status": "0",
                "count": "0",
                "info": "天气数据为空",
                "infocode": "10004",
                "lives": []
            }

        # 转换为与高德一致的格式（便于下游兼容）
        location_info = data_lookup["location"][0]
        return {
            "status": "1",
            "count": "1",
            "info": "OK",
            "infocode": "10000",
            "lives": [{
                "province": location_info.get("admin_area", location_info.get("country", "未知")),
                "city": location_info.get("city", location_info.get("name", "未知")),
                "adcode": city_id,
                "reporttime": data_now["updateTime"].replace("T", " ").split("+")[0],
                "temperature": now_data["temp"],
                "weather": now_data["text"],
                "winddirection": now_data["windDir"],
                "windpower": f"{now_data['windScale']}级"
            }]
        }

    except requests.exceptions.Timeout:
        logger.error("请求超时")
        return {
            "status": "0",
            "count": "0",
            "info": "请求超时，请稍后重试",
            "infocode": "10005",
            "lives": []
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求异常: {str(e)}")
        return {
            "status": "0",
            "count": "0",
            "info": f"网络请求异常: {str(e)}",
            "infocode": "10006",
            "lives": []
        }
    except KeyError as e:
        logger.error(f"数据字段缺失: {str(e)}")
        return {
            "status": "0",
            "count": "0",
            "info": f"数据字段缺失: {str(e)}",
            "infocode": "10007",
            "lives": []
        }
    except Exception as e:
        logger.error(f"未知异常: {str(e)}")
        return {
            "status": "0",
            "count": "0",
            "info": f"未知异常: {str(e)}",
            "infocode": "10003",
            "lives": []
        }


def get_weather_from_gaode(location: str, adcode: str = None):
    """调用高德地图API查询天气"""
    gaode_api_key = os.environ.get('GAODE_API_KEY', "f6397598fba9fd0641b6afc98da6d9da")  # 替换成你的高德API Key
    base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    
    params = {
        "key": gaode_api_key,
        "city": adcode if adcode else location,
        "extensions": "base",  # 可改为 "all" 获取预报
    }
    
    try:
        logger.info(f"正在查询高德天气: {location}")
        response = requests.get(base_url, params=params, timeout=5)
        logger.info(f"高德天气查询返回: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "1":
                return data
            else:
                logger.error(f"高德API返回错误: {data.get('info', '未知错误')}")
                return {"error": f"API错误: {data.get('info', '未知错误')}"}
        else:
            logger.error(f"HTTP请求失败: {response.status_code}")
            return {"error": f"HTTP请求失败: {response.status_code}"}
    except Exception as e:
        logger.error(f"高德天气查询异常: {str(e)}")
        return {"error": f"查询异常: {str(e)}"}

def run_weather_query():
    """使用 Qwen3 + 查询天气"""
    try:
        messages = [
            {"role": "system", "content": "你是一个智能助手，可以查询天气信息。"},
            {"role": "user", "content": "深圳市现在天气怎么样？"}
        ]
        
        logger.info("正在调用Qwen模型...")
        response = dashscope.Generation.call(
            model="qwen-turbo",  # 可使用 Qwen3 最新版本
            messages=messages,
            tools=[weather_tool_qweather],  # 传入工具定义
            tool_choice="auto",  # 让模型决定是否调用工具
        )

        logger.info(f"Qwen模型响应: {response}")
        
        if response.status_code == HTTPStatus.OK:
            # 检查是否需要调用工具
            if "tool_calls" in response.output.choices[0].message:
                tool_call = response.output.choices[0].message.tool_calls[0]
                function_name = tool_call["function"]["name"]
                logger.info(f"调用工具: {function_name}")
                
                if function_name == "get_current_weather":
                    # 解析参数并调用高德API
                    args = json.loads(tool_call["function"]["arguments"])
                    location = args.get("location", "北京")
                    adcode = args.get("adcode", None)
                    logger.info(f"高德天气查询地点: {location}")
                    weather_data = get_weather_from_gaode(location, adcode)
                    logger.info(f"高德天气查询结果: {weather_data}")
                    print(f"查询结果：{weather_data}")
                    
                elif function_name == "get_current_weather_qweather":
                    # 解析参数并调用和风天气API
                    args = json.loads(tool_call["function"]["arguments"])
                    location = args.get("location", "北京")
                    logger.info(f"和风天气查询地点: {location}")
                    weather_data = get_current_weather_qweather(location)
                    logger.info(f"和风天气查询结果: {weather_data}")
                    print(f"查询结果：{weather_data}")
                else:
                    logger.warning(f"未知的工具调用: {function_name}")
            else:
                logger.info("模型直接回复，无需调用工具")
                print(response.output.choices[0].message.content)
        else:
            logger.error(f"Qwen模型请求失败: {response.code} - {response.message}")
            print(f"请求失败: {response.code} - {response.message}")
            
    except Exception as e:
        logger.error(f"运行天气查询时发生异常: {str(e)}")
        print(f"运行异常: {str(e)}")

def run_weather_query_with_tool_choice():
    """使用 Qwen3 + 查询天气，并让大模型输出最终结果"""
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以查询天气信息。"},
        {"role": "user", "content": "深圳市光明区现在天气怎么样？"}
    ]
    
    print("第一次调用大模型...")
    response = dashscope.Generation.call(
        model="qwen-turbo",  # 可使用 Qwen3 最新版本
        messages=messages,
        tools=[weather_tool],  # 传入工具定义
        tool_choice="auto",  # 让模型决定是否调用工具
    )
    
    if response.status_code == HTTPStatus.OK:
        tool_map = {
            "get_current_weather": get_current_weather_qweather,
            # 如有更多工具，在此添加
        }
        
        # 从响应中获取消息
        assistant_message = response.output.choices[0].message
        
        # 检查是否需要调用工具
        if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
            print("检测到工具调用...")
            
            # 转换 assistant 消息为标准字典格式
            assistant_dict = {
                "role": "assistant",
                "content": assistant_message.content if hasattr(assistant_message, "content") else None
            }
            
            # 添加 tool_calls 到 assistant 消息
            if hasattr(assistant_message, "tool_calls"):
                assistant_dict["tool_calls"] = assistant_message.tool_calls
                
                # 生成工具调用回复消息
                tool_response_messages = []
                import json
                for tool_call in assistant_message.tool_calls:
                    print(f"处理工具调用: {tool_call['function']['name']}, ID: {tool_call['id']}")
                    
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])
                    
                    if func_name in tool_map:
                        # 调用工具函数
                        from inspect import signature
                        sig = signature(tool_map[func_name])
                        valid_args = {k: v for k, v in func_args.items() if k in sig.parameters}
                        result = tool_map[func_name](**valid_args)
                        
                        # 创建工具回复消息
                        tool_response = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": func_name,
                            "content": json.dumps(result, ensure_ascii=False)
                        }
                        tool_response_messages.append(tool_response)
                
                # 组装完整消息列表
                updated_messages = messages + [assistant_dict] + tool_response_messages
                
                print(f"完整消息列表: {updated_messages}")
                
                # 第二次调用大模型
                print("第二次调用大模型...")
                response2 = dashscope.Generation.call(
                    model="qwen-turbo",
                    messages=updated_messages,
                    tools=[weather_tool],
                    tool_choice="auto",
                )
                
                if response2.status_code == HTTPStatus.OK:
                    final_response = response2.output.choices[0].message.content
                    print("最终回复:", final_response)
                else:
                    print(f"请求失败: {response2.code} - {response2.message}")
            else:
                print("assistant 消息中没有 tool_calls 字段")
                print(assistant_message)
        else:
            # 如果没有调用工具，直接输出模型回复
            print("无工具调用，直接输出回复:", assistant_message.content)
    else:
        print(f"请求失败: {response.code} - {response.message}")

if __name__ == "__main__":
    #run_weather_query()
    run_weather_query_with_tool_choice()