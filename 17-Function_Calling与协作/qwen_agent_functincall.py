
import os
import requests
from typing import Optional
import dashscope
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv
import logging
import httpx
load_dotenv()
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

# 清除所有可能污染的代理环境变量
PROXY_ENV_VARS = [
    'http_proxy', 'https_proxy',
    'HTTP_PROXY', 'HTTPS_PROXY',
    'all_proxy', 'ALL_PROXY',
    'socks_proxy', 'SOCKS_PROXY'
]

for var in PROXY_ENV_VARS:
    if var in os.environ:
        print(f"⚠️ 清除代理环境变量: {var} = {os.environ[var]}")
        del os.environ[var]

# 强制 httpx 不使用任何代理
httpx._config.DEFAULT_PROXIES = None

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI  # 暂时注释掉GUI组件以避免依赖问题

# 配置（请替换成你自己的 API Key）
QWEATHER_API_KEY = os.environ.get('QWEATHER_API_KEY', "687ed20ce0ce4567b032fde58a5938a8")

# 和风天气接口地址（注意：无空格！）
CITY_LOOKUP_URL = "https://geoapi.qweather.com/v2/city/lookup"
WEATHER_NOW_URL = "https://devapi.qweather.com/v7/weather/now"

# 移除复杂的functions_desc定义，使用简化的工具注册

# ====== 天气查询工具实现 ======
@register_tool('get_current_weather')
class WeatherTool(BaseTool):
    """
    天气查询工具，通过和风天气API查询指定位置的天气情况。
    """
    description = '获取指定位置的当前天气情况'
    parameters = [{
        'name': 'location',
        'type': 'string',
        'description': '城市名称，例如：北京',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        location = args['location']
        
        return self.get_current_weather_qweather(location)
    
    def get_current_weather_qweather(self, location: str):
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


# ====== 初始化助手服务 ======
def init_agent_service():
    """初始化助手服务"""
    try:
        bot = Assistant(
            name='天气助手',
            description='天气助手，查询天气'
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise


def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                # 获取可选的文件输入
                file = input('file url (press enter if no file): ').strip()
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型门票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '北京今天的天气怎么样？',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_tui()          # 图形界面模式（默认）