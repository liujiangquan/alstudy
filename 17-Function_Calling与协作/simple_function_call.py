#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import httpx

# ========================
# 🔧 修复代理问题 - 在任何 GUI 导入前执行
# ========================

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

print("✅ 已清除所有代理设置，确保 Gradio 可正常启动")

# ========================
# ✅ 安全导入 qwen_agent.gui
# ========================
from qwen_agent.gui import WebUI
from qwen_agent.agents import Assistant

# 你的业务逻辑
def main():
    agent = Assistant(
        llm=None,  # 使用默认模型
        tools=[],
    )

    web_ui = WebUI(agent)
    web_ui.launch()

if __name__ == "__main__":
    main()