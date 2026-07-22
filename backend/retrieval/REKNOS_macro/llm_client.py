# -*- coding: utf-8 -*-
"""
本地 LLM 调用封装。

直接继承原 REKNOS 项目 utils.py 中 run_llm() 的写法与思路：
通过 OpenAI 兼容接口访问本地 Ollama 服务来调用 Phi-3。
区别：
- 原版发生异常时无限重试（while f==0），这里改为有限次重试后抛出异常，
  避免在离线 / 未启动 Ollama 的环境下卡死，便于单元测试用 mock 替换。
"""

import time

import config


class LLMError(RuntimeError):
    pass


def run_llm(prompt: str,
            temperature: float = config.LLM_TEMPERATURE,
            max_tokens: int = config.LLM_MAX_TOKENS,
            engine: str = config.LLM_TYPE,
            max_retry: int = config.LLM_MAX_RETRY) -> str:
    """
    调用本地 Ollama 部署的 Phi-3。
    与原版 run_llm 签名保持相似风格，方便熟悉原项目的人直接迁移调用方式。

    注意：openai 包在函数内部懒加载，这样只用 mock LLM 做离线结构测试
    （见 test_pipeline.py）时，不需要安装/联网即可运行。
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError("未安装 openai 包，请先 `pip install -r requirements.txt`。") from e

    client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.OLLAMA_API_KEY)

    messages = [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": prompt},
    ]

    last_err = None
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - 与原版一致，捕获所有异常后重试
            last_err = e
            print(f"LLM Error: {e}, retrying ({attempt + 1}/{max_retry}) ...")
            time.sleep(1)

    raise LLMError(
        f"调用本地 Ollama / Phi-3 失败，请确认已执行 `ollama run {engine}` 且服务地址为 "
        f"{config.OLLAMA_BASE_URL}。原始错误: {last_err}"
    )
