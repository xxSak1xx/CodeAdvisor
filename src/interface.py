# src/interface.py

import os
from typing import Dict, Any
from src.configls import load_config, save_config

def interactive_setup() -> Dict[str, Any]:
    """
    Guides the user through setting up the configuration interactively.
    """
    print("\n" + "="*50)
    print("      CodeAdvisor - 交互式配置引导")
    print("="*50 + "\n")

    # Load existing config for defaults
    config = load_config()

    # --- 1. 项目路径配置 (无论是否 Reindex 都需要) ---
    print(f"1. [项目分析路径] 当前: {config.get('input_dir', '.')}")
    input_dir = input("请输入要分析的项目路径 (直接回车保持默认): ").strip()
    if input_dir:
        config['input_dir'] = input_dir

    print(f"\n2. [分析结果输出路径] 当前: {config.get('output_dir', './output')}")
    output_dir = input("请输入输出路径 (直接回车保持默认): ").strip()
    if output_dir:
        config['output_dir'] = output_dir

    # --- 2. 向量处理配置 ---
    print(f"\n3. [向量处理] 是否需要重新扫描代码并构建向量库?")
    print("   (注: 初次运行或代码有变动时选 'y', 仅更换模型分析选 'n')")
    reindex_input = input("请输入 (y/n, 默认 n): ").strip().lower()
    config['reindex'] = (reindex_input == 'y')

    if config['reindex']:
        # 3.1 Ignore Path
        current_ignore = config.get('ignore_dir', [".git", "node_modules", ".venv"])
        print(f"\n3.1 [忽略路径] 当前: {', '.join(current_ignore)}")
        ignore_input = input("请输入要忽略的路径 (多个请用逗号分隔，直接回车保持默认): ").strip()
        if ignore_input:
            config['ignore_dir'] = [i.strip() for i in ignore_input.split(",")]

        # 3.2 Embeddings Config
        print(f"\n3.2 [Embeddings 配置] (注: DeepSeek 不支持 Embeddings，建议使用 OpenAI 或默认)")
        
        # 3.2.1 Embeddings Model Name
        print(f"3.2.1 [Embeddings 模型] 当前: {config.get('model_embed', 'text-embedding-3-small')}")
        model_embed = input("请输入 Embeddings 模型名称 (直接回车保持默认): ").strip()
        if model_embed:
            config['model_embed'] = model_embed

        print(f"3.2.2 [Embeddings Base URL] 当前: {config.get('base_url_embed', '默认 (OpenAI)')}")
        base_url_embed = input("请输入 Embeddings API Base URL (直接回车保持默认): ").strip()
        if base_url_embed:
            config['base_url_embed'] = base_url_embed
        
        key_embed = os.getenv("OPENAI_API_KEY_EMBED", "")
        print(f"3.2.3 [Embeddings API Key] {'已设置' if key_embed else '未设置'}")
        print("   (注: 如果使用 DashScope，请确保此 Key 是阿里云的 API Key)")
        new_key_embed = input("请输入 API Key (直接回车跳过/保持): ").strip()
        if new_key_embed:
            os.environ["OPENAI_API_KEY_EMBED"] = new_key_embed
    else:
        print("\nℹ️  跳过扫描，将尝试加载现有向量库。")
        # 即使不 Reindex，检索也需要 Embeddings API Key 将问题向量化
        key_embed = os.getenv("OPENAI_API_KEY_EMBED", "")
        if not key_embed:
            print("⚠️  警告: 未检测到 OPENAI_API_KEY_EMBED。检索现有向量库仍需此 Key。")
            new_key_embed = input("请输入 OPENAI_API_KEY_EMBED (直接回车跳过): ").strip()
            if new_key_embed:
                os.environ["OPENAI_API_KEY_EMBED"] = new_key_embed

    # --- 3. 模型配置 ---
    # 4. Summarizer Model
    print(f"\n4. [Summarizer 模型] 当前: {config.get('model1', 'gpt-3.5-turbo')}")
    model1 = input("请输入 Summarizer 模型名称 (直接回车保持默认): ").strip()
    if model1:
        config['model1'] = model1

    # 4.1 Summarizer Base URL
    print(f"4.1 [Summarizer Base URL] 当前: {config.get('base_url1', '默认 (OpenAI)')}")
    base_url1 = input("请输入 API Base URL (直接回车保持默认): ").strip()
    if base_url1:
        config['base_url1'] = base_url1

    # 4.2 OPENAI_API_KEY1
    key1 = os.getenv("OPENAI_API_KEY1", "")
    print(f"4.2 [API Key 1] {'已设置' if key1 else '未设置'}")
    new_key1 = input("请输入 OPENAI_API_KEY1 (直接回车保持): ").strip()
    if new_key1:
        os.environ["OPENAI_API_KEY1"] = new_key1

    # 5. Advisor Model
    print(f"\n5. [Advisor 模型] 当前: {config.get('model2', 'gpt-4')}")
    model2 = input("请输入 Advisor 模型名称 (直接回车保持默认): ").strip()
    if model2:
        config['model2'] = model2

    # 5.1 Advisor Base URL
    print(f"5.1 [Advisor Base URL] 当前: {config.get('base_url2', '默认 (OpenAI)')}")
    base_url2 = input("请输入 API Base URL (直接回车保持默认): ").strip()
    if base_url2:
        config['base_url2'] = base_url2

    # 5.2 OPENAI_API_KEY2
    key2 = os.getenv("OPENAI_API_KEY2", "")
    print(f"5.2 [API Key 2] {'已设置' if key2 else '未设置'}")
    new_key2 = input("请输入 OPENAI_API_KEY2 (直接回车保持): ").strip()
    if new_key2:
        os.environ["OPENAI_API_KEY2"] = new_key2

    # Save to dir.config.yaml
    save_config(config)
    print("\n" + "-"*50)
    print("配置已保存到 config/dir.config.yaml")
    print("-"*50 + "\n")

    return config
