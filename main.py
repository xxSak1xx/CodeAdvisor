# main.py

import sys
from dotenv import load_dotenv
from src.interface import interactive_setup
from src.workflow import run_code_advisor_workflow

def main():
    """
    Main entry point for CodeAdvisor.
    Provides an interactive setup and runs the workflow.
    """
    # Load environment variables from .env file if it exists
    load_dotenv() # Load from root
    load_dotenv("config/.env") # Load from config dir
    try:
        # 1. Start interactive setup
        config = interactive_setup()

        # 2. Ask if the user wants to run the analysis now
        prompt = input("配置已就绪。是否立即开始分析项目? (Y/n, 默认 Y): ").strip().lower()
        if prompt == 'n':
            print("分析已取消。您可以随时运行此脚本来重新配置或启动分析。")
            return
        
        print("\n" + "="*50)
        print("      🚀 正在启动分析流程...")
        print("="*50 + "\n")

        # 3. Run the workflow with the saved/updated config
        run_code_advisor_workflow(
            input_dir=config.get('input_dir', '.'),
            ignore_patterns=config.get('ignore_dir', [".git", "node_modules", ".venv"]),
            output_dir=config.get('output_dir', './output'),
            model_summarizer=config.get('model1', 'gpt-3.5-turbo'),
            model_advisor=config.get('model2', 'gpt-4')
        )

    except KeyboardInterrupt:
        print("\n操作已取消。")
        sys.exit(0)
    except Exception as e:
        print(f"\n[发生错误] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
