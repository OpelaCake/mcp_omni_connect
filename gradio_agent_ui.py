import gradio as gr
import asyncio
from typing import Optional, List
import time
import threading
from src.mcpomni_connect.cli import MCPClientCLI
from src.mcpomni_connect.cli import CommandType

from src.mcpomni_connect.client import Configuration, MCPClient
from src.mcpomni_connect.llm import LLMConnection
from src.mcpomni_connect.main import check_config_exists

class GradioAgentUI:
    def __init__(self):
        self.agent_cli = None
        self.is_running = False
        self.auto_run_interval = 5  # 默认自动运行间隔为5秒
        self.auto_run_task = None  # 添加任务引用
        self.history: List[str] = []  # 添加历史记录列表
        self.max_history = 5  # 最大历史记录数量
        self.refresh_timer = None  # 刷新定时器
        
    async def initialize(self):
        config_path = check_config_exists()
        config = Configuration()
        self.client = MCPClient(config, debug=True)
        llm_connection = LLMConnection(config)
        self.agent_cli = MCPClientCLI(self.client, llm_connection)
        return self
    
    def add_to_history(self, result: str):
        """添加结果到历史记录"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_result = f"[{timestamp}] {result}"
        self.history.append(formatted_result)
        
        # 保持最新的5个记录
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history_display(self) -> str:
        """获取历史记录显示文本"""
        if not self.history:
            return "暂无历史记录"
        return "\n\n".join(self.history)
    
    async def process_command(self, command: str) -> str:
        """处理用户输入的命令"""
        try:
            await self.client.connect_to_servers()
            handlers = {
            CommandType.DEBUG: self.agent_cli.handle_debug_command,
            CommandType.REFRESH: self.agent_cli.handle_refresh_command,
            CommandType.HELP: self.agent_cli.handle_help_command,
            CommandType.TOOLS: self.agent_cli.handle_tools_command,
            CommandType.RESOURCES: self.agent_cli.handle_resources_command,
            CommandType.RESOURCE: self.agent_cli.handle_resource_command,
            CommandType.LOAD_PICTURE: self.agent_cli.handle_load_picture_command,
            CommandType.QUERY: self.agent_cli.handle_query,
            CommandType.PROMPTS: self.agent_cli.handle_prompts_command,
            CommandType.PROMPT: self.agent_cli.handle_prompt_command,
            CommandType.HISTORY: self.agent_cli.handle_history_command,
            CommandType.CLEAR_HISTORY: self.agent_cli.handle_clear_history_command,
            CommandType.SAVE_HISTORY: self.agent_cli.handle_save_history_command,
            CommandType.SUBSCRIBE: self.agent_cli.handle_subscribe,
            CommandType.UNSUBSCRIBE: self.agent_cli.handle_unsubscribe,
            CommandType.MEMORY: self.agent_cli.handle_memory_command,
            CommandType.MODE: self.agent_cli.handle_mode_command,
            CommandType.LOAD_HISTORY: self.agent_cli.handle_load_history_command,
            }
            command_type, payload = self.agent_cli.parse_command(command)
            if command_type == CommandType.QUIT:
                return "退出程序"
            handler = handlers[command_type]
            print(f"command_type: {command_type}, payload: {payload}, handler: {handler}")
            if handler:
                response = await handler(payload)
                print(f"response: {response}")
                return response
            else:
                return f"未知命令: {command}"
        except Exception as e:
            return f"错误: {str(e)}"
    
    async def auto_run_loop(self):
        """自动运行循环"""
        while self.is_running:
            try:
                # 这里可以设置自动执行的命令
                result = await self.process_command("目前处于自动运行状态，默认用户确认所有操作，请分析当前状态并进行下一步操作")
                print(f"自动运行结果: {result}")
                
                # 添加到历史记录
                self.add_to_history(result)
                
                await asyncio.sleep(self.auto_run_interval)
            except Exception as e:
                error_msg = f"自动运行出错: {str(e)}"
                print(error_msg)
                self.add_to_history(error_msg)
                self.is_running = False
                break
    
    def create_ui(self):
        """创建Gradio界面"""
        with gr.Blocks(title="Agent交互界面") as demo:
            gr.Markdown("# Agent交互界面")
            
            with gr.Row():
                with gr.Column():
                    command_input = gr.Textbox(
                        label="输入命令",
                        placeholder="输入命令，例如: query 你好",
                        lines=2
                    )
                    submit_btn = gr.Button("提交")
                    
                with gr.Column():
                    output = gr.Textbox(
                        label="输出结果 (最新5条记录)",
                        lines=15,
                        interactive=False,
                        value="暂无历史记录"
                    )
            
            with gr.Row():
                auto_run_btn = gr.Button("开始自动运行")
                stop_btn = gr.Button("停止自动运行")
                clear_history_btn = gr.Button("清空历史记录")
                refresh_btn = gr.Button("刷新显示")
                interval_slider = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=5,
                    step=1,
                    label="自动运行间隔(秒)"
                )
            
            # 事件处理
            async def submit_command(command):
                result = await self.process_command(command)
                self.add_to_history(result)
                return self.get_history_display()
            
            async def start_auto_run():
                if not self.is_running:
                    self.is_running = True
                    self.auto_run_interval = interval_slider.value
                    
                    # 启动后台任务
                    self.auto_run_task = asyncio.create_task(self.auto_run_loop())
                    return "开始自动运行"
                else:
                    return "自动运行已在运行中"
            
            async def stop_auto_run():
                if self.is_running:
                    self.is_running = False
                    if self.auto_run_task:
                        self.auto_run_task.cancel()
                        try:
                            await self.auto_run_task
                        except asyncio.CancelledError:
                            pass
                    return "停止自动运行"
                else:
                    return "自动运行未在运行"
            
            def clear_history():
                self.history.clear()
                return "历史记录已清空"
            
            def refresh_display():
                return self.get_history_display()
            
            submit_btn.click(
                fn=submit_command,
                inputs=[command_input],
                outputs=[output]
            )
            
            auto_run_btn.click(
                fn=start_auto_run,
                outputs=[output]
            )
            
            stop_btn.click(
                fn=stop_auto_run,
                outputs=[output]
            )
            
            clear_history_btn.click(
                fn=clear_history,
                outputs=[output]
            )
            
            refresh_btn.click(
                fn=refresh_display,
                outputs=[output]
            )
        
        return demo

async def main():
    ui = GradioAgentUI()
    await ui.initialize()
    demo = ui.create_ui()
    demo.launch(share=True)

if __name__ == "__main__":
    asyncio.run(main()) 