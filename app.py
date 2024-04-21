import os
from typing import Any, Generator, Tuple

import gradio as gr
import lmdeploy
from lmdeploy import (ChatTemplateConfig, GenerationConfig,
                      TurbomindEngineConfig, pipeline)

system_prompt = '''你是非常棒的私人助手，请尽量严谨的回答问题，不知道的直接回复不知道不要乱说。'''

def print_info():
    '''打印提示信息'''

    print('lmdeploy version: ', lmdeploy.__version__, 'gradio version: ', gr.__version__)
    print('system_prompt: ', system_prompt)

def prepare_model() -> str:
    '''
    准备模型

    Returns:
        模型存放路径
    '''

    local_model_path = '/root/ft/final_model'
    if os.path.exists(local_model_path):
        return local_model_path
    else:
        if not os.path.exists(download_model_path):
            download_model_path = '/home/xlab-app-center/internlm2-chat-1-8b'
            remote_repo_url = 'https://code.openxlab.org.cn/csg2008/internlm2_chat_1_8b_demo.git'
            os.system(f'git clone {remote_repo_url} {download_model_path}')
            os.system(f'cd {download_model_path} && git lfs pull')

        return download_model_path

def get_model_pipeline(model_path: str):
    '''
    加载模型并返回模型的pipeline

    Args:
        base_path: 模型存放路径

    Returns:
        pipeline
    '''
    backend_config = TurbomindEngineConfig(
        model_name = 'internlm2',
        model_format = 'hf',
        cache_max_entry_count = 0.8,
    )

    chat_template_config = ChatTemplateConfig(
        model_name = 'internlm2',
        system = None,
        meta_instruction = system_prompt,
    )

    return pipeline(
        model_path = model_path,
        model_name = 'internlm2_chat_1_8b',
        backend_config = backend_config,
        chat_template_config = chat_template_config,
    )


pipe = get_model_pipeline(prepare_model())

def stream_chat(
    query: str,
    history: list = [],
    max_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
    temperature: float = 0.8,
    regenerate: str = '',
) -> Generator[Any, Any, Any]:
    '''流式聊天'''

    if regenerate:
        if len(history) > 0:
            query, _ = history.pop(-1)
        else:
            yield history
            return
    else:
        query = query.strip()
        if query is None or len(query) < 1:
            yield history
            return

    # 转换历史消息格式及添加最新消息
    prompts = []
    for user, assistant in history:
        prompts.append(
            {
                'role': 'user',
                'content': user
            }
        )
        prompts.append(
            {
                'role': 'assistant',
                'content': assistant
            })
    prompts.append(
        {
            'role': 'user',
            'content': query
        }
    )

    gen_config = GenerationConfig(
        n = 1,
        max_new_tokens = max_tokens,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature
    )

    # 实现流式输出的打字机效果
    resp_text = ''
    for response in pipe.stream_infer(prompts = prompts, gen_config = gen_config):
        resp_text += response.text

        yield history + [[query, resp_text]]


def undo_history(history: list = []) -> Tuple[str, list]:
    '''
    恢复到上一轮对话

    Args:
        history: 历史对话
    Returns:
        query: 最后一次对话消息
        history: 历史对话
    '''

    query = ''
    if len(history) > 0:
        query, _ = history.pop(-1)

    return query, history

def clean_history() -> list:
    '''
    清空历史对话

    Returns:
        history: 历史对话
    '''

    history = []

    return history

def main():
    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown('''## 书生浦语大模型 LMDeploy Web 部署实验''')

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(show_copy_button=True, label='基于 InternLM2-Chat-1.8B 模型')

                with gr.Row():
                    max_new_tokens = gr.Slider(
                        minimum=1,
                        maximum=2048,
                        value=1024,
                        step=1,
                        label='Maximum new tokens'
                    )
                    top_p = gr.Slider(
                        minimum=0.01,
                        maximum=1,
                        value=0.8,
                        step=0.01,
                        label='Top_p'
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=40,
                        step=1,
                        label='Top_k'
                    )
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.5,
                        value=0.8,
                        step=0.01,
                        label='Temperature'
                    )

                with gr.Row():
                    query = gr.Textbox(label = '聊天消息', placeholder='请输入聊天消息，按 Ctrl+Enter 发送')
                    submit = gr.Button('发送', variant='primary', scale=0)
                    # stop = gr.Button('停止', variant='secondary')

                with gr.Row():
                    regenerate = gr.Button('重新生成', variant='secondary')
                    undo = gr.Button('撤销', variant='secondary')
                    clear = gr.Button('清空', variant='secondary')

            # 回车提交
            query.submit(
                stream_chat,
                inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature],
                outputs=[chatbot]
            )

            # 按钮提交
            submit.click(
                stream_chat,
                inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature],
                outputs=[chatbot]
            )

            # 重新生成
            regenerate.click(
                stream_chat,
                inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature, regenerate],
                outputs=[chatbot]
            )

            # 撤销消息
            undo.click(
                undo_history,
                inputs=[chatbot],
                outputs=[query, chatbot]
            )

            # 清空消息
            clear.click(
                clean_history,
                [],
                outputs=[chatbot]
            )

        gr.Markdown('''温馨提示：请文明聊天，共创文明网络环境''')

    gr.close_all()

    demo.queue(max_size=100).launch(share=True, server_port=7860)


if __name__ == '__main__':
    main()
