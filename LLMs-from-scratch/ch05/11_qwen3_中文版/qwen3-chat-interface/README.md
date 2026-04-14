# Qwen3 从零实现与聊天界面



本补充文件夹包含运行类似 ChatGPT 的用户界面来与预训练 Qwen3 模型交互的代码。



![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



为了实现这个用户界面，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 第 1 步：安装依赖

首先，我们通过以下方式从 [requirements-extra.txt](requirements-extra.txt) 列表安装 `chainlit` 包和依赖：

```bash
pip install -r requirements-extra.txt
```

或者，如果您使用 `uv`：

```bash
uv pip install -r requirements-extra.txt
```



&nbsp;

## 第 2 步：运行 `app` 代码

本文件夹包含 2 个文件：

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py)：此文件加载并使用 Qwen3 0.6B 模型的思考模式。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：与上述相同，但配置为记住消息历史。

（打开并查看这些文件以了解更多。）

从终端运行以下命令之一来启动 UI 服务器：

```bash
chainlit run qwen3-chat-interface.py
```

或者，如果您使用 `uv`：

```bash
uv run chainlit run qwen3-chat-interface.py
```

运行上述命令之一应该会打开一个新的浏览器标签页，您可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端命令并将本地地址复制到浏览器地址栏中（通常地址为 `http://localhost:8000`）。
