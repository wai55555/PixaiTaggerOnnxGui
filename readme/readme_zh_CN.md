# PixAI Tagger ONNX GUI
此应用程序是一个GUI工具，用于在本地环境中对大量图像自动分配快速准确的标签。通过直观的操作，它极大地简化了数据集的组织和管理。

|![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/main_window_01.png)|![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/main_window_02.jpg)|
|:-:|:-:|

![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/gridview_01.jpg)

## 概述 (Overview)

**PixAI Tagger ONNX GUI** 利用 [PixAI](https://pixai.art/) 开发的图像标注模型的 [ONNX 版本](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) 为本地图像分配标签。

原始的 PixAI Tagger 支持超过 13,000 个丰富的标签，与常见的标注模型（例如 wd-tagger 的约 10,000 个标签）相比具有优势。此工具旨在最大限度地发挥其性能并支持您的图像管理。

它具有自动图像标注、标签浏览、单独编辑以及强大的批量编辑功能（添加/删除特定标签）。

## 安装与使用 (Installation & Usage)

1.  从发布页面下载最新的 `PixaiTaggerOnnxGui-vX.X.X.zip`。
2.  将 zip 文件解压到您喜欢的位置。
3.  运行其中的 `pixai_tagger_gui.exe`。

## 如何使用 (How to Use)
双击图像并使用滚轮或拖动来移动它。可以进行直观的操作。
在显示 3x3 时，双击、滚轮和拖动也有效。Ctrl+滚轮可以前后移动图像。

1.  **指定文件夹**: 点击 `浏览` 按钮或拖放图像文件夹来加载目标图像组。
2.  **准备模型 (仅限首次)**: 如果显示 `开始下载` 而不是 `TAG` 按钮，请点击它以下载模型。
3.  **执行标注**: 按下 `TAG` 按钮开始对文件夹中的所有图像进行标注处理。如果找到现有的 `.txt` 文件，将出现对话框确认是否覆盖。
4.  **检查和编辑结果**:
    -   从左侧列表中选择图像，在中心显示图像，右侧显示其标签。在标签输入字段中，您可以使用 Ctrl+上下键前后导航图像。
    -   通过点击右侧的标签按钮删除不需要的标签。
    -   底部 `批量标签` 部分允许在整个文件夹中添加/删除标签。

## 主要功能 (Features)

### 1. 直观舒适的用户界面
-   **拖放支持**: 图像文件夹、单个图像文件，甚至包含标签的 `.txt` 文件都可以通过直接拖放到窗口中立即加载。
-   **轻量级图像查看器**:
    -   只需从列表中选择图像即可轻松操作。
    -   双击图像可显示可自由调整大小和移动的放大窗口。
    -   使用鼠标滚轮或键盘（箭头键、WASD 等）快速切换图像进行查看。

### 2. 强大的自动标注
-   **高速 ONNX Runtime**: 采用 ONNX 模型，即使在 CPU 上也能流畅运行。轻松处理大量图像。
-   **自动模型下载**: 首次启动时，只需单击一个按钮，即可从 Hugging Face 自动下载 PixAI Tagger 模型和标签文件。无需繁琐的手动操作。

### 3. 灵活高级的标签编辑
-   **单独编辑**:
    -   图像的现有标签以按钮形式显示。只需单击按钮即可删除不需要的标签。
    -   可以轻松添加新标签（允许使用逗号分隔的多个标签）。
-   **强大的批量编辑**:
    -   聚合文件夹中所有 `.txt` 文件中的标签，并按频率顺序显示。
    -   只需单击一个按钮，即可**从所有文件中批量删除特定标签**。
    -   还可以**将指定标签批量添加到所有文件的开头或结尾**。
-   **网格编辑视图**:
    -   从 `3x3` 按钮切换到网格视图。同时查看多张图像，确认和编辑标签。

### 4. 详细自定义
-   **标签生成调整**: 使用滑块直观地调整 `general` 和 `character` 类别的标签生成**阈值**和**最大标签数**。
-   **自动设置保存**: 窗口大小和所有设置在应用程序退出时自动保存到 `config.ini`。下次在相同的环境中继续工作。
-   **多语言支持**: 支持日语和英语。用户界面根据操作系统语言设置自动切换。

## 许可证 (License)

本项目根据 **LGPLv3** 和 **Apache License 2.0** 许可证发布。

## 致谢 (Acknowledgements)

- 此工具利用了 [PixAI](https://pixai.art/) 训练的优秀标注模型。如果没有 Pixai Tagger 的公开发布，此应用程序将无法诞生。衷心感谢。
- 使用的 ONNX 模型由 [deepghs](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) 在 Hugging Face 上公开。谢谢。
