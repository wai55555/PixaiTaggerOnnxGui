# PixAI Tagger ONNX GUI
此應用程式是一個GUI工具，用於在本地環境中對大量圖像自動分配快速準確的標籤。透過直觀的操作，它極大地簡化了資料集的組織和管理。

|![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/main_window_01.png)|![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/main_window_02.jpg)|
|:-:|:-:|

![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/gridview_01.jpg)

## 概述 (Overview)

**PixAI Tagger ONNX GUI** 利用 [PixAI](https://pixai.art/) 開發的圖像標註模型的 [ONNX 版本](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) 為本地圖像分配標籤。

原始的 PixAI Tagger 支援超過 13,000 個豐富的標籤，與常見的標註模型（例如 wd-tagger 的約 10,000 個標籤）相比具有優勢。此工具旨在最大限度地發揮其性能並支持您的圖像管理。

它具有自動圖像標註、標籤瀏覽、單獨編輯以及強大的批量編輯功能（添加/刪除特定標籤）。

## 安裝與使用 (Installation & Usage)

1.  從發布頁面下載最新的 `PixaiTaggerOnnxGui-vX.X.X.zip`。
2.  將 zip 文件解壓縮到您喜歡的位置。
3.  運行其中的 `pixai_tagger_gui.exe`。

## 如何使用 (How to Use)
雙擊圖像並使用滾輪或拖動來移動它。可以進行直觀的操作。
在顯示 3x3 時，雙擊、滾輪和拖動也有效。Ctrl+滾輪可以前後移動圖像。

1.  **指定資料夾**: 點擊 `瀏覽` 按鈕或拖放圖像資料夾來載入目標圖像組。
2.  **準備模型 (僅限首次)**: 如果顯示 `開始下載` 而不是 `TAG` 按鈕，請點擊它以下載模型。
3.  **執行標註**: 按下 `TAG` 按鈕開始對資料夾中的所有圖像進行標註處理。如果找到現有的 `.txt` 文件，將出現對話框確認是否覆蓋。
4.  **檢查和編輯結果**:
    -   從左側列表中選擇圖像，在中心顯示圖像，右側顯示其標籤。在標籤輸入欄位中，您可以使用 Ctrl+上下鍵前後導航圖像。
    -   透過點擊右側的標籤按鈕刪除不需要的標籤。
    -   底部 `批量標籤` 部分允許在整個資料夾中添加/刪除標籤。

## 主要功能 (Features)

### 1. 直觀舒適的使用者介面
-   **拖放支援**: 圖像資料夾、單個圖像文件，甚至包含標籤的 `.txt` 文件都可以透過直接拖放到視窗中立即載入。
-   **輕量級圖像檢視器**:
    -   只需從列表中選擇圖像即可輕鬆操作。
    -   雙擊圖像可顯示可自由調整大小和移動的放大視窗。
    -   使用滑鼠滾輪或鍵盤（箭頭鍵、WASD 等）快速切換圖像進行查看。

### 2. 強大的自動標註
-   **高速 ONNX Runtime**: 採用 ONNX 模型，即使在 CPU 上也能流暢運行。輕鬆處理大量圖像。
-   **自動模型下載**: 首次啟動時，只需點擊一個按鈕，即可從 Hugging Face 自動下載 PixAI Tagger 模型和標籤文件。無需繁瑣的手動操作。

### 3. 靈活高級的標籤編輯
-   **單獨編輯**:
    -   圖像的現有標籤以按鈕形式顯示。只需點擊按鈕即可刪除不需要的標籤。
    -   可以輕鬆添加新標籤（允許使用逗號分隔的多個標籤）。
-   **強大的批量編輯**:
    -   聚合資料夾中所有 `.txt` 文件中的標籤，並按頻率順序顯示。
    -   只需點擊一個按鈕，即可**從所有文件中批量刪除特定標籤**。
    -   還可以**將指定標籤批量添加到所有文件的開頭或結尾**。
-   **網格編輯檢視**:
    -   從 `3x3` 按鈕切換到網格檢視。同時查看多張圖像，確認和編輯標籤。

### 4. 詳細自定義
-   **標籤生成調整**: 使用滑塊直觀地調整 `general` 和 `character` 類別的標籤生成**閾值**和**最大標籤數**。
-   **自動設定保存**: 視窗大小和所有設定在應用程式退出時自動保存到 `config.ini`。下次在相同的環境中繼續工作。
-   **多語言支援**: 支援日語和英語。使用者介面根據作業系統語言設定自動切換。

## 授權 (License)

本專案根據 **LGPLv3** 和 **Apache License 2.0** 授權發布。

## 致謝 (Acknowledgements)

- 此工具利用了 [PixAI](https://pixai.art/) 訓練的優秀標註模型。如果沒有 Pixai Tagger 的公開發布，此應用程式將無法誕生。衷心感謝。
- 使用的 ONNX 模型由 [deepghs](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) 在 Hugging Face 上公開。謝謝。
