# OpenCV KCF 追蹤器
![https://github.com/Oliver0804/opencv_kcf/blob/main/pic/demo.gif](https://github.com/Oliver0804/opencv_kcf/blob/main/pic/demo.gif)

這個專案展示了如何使用 OpenCV 的 KCF 追蹤器來追蹤目標。程式會在視窗中心以及追蹤區域的中心繪製一條線，並顯示這條線的長度以及視窗中心與追蹤區域中心的 x 與 y 軸的差值。



## 功能

1. 使用 OpenCV 的 KCF 追蹤器追蹤目標。
2. 在視窗中心與追蹤區域的中心繪製一條線。
3. 顯示視窗中心與追蹤區域中心的 x 與 y 軸的差值。
4. 顯示視窗中心與追蹤區域中心連線的長度。

## 使用方式

以下是此專案的安裝與使用步驟：

1. 下載此專案的源碼，你可以直接下載壓縮檔，或是在終端機使用 `git clone` 指令克隆此專案。

2. 安裝所需的 Python 套件。在專案的根目錄中，開啟終端機並執行以下指令：

    ```
    pip install -r requirements.txt
    ```

    這個指令會安裝 `requirements.txt` 檔案中列出的所有 Python 套件。

3. 執行主程式。在終端機中，執行以下指令：

    ```
    python main.py
    ```

    這個指令會啟動主程式:
    
    使用按鍵s進行繪製關注區域 enter確認後進行KCF追蹤

    使用按鍵q離開程序
    
