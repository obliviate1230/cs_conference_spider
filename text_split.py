def split_text(text, max_length=1000):
    # 如果文本长度小于最大长度，直接返回
    if len(text) <= max_length:
        return [text]
    
    # 存储分割后的文本
    result = []
    
    # 当还有文本需要处理时继续循环
    while text:
        # 如果剩余文本小于最大长度，直接添加并结束
        if len(text) <= max_length:
            result.append(text)
            break
            
        # 在最大长度位置截取
        split_point = max_length
        
        # 尝试在句子结束符处分割
        for end_char in ['. ', '! ', '? ']:
            last_sentence = text[:max_length].rfind(end_char)
            if last_sentence != -1:
                split_point = last_sentence + 2  # +2 to include the end char and space
                break
                
        # 如果找不到句子结束符，尝试在空格处分割
        if split_point == max_length:
            last_space = text[:max_length].rfind(' ')
            if last_space != -1:
                split_point = last_space + 1
        
        # 添加分割的文本
        result.append(text[:split_point].strip())
        
        # 更新剩余文本
        text = text[split_point:].strip()
    
    return result
if __name__ == "__main__":
    # 使用示例
    text = "Robot-assisted Endoscopic Submucosal Dissection (ESD) improves the surgical procedure by providing a more comprehensive view through advanced robotic instruments and bimanual operation, thereby enhancing dissection efficiency and accuracy. Accurate prediction of dissection trajectories is crucial for better decision-making, reducing intraoperative errors, and improving surgical training. Nevertheless, predicting these trajectories is challenging due to variable tumor margins and dynamic visual conditions. To address this issue, we create the ESD Trajectory and Confidence Map-based Safety Margin (ETSM) dataset with 1849 short clips, focusing on submucosal dissection with a dual-arm robotic system. We also introduce a framework that combines optimal dissection trajectory prediction with a confidence map-based safety margin, providing a more secure and intelligent decision-making tool to minimize surgical risks for ESD procedures. Additionally, we propose the Regression-based Confidence Map Prediction Network (RCMNet), which utilizes a regression approach to predict confidence maps for dissection areas, thereby delineating various levels of safety margins. We evaluate our RCMNet using three distinct experimental setups: in-domain evaluation, robustness assessment, and out-of-domain evaluation. Experimental results show that our approach excels in the confidence map-based safety margin prediction task, achieving a mean absolute error (MAE) of only 3.18. To the best of our knowledge, this is the first study to apply a regression approach for visual guidance concerning delineating varying safety levels of dissection areas. Our approach bridges gaps in current research by improving prediction accuracy and enhancing the safety of the dissection process, showing great clinical significance in practice."
    chunks = split_text(text, max_length=1000)

    # 打印结果
    print(len(text))

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(chunk)
        print(f"Length: {len(chunk)}")
        print("---")