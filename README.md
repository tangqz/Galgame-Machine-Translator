# Galgame-Machine-Translator
使用 Deepseek-v3 和 OpenAI 融合实现多线程、全自动化的Galgame剧本文件翻译脚本

## 特性 
- 角色名的全文统一
- 多线程翻译，可提高翻译速度上百倍
- 在模型出错时自动重译
- 精细的日志记录，便于问题排查
- 控制台实时显示各线程翻译进度条和警告信息

## 使用方法

1. 阅读 `trans-v3-MT.py` ，使用 `pip` 安装缺失依赖
2. 使用 [VNTranslationTools](https://github.com/arcusmaximus/VNTranslationTools) 提取剧本文件，将提取出的 `.json` 文件置于一个文件夹中
3. 在 `trans-v3-MT.py` 中填写输入输出目录和 **Deepseek** 与 **OpenAI** 的 API Key
4. 运行脚本
5. 再次使用 VNT 将翻译后的文本封回。

## 注意事项

- 绝大部分情况下，使用 Deepseek-v3 模型翻译。当一次请求的自动重试次数≥3，则会换用 OpenAI API。
- LLM 模型始终存在不稳定性。因此翻译过程中仍有极小概率翻译失败。如遇这种情况，请根据脚本的运行目录下自动保存的日志文件进行问题排查。
- 请根据需要自行调整线程数。默认参数如下：

```python
max_file_workers = min(16, os.cpu_count() or 1)
max_batch_workers = min(64, os.cpu_count() * 8 or 1)
```
