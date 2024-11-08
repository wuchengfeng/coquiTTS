from TTS.utils.manage import ModelManager

# 初始化 ModelManager 对象
model_manager = ModelManager()

# 获取可用模型列表
available_models = model_manager.list_models()
print(available_models)
