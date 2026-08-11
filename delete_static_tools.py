import os

keep = {
    '__init__.py', 
    'code_interpreter.py', 
    'recon_engine.py', 
    'web_search.py', 
    'assimilation_engine.py', 
    'raw_weight_extractor.py', 
    'memory_retriever.py', 
    'model_builder.py'
}

tools_dir = r"d:\project\genAI\src\tools"
for filename in os.listdir(tools_dir):
    if filename.endswith(".py") and filename not in keep:
        filepath = os.path.join(tools_dir, filename)
        try:
            os.remove(filepath)
            print(f"Deleted static tool: {filename}")
        except Exception as e:
            print(f"Failed to delete {filename}: {e}")
