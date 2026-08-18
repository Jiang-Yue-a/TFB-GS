import torch
import pandas as pd
from PIL import Image
import numpy as np
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0" 

torch.manual_seed(1234)

work='test'
TSV_name='multimodal_validate.tsv'
json_file_name = 'data/weibo/test.json'

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


'''
# Read TSV file
data = pd.read_csv('data/Fakeddit-master/multimodal_only_samples/'+TSV_name, sep='\t')
results = []
batch_size = 100

for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]

    for index, row in batch.iterrows():
        image_name = row['id']
        image_path = f'data/Fakeddit-master/multimodal_only_samples/{work}/{image_name}.jpg'
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue'''
# 读取 JSON 文件
with open(json_file_name, 'r', encoding='utf-8') as file:
    data = json.load(file)
count=0
count_0 = 0
count_1 = 0
results = []
batch_size = 50

for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]

    for item in batch:
        image_names = item['image_name']
        if isinstance(image_names, str):
            image_paths = [f'data/weibo/image/{image_names}']
        else:
            image_paths = [f'data/weibo/image/{image_name}' for image_name in image_names]
        #print("image_paths",image_paths)
        valid_image_paths = [path for path in image_paths if os.path.exists(path)]

        if not valid_image_paths:
            print(f"Warning: No valid images found for i{image_paths}")
            continue
        
        
        #print(image_paths)
        print(item['description'])
        structured_prompt = (
            f'''The caption provided for the image is '{item['description']}'. Please analyze both the image and the caption, '''
            '''and then provide a detailed textual description of what happened throughout when considering the text and images. '''
            '''After that, please rate the relevance between the image and the caption on a scale from 1 to 10, where 10 means they tell the same story, '''
            '''and 1 means they are not related at all. Only give me the number for the score. '''
            '''Finally, evaluate how effectively the content is conveyed by both the image and the text. '''
            '''Use the following format for your answer: '''
            '''{"description": "{Textual description}", "relevance_score": {X}, "effectiveness_score": {Y}}. '''
            '''The "{Textual description}" should provide a clear and concise summary of the content.'''
            '''X should be a number between 1 and 10 indicating relevance, and Y should be a number between 1 and 5 '''
            '''indicating effectiveness (1: both effective, 2: neither effective, 3: image effective, 4: text effective, 5: can be improved).'''
        )
        
        query_structured = tokenizer.from_list_format([{'image': path} for path in valid_image_paths] + [{'text': structured_prompt}])
   
        max_attempts = 3  
        attempts = 0
        parsed_response = None
        
        while attempts < max_attempts and parsed_response is None:
            try:
                
                response, _ = model.chat(tokenizer, query=query_structured, history=None)
                
                # 尝试解析为 JSON
                try:
                    parsed_response = eval(response)
                    print(attempts,"***",parsed_response)
                    if isinstance(parsed_response, tuple):
                        parsed_response=parsed_response[0]
                    print(parsed_response["description"]=='Textual description')
                    if parsed_response["description"]=='Textual description':
                        parsed_response=None
                        query_structured = tokenizer.from_list_format([{'image': path} for path in valid_image_paths] + [{'text': structured_prompt}])
                        continue
                    break
                except Exception as e:
                    print(f"Failed to parse JSON: {e}")
                    print(f"Response content: {response}")
                    
                    feedback_prompt = (
                        f'''Your response did not follow the required JSON format. Please make sure to use the following format: '''
                        '''{"description": "Textual description", "relevance_score": X, "effectiveness_score": Y}. '''
                        '''X should be a number between 1 and 10, and Y should be a number between 1 and 5. '''
                        '''Please try again.'''
                    )
                    query_structured = tokenizer.from_list_format([{'image': path} for path in valid_image_paths] + [{'text': feedback_prompt}])
                    attempts += 1

            except Exception as e:
                print(f"Error processing with image paths {valid_image_paths}: {e}")
                break
        
        if parsed_response is None:
            parsed_response = {
                "description": item["description"],
                "relevance_score": 5,
                "effectiveness_score": 3
            }
        
        if isinstance(parsed_response, tuple):
            parsed_response=parsed_response[0]
        

        print("description:", parsed_response.get("description",""))
        print("relevance_score:", parsed_response.get("relevance_score","0"))
        print("effectiveness_score:", parsed_response.get("effectiveness_score","0"))

        count += 1
        if item['2_way_label'] == 0:
            count_0 += 1
        elif item['2_way_label'] == 1:
            count_1 += 1
        result = {
            "image_name": image_names,
            "caption": item['description'],
            "description": parsed_response.get("description",""),
            "2_way_label": item['2_way_label'],
            "relevance_score": parsed_response.get("relevance_score","0"),
            "effectiveness_score": parsed_response.get("effectiveness_score","0")
        }
        results.append(result)

    del batch
    gc.collect()

# Save results to JSON
json_name=work+'_llm_response.json'
with open('data/weibo/'+json_name, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("处理完成！")
print(f"总共有 {count} 个条目")
print(f"2_way_label为0的数量: {count_0}")
print(f"2_way_label为1的数量: {count_1}")